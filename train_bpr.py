"""
Train Hybrid-NCF with Bayesian Personalized Ranking (BPR) loss.

WHY THIS EXISTS:
The MSE-trained model in train.py optimizes for rating prediction accuracy
(MAE / RMSE). It does NOT optimize for ranking quality, and on MovieLens 1M
it loses to a popularity baseline on top-N metrics. BPR fixes that.

BPR is the standard approach for implicit-feedback recommenders. For each
(user, positive_item) training pair, we sample a random negative item the
user has NOT rated, and minimize:
    L = -log(sigmoid(score(u,pos) - score(u,neg)))
This directly trains the model to rank positives above negatives, which is
exactly what Precision@K / Recall@K / NDCG@K measure.

We reuse the same HybridNCF architecture -- only the loss and training
loop change. The output is no longer a calibrated rating in [1,5] but a
relative score suitable for ranking.

Usage:
    python train_bpr.py --data data/processed.pkl --epochs 20 \
        --ckpt checkpoints/hybrid_ncf_bpr.pt

Note on dataset filtering: classical BPR is for implicit feedback, so we
treat rating >= 4 as "positive" by default (most papers do). If you want
to treat ALL ratings as positive, pass --positive_threshold 1.
"""
import argparse
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(str(Path(__file__).parent))
from models.hybrid_ncf import HybridNCF


class BPRTripletDataset(Dataset):
    """
    Yields (user, pos_item, neg_item) triplets for BPR training.

    For each epoch, every user's positive items are seen once, each paired
    with a freshly-sampled negative. This is the standard BPR protocol.
    """
    def __init__(self, train_arr, n_items, positive_threshold=4):
        # Keep only positive interactions
        ratings = train_arr[:, 2].astype(np.float32)
        mask = ratings >= positive_threshold
        filtered = train_arr[mask]
        print(f"[bpr] Kept {len(filtered):,} / {len(train_arr):,} interactions "
              f"(rating >= {positive_threshold})")

        self.users = filtered[:, 0].astype(np.int64)
        self.pos_items = filtered[:, 1].astype(np.int64)
        self.n_items = n_items

        # Per-user positive item sets (for fast negative sampling)
        self.user_positives = defaultdict(set)
        for u, i in zip(self.users, self.pos_items):
            self.user_positives[int(u)].add(int(i))

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = int(self.users[idx])
        pos = int(self.pos_items[idx])
        # Sample a negative uniformly, rejecting items the user liked
        pos_set = self.user_positives[u]
        neg = np.random.randint(self.n_items)
        # Bounded retries; for MovieLens 1M, users have at most ~2000 positives
        # out of 3706 items, so this terminates quickly
        while neg in pos_set:
            neg = np.random.randint(self.n_items)
        return u, pos, neg


def train_epoch_bpr(model, loader, optimizer, genre_matrix, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for u, pos, neg in loader:
        u = u.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True)
        neg = neg.to(device, non_blocking=True)
        pos_g = genre_matrix[pos]
        neg_g = genre_matrix[neg]

        pos_score = model(u, pos, pos_g)
        neg_score = model(u, neg, neg_g)

        # BPR loss: -log sigmoid(pos - neg)
        loss = -F.logsigmoid(pos_score - neg_score).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def quick_topn_eval(model, train_arr, test_arr, genre_matrix, n_items, device,
                    k=10, max_users=500, relevance_threshold=4):
    """
    A fast mini-eval during training -- not full catalog scoring.
    Uses 500 users and scores against a sampled pool for speed.
    Returns NDCG@10 for early stopping.
    """
    model.eval()
    # Build per-user train/test sets
    train_by_user = defaultdict(set)
    for u, i, _, _ in train_arr:
        train_by_user[int(u)].add(int(i))
    test_by_user = defaultdict(list)
    for u, i, r, _ in test_arr:
        test_by_user[int(u)].append((int(i), float(r)))

    users = list(test_by_user.keys())
    np.random.shuffle(users)
    users = users[:max_users]

    genre_t = genre_matrix.to(device)
    items_full = torch.arange(n_items, dtype=torch.long, device=device)

    ndcgs = []
    for uid in users:
        relevant = {i for i, r in test_by_user[uid] if r >= relevance_threshold}
        if not relevant:
            continue
        u_tensor = torch.full((n_items,), uid, dtype=torch.long, device=device)
        scores = model(u_tensor, items_full, genre_t).cpu().numpy()
        # Mask seen
        seen = train_by_user[uid]
        if seen:
            scores[list(seen)] = -np.inf
        topk = np.argsort(-scores)[:k]
        dcg = 0.0
        for rank, item in enumerate(topk):
            if item in relevant:
                dcg += 1.0 / np.log2(rank + 2)
        ideal = min(k, len(relevant))
        idcg = sum(1.0 / np.log2(j + 2) for j in range(ideal))
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed.pkl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--gmf_dim", type=int, default=32)
    parser.add_argument("--mlp_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--positive_threshold", type=float, default=4.0,
                        help="Ratings >= this value count as positive")
    parser.add_argument("--ckpt", type=str, default="checkpoints/hybrid_ncf_bpr.pt")
    parser.add_argument("--curve_csv", type=str,
                        default="results/training_curve_bpr.csv")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bpr] Device: {device}")
    if device.type == "cuda":
        print(f"[bpr] GPU: {torch.cuda.get_device_name(0)}")

    with open(args.data, "rb") as f:
        data = pickle.load(f)
    print(f"[bpr] Users={data['n_users']:,} Movies={data['n_movies']:,}")

    genre_matrix = torch.from_numpy(data["genre_matrix"]).to(device)

    train_ds = BPRTripletDataset(
        data["train"], data["n_movies"],
        positive_threshold=args.positive_threshold,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    model = HybridNCF(
        n_users=data["n_users"],
        n_movies=data["n_movies"],
        n_genres=data["n_genres"],
        gmf_dim=args.gmf_dim,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bpr] Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.curve_csv).parent.mkdir(parents=True, exist_ok=True)

    best_ndcg = -1.0
    no_improve = 0
    curve = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch_bpr(model, train_loader, optimizer,
                                      genre_matrix, device)
        val_ndcg = quick_topn_eval(
            model, data["train"], data["test"], genre_matrix,
            n_items=data["n_movies"], device=device, k=10, max_users=500,
        )
        dt = time.time() - t0
        curve.append({
            "epoch": epoch, "train_bpr_loss": train_loss,
            "val_ndcg10_sampled": val_ndcg, "time_sec": dt,
        })
        print(f"[epoch {epoch:3d}] bpr_loss={train_loss:.4f} "
              f"| val_ndcg@10 (500 users) = {val_ndcg:.4f} | {dt:.1f}s")

        if val_ndcg > best_ndcg + 1e-5:
            best_ndcg = val_ndcg
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "n_users": data["n_users"],
                "n_movies": data["n_movies"],
                "n_genres": data["n_genres"],
                "epoch": epoch,
                "val_ndcg10_sampled": val_ndcg,
                "loss_type": "bpr",
            }, args.ckpt)
            print(f"[bpr] ✓ Saved checkpoint (val_ndcg@10={val_ndcg:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[bpr] Early stopping (no improvement for {args.patience} epochs)")
                break

    pd.DataFrame(curve).to_csv(args.curve_csv, index=False)
    print(f"[bpr] Training curve -> {args.curve_csv}")
    print(f"[bpr] Best val NDCG@10: {best_ndcg:.4f}")


if __name__ == "__main__":
    main()
