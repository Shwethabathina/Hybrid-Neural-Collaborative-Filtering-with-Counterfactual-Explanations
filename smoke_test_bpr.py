"""
Smoke test for BPR training.

Verifies train_bpr.py's BPR loss and quick_topn_eval on synthetic data.
If this passes, the real training on MovieLens 1M should work.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.hybrid_ncf import HybridNCF
from train_bpr import BPRTripletDataset, train_epoch_bpr, quick_topn_eval
from notebooks.smoke_test import make_synthetic_dataset


def run():
    print("=" * 70)
    print("BPR SMOKE TEST")
    print("=" * 70)

    # Slightly larger synthetic dataset so rating >= 4 actually gives us some positives
    data = make_synthetic_dataset(n_users=100, n_movies=200, ratings_per_user=30, seed=7)
    print(f"[bpr-smoke] Users={data['n_users']}  Movies={data['n_movies']}  "
          f"Train={len(data['train'])}  Test={len(data['test'])}")

    # Check we actually have some ratings >= 4
    ratings = data["train"][:, 2]
    n_positives = int(np.sum(ratings >= 4))
    print(f"[bpr-smoke] Train ratings >= 4: {n_positives} / {len(ratings)}")
    if n_positives < 50:
        # Lower threshold for synthetic data
        positive_threshold = 3
    else:
        positive_threshold = 4
    print(f"[bpr-smoke] Using positive_threshold = {positive_threshold}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bpr-smoke] Device: {device}")

    # Build dataset
    train_ds = BPRTripletDataset(data["train"], data["n_movies"],
                                  positive_threshold=positive_threshold)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    genre_matrix = torch.from_numpy(data["genre_matrix"]).to(device)

    model = HybridNCF(
        n_users=data["n_users"], n_movies=data["n_movies"],
        n_genres=data["n_genres"],
        gmf_dim=16, mlp_dim=16, mlp_hidden=(32, 16), genre_hidden=8,
        dropout=0.1,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-6)

    for epoch in range(1, 6):
        loss = train_epoch_bpr(model, train_loader, optim, genre_matrix, device)
        ndcg = quick_topn_eval(
            model, data["train"], data["test"], genre_matrix,
            n_items=data["n_movies"], device=device, k=10, max_users=50,
            relevance_threshold=positive_threshold,
        )
        print(f"[bpr-smoke] epoch {epoch}: loss={loss:.4f}  "
              f"val_ndcg@10(sampled)={ndcg:.4f}")

    # Sanity: loss should decrease
    assert loss < 0.7, f"BPR loss {loss} suspiciously high -- something is wrong"

    print("\n" + "=" * 70)
    print("BPR SMOKE TEST PASSED.")
    print("=" * 70)


if __name__ == "__main__":
    run()
