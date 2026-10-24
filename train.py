"""
Train the Hybrid NCF model on MovieLens 1M.

Usage:
  python train.py --data data/processed.pkl --epochs 20 --batch_size 1024

Saves:
  checkpoints/hybrid_ncf.pt        best model (lowest val loss)
  results/training_curve.csv       epoch-by-epoch train/val MSE
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(str(Path(__file__).parent))
from models.hybrid_ncf import HybridNCF


class RatingsDataset(Dataset):
    def __init__(self, ratings_arr: np.ndarray, genre_matrix: np.ndarray):
        # ratings_arr columns: [user_idx, movie_idx, rating, timestamp]
        self.users = torch.from_numpy(ratings_arr[:, 0].astype(np.int64))
        self.movies = torch.from_numpy(ratings_arr[:, 1].astype(np.int64))
        self.ratings = torch.from_numpy(ratings_arr[:, 2].astype(np.float32))
        self.genre_matrix = torch.from_numpy(genre_matrix)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        mid = self.movies[idx]
        return self.users[idx], mid, self.genre_matrix[mid], self.ratings[idx]


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n = 0
    for u, i, g, r in loader:
        u = u.to(device, non_blocking=True)
        i = i.to(device, non_blocking=True)
        g = g.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True)
        pred = model(u, i, g)
        loss = loss_fn(pred, r)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * u.size(0)
        n += u.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for u, i, g, r in loader:
        u = u.to(device, non_blocking=True)
        i = i.to(device, non_blocking=True)
        g = g.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True)
        pred = model(u, i, g)
        loss = loss_fn(pred, r)
        total_loss += loss.item() * u.size(0)
        n += u.size(0)
    return total_loss / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed.pkl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--gmf_dim", type=int, default=32)
    parser.add_argument("--mlp_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ckpt", type=str, default="checkpoints/hybrid_ncf.pt")
    parser.add_argument("--curve_csv", type=str, default="results/training_curve.csv")
    parser.add_argument("--patience", type=int, default=4,
                        help="Early stopping patience on val loss")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    if device.type == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}")

    # --- Load data ---
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    print(f"[train] Users={data['n_users']:,} Movies={data['n_movies']:,}  "
          f"Train={len(data['train']):,} Test={len(data['test']):,}")

    train_ds = RatingsDataset(data["train"], data["genre_matrix"])
    test_ds = RatingsDataset(data["test"], data["genre_matrix"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 4, shuffle=False,
                             num_workers=2, pin_memory=(device.type == "cuda"))

    # --- Model ---
    model = HybridNCF(
        n_users=data["n_users"],
        n_movies=data["n_movies"],
        n_genres=data["n_genres"],
        gmf_dim=args.gmf_dim,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    # --- Train loop ---
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.curve_csv).parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    no_improve = 0
    curve = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate_loss(model, test_loader, loss_fn, device)
        dt = time.time() - t0
        train_rmse = train_loss ** 0.5
        val_rmse = val_loss ** 0.5
        curve.append({
            "epoch": epoch,
            "train_mse": train_loss, "val_mse": val_loss,
            "train_rmse": train_rmse, "val_rmse": val_rmse,
            "time_sec": dt,
        })
        print(f"[epoch {epoch:3d}] train_mse={train_loss:.4f} train_rmse={train_rmse:.4f} "
              f"| val_mse={val_loss:.4f} val_rmse={val_rmse:.4f} | {dt:.1f}s")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "n_users": data["n_users"],
                "n_movies": data["n_movies"],
                "n_genres": data["n_genres"],
                "epoch": epoch,
                "val_mse": val_loss,
            }, args.ckpt)
            print(f"[train] ✓ Saved checkpoint (val_mse={val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[train] Early stopping (no improvement for {args.patience} epochs)")
                break

    pd.DataFrame(curve).to_csv(args.curve_csv, index=False)
    print(f"[train] Training curve -> {args.curve_csv}")
    print(f"[train] Best val MSE: {best_val:.4f}  (RMSE={best_val**0.5:.4f})")


if __name__ == "__main__":
    main()
