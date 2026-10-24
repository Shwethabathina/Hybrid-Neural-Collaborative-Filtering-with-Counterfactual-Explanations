"""
End-to-end smoke test using a tiny SYNTHETIC MovieLens-shaped dataset.

Why this exists: before scaling to 1M ratings on Amarel, we want to
verify that preprocessing -> train -> eval -> explain ALL work together.
If they do, scaling up is just a bigger hammer.

This script:
  1. Generates a tiny synthetic (users, movies, ratings) setup
  2. Saves it in the same format preprocess.py would produce
  3. Runs a very short training run
  4. Runs full evaluation (MAE, RMSE, P@K, R@K, F1@K, NDCG@K)
  5. Runs the counterfactual explainer on one recommendation
  6. Prints everything

Run with:
    cd cs550_project
    python notebooks/smoke_test.py
"""
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make repo importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.hybrid_ncf import HybridNCF
from evaluation.evaluate import compute_rating_metrics, compute_topn_metrics
from explain.counterfactual import CounterfactualExplainer
from explain.recommend import Recommender
from train import RatingsDataset, train_one_epoch, evaluate_loss


GENRE_LIST = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def make_synthetic_dataset(n_users=100, n_movies=200, ratings_per_user=25, seed=42):
    """Create a tiny but realistic MovieLens-shaped payload."""
    rng = np.random.RandomState(seed)

    # Random genre assignments (1-3 genres per movie)
    n_genres = len(GENRE_LIST)
    genre_matrix = np.zeros((n_movies, n_genres), dtype=np.float32)
    movie_title = {}
    movie_genres_str = {}
    for m in range(n_movies):
        k = rng.randint(1, 4)
        genres = rng.choice(n_genres, size=k, replace=False)
        genre_matrix[m, genres] = 1.0
        movie_title[m] = f"Synth Movie {m+1}"
        movie_genres_str[m] = "|".join(GENRE_LIST[g] for g in genres)

    # Each user has a latent "taste" vector; rating = similarity + noise, clipped to 1-5
    user_taste = rng.randn(n_users, n_genres) * 0.5
    rows = []
    timestamp = 1_000_000_000
    for u in range(n_users):
        # sample unique movies for this user
        movies = rng.choice(n_movies, size=ratings_per_user, replace=False)
        for m in movies:
            aff = float(user_taste[u] @ genre_matrix[m])
            noise = rng.randn() * 0.5
            r = aff + noise + 3.0  # center around 3
            r = int(np.clip(round(r), 1, 5))
            rows.append([u, m, r, timestamp])
            timestamp += 1

    all_arr = np.array(rows, dtype=np.int64)
    # per-user 80/20 split
    train_rows, test_rows = [], []
    for u in range(n_users):
        mask = all_arr[:, 0] == u
        user_rows = all_arr[mask]
        idx = np.arange(len(user_rows))
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * 0.2))
        test_rows.append(user_rows[idx[:n_test]])
        train_rows.append(user_rows[idx[n_test:]])
    train = np.concatenate(train_rows)
    test = np.concatenate(test_rows)

    payload = {
        "train": train,
        "test": test,
        "n_users": n_users,
        "n_movies": n_movies,
        "n_genres": n_genres,
        "genre_list": GENRE_LIST,
        "genre_matrix": genre_matrix,
        "movie_title": movie_title,
        "movie_genres_str": movie_genres_str,
        "user2idx": {u: u for u in range(n_users)},
        "movie2idx": {m: m for m in range(n_movies)},
        "idx2user": {u: u for u in range(n_users)},
        "idx2movie": {m: m for m in range(n_movies)},
    }
    return payload


def run():
    print("=" * 70)
    print("SMOKE TEST: end-to-end pipeline on synthetic data")
    print("=" * 70)

    data = make_synthetic_dataset(n_users=80, n_movies=150, ratings_per_user=20)
    print(f"[smoke] Users={data['n_users']}  Movies={data['n_movies']}  "
          f"Train={len(data['train'])}  Test={len(data['test'])}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] Device: {device}")

    # ---- Build model ----
    model = HybridNCF(
        n_users=data["n_users"],
        n_movies=data["n_movies"],
        n_genres=data["n_genres"],
        gmf_dim=16, mlp_dim=16, mlp_hidden=(32, 16), genre_hidden=8,
        dropout=0.1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] Model params: {n_params:,}")

    # ---- Train a few epochs ----
    train_ds = RatingsDataset(data["train"], data["genre_matrix"])
    test_ds = RatingsDataset(data["test"], data["genre_matrix"])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    for epoch in range(1, 6):
        tr = train_one_epoch(model, train_loader, optim, loss_fn, device)
        va = evaluate_loss(model, test_loader, loss_fn, device)
        print(f"[smoke] epoch {epoch}: train_mse={tr:.4f}  val_mse={va:.4f}")

    # ---- Rating metrics ----
    print("\n[smoke] Rating prediction metrics:")
    rating_m, _ = compute_rating_metrics(model, data["test"], data["genre_matrix"], device)
    for k, v in rating_m.items():
        print(f"  {k} = {v:.4f}")

    # ---- Top-N metrics ----
    print("\n[smoke] Top-10 recommendation metrics:")
    topn = compute_topn_metrics(
        model, data["train"], data["test"], data["genre_matrix"],
        n_movies=data["n_movies"], device=device, k=10, verbose=False,
    )
    for k, v in topn.items():
        if isinstance(v, float):
            print(f"  {k} = {v:.4f}")
        else:
            print(f"  {k} = {v}")

    # ---- Recommend & explain ----
    print("\n[smoke] Recommending and explaining for user 0:")
    rec = Recommender(model, data, device)
    recs = rec.recommend(user_idx=0, k=5)
    for r in recs:
        print(f"  #{r.rank}  {r.title:30s}  pred={r.predicted_rating:.2f}  [{r.genres}]")

    explainer = CounterfactualExplainer(model, data, device)
    exp = explainer.explain(user_idx=0, target_movie_idx=recs[0].movie_idx)
    print("\n[smoke] Explanation:")
    print("  " + exp.summary())
    print("\n[smoke] Top causes:")
    for c in exp.cause_items:
        print(f"  - {c.title}  rating={c.user_rating:.1f}  Δ={c.delta_score:+.4f}")
    print("\n[smoke] Alternatives if causes were removed:")
    for a in exp.alternatives:
        print(f"  - #{a.new_rank}  {a.title}  new_score={a.new_score:.3f}")
    print("\n[smoke] Top genres for target:")
    for g, w in exp.top_genres:
        print(f"  - {g}  weight={w:.2%}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED.")
    print("=" * 70)


if __name__ == "__main__":
    run()
