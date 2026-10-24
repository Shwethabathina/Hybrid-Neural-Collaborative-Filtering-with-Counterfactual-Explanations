"""
Recommendation generator -- produces top-K item recommendations for a user.

Used by the demo. The evaluation script does its own top-N computation
internally, but this module exposes a clean API for interactive use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch


@dataclass
class Recommendation:
    rank: int
    movie_idx: int
    title: str
    genres: str
    predicted_rating: float


class Recommender:
    def __init__(self, model, data: dict, device: torch.device):
        self.model = model
        self.model.eval()
        self.device = device
        self.data = data
        self.n_movies = data["n_movies"]
        self.movie_title = data["movie_title"]
        self.movie_genres_str = data["movie_genres_str"]
        self.genre_matrix = torch.from_numpy(data["genre_matrix"]).to(device)

        # Build user history from train+test -- for masking we only want train
        self.train_seen = {}
        for row in data["train"]:
            u, m, _, _ = row
            self.train_seen.setdefault(int(u), set()).add(int(m))

        self.test_items = {}
        for row in data["test"]:
            u, m, r, _ = row
            self.test_items.setdefault(int(u), []).append((int(m), float(r)))

    @torch.no_grad()
    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[Recommendation]:
        scores = self.model.score_all_items(
            user_idx, self.genre_matrix, device=self.device
        ).numpy()
        if exclude_seen:
            seen = self.train_seen.get(user_idx, set())
            if seen:
                scores[list(seen)] = -np.inf

        topk = np.argsort(-scores)[:k]
        recs = []
        for rank, m in enumerate(topk, start=1):
            recs.append(Recommendation(
                rank=rank,
                movie_idx=int(m),
                title=self.movie_title.get(int(m), f"movie {int(m)}"),
                genres=self.movie_genres_str.get(int(m), ""),
                predicted_rating=float(np.clip(scores[m], 1.0, 5.0)),
            ))
        return recs

    def get_user_history(self, user_idx: int, top_rated_first: bool = True):
        """Return the user's training history with titles/ratings."""
        hist_pairs = []
        for row in self.data["train"]:
            u, m, r, _ = row
            if int(u) == user_idx:
                hist_pairs.append((int(m), float(r)))
        if top_rated_first:
            hist_pairs.sort(key=lambda x: -x[1])
        return [
            {
                "movie_idx": m,
                "title": self.movie_title.get(m, f"movie {m}"),
                "genres": self.movie_genres_str.get(m, ""),
                "rating": r,
            }
            for m, r in hist_pairs
        ]

    def get_test_items(self, user_idx: int):
        """Return the user's holdout test items (used for evaluation display)."""
        items = self.test_items.get(user_idx, [])
        return [
            {
                "movie_idx": m,
                "title": self.movie_title.get(m, f"movie {m}"),
                "genres": self.movie_genres_str.get(m, ""),
                "rating": r,
            }
            for m, r in sorted(items, key=lambda x: -x[1])
        ]
