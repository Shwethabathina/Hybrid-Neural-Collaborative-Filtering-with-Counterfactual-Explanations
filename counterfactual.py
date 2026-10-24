"""
Counterfactual Explanations for Hybrid-NCF.

Inspired by: Tan et al., "Counterfactual Explainable Recommendation"
             (arxiv 2108.10539) -- cited in the project description.

The core idea of counterfactual explanation for recommender systems:
    "We recommended item X because you rated these specific past items
     highly. If we REMOVED those items from your history, we would
     recommend something else."

Because our Hybrid-NCF uses a single learned user embedding rather than
an explicit attention-over-history, we implement counterfactual
explanations by *perturbing* the user embedding:

    Step 1. Identify the user's top-rated training items.
    Step 2. For each candidate "cause" item c:
              - Rebuild the user's profile as if item c were never rated
              - Re-score the recommendation target
              - Record the change in predicted score (delta_c)
    Step 3. Items with largest drop in score for the target are the
            counterfactual causes.
    Step 4. Additionally, rank alternative recommendations under the
            ablated profile -- these are the "what you would have been
            recommended instead" items.

Profile perturbation is done by taking a weighted average of the user's
item embeddings (gmf_emb + mlp_emb) minus the removed item. The result
is passed to the prediction head as a synthetic user representation.
This is an approximation (the learned user embedding is not literally
the average of its items), but it's a well-known technique and works
well in practice for explanation purposes.

We additionally provide GENRE-LEVEL explanations -- which genres in the
recommended item most drove the prediction -- using a gradient-based
attribution on the genre tower input.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# Data classes for readable outputs
# =============================================================================

@dataclass
class CauseItem:
    movie_idx: int
    title: str
    user_rating: float
    delta_score: float  # how much the target's predicted score drops if this is removed

    def __repr__(self):
        return (f"<CauseItem {self.title!r} "
                f"(rated {self.user_rating:.1f}, Δ={self.delta_score:+.3f})>")


@dataclass
class AlternativeItem:
    movie_idx: int
    title: str
    original_rank: Optional[int]
    new_rank: int
    new_score: float


@dataclass
class Explanation:
    user_idx: int
    target_movie_idx: int
    target_title: str
    predicted_score: float
    cause_items: List[CauseItem] = field(default_factory=list)
    alternatives: List[AlternativeItem] = field(default_factory=list)
    top_genres: List[tuple] = field(default_factory=list)  # [(genre_name, weight), ...]

    def summary(self) -> str:
        """Human-readable one-paragraph summary."""
        lines = []
        lines.append(
            f"Recommended '{self.target_title}' to user {self.user_idx} "
            f"(predicted rating: {self.predicted_score:.2f}/5.0)."
        )
        if self.cause_items:
            causes = ", ".join(
                f"'{c.title}' (you rated {c.user_rating:.0f})"
                for c in self.cause_items[:3]
            )
            lines.append(
                f"This was driven primarily by your high ratings of: {causes}."
            )
        if self.top_genres:
            genres = ", ".join(f"{g} ({w:.0%})" for g, w in self.top_genres[:3])
            lines.append(f"Key genre signals: {genres}.")
        if self.alternatives:
            alts = ", ".join(
                f"'{a.title}'" for a in self.alternatives[:3]
            )
            lines.append(
                f"Counterfactual: if the top causes were removed, "
                f"we would instead recommend: {alts}."
            )
        return " ".join(lines)


# =============================================================================
# Explainer
# =============================================================================

class CounterfactualExplainer:
    """
    Explainer for a trained HybridNCF model.

    Usage:
        explainer = CounterfactualExplainer(model, data, device)
        exp = explainer.explain(user_idx=42, target_movie_idx=1234)
        print(exp.summary())
    """

    def __init__(self, model, data: dict, device: torch.device):
        self.model = model
        self.model.eval()
        self.device = device
        self.data = data
        self.n_movies = data["n_movies"]
        self.n_genres = data["n_genres"]
        self.genre_list = data["genre_list"]
        self.movie_title = data["movie_title"]
        self.genre_matrix_np = data["genre_matrix"]
        self.genre_matrix = torch.from_numpy(self.genre_matrix_np).to(device)

        # Build user -> [(movie_idx, rating)] from train set
        self.user_history = {}
        train = data["train"]
        for row in train:
            u, m, r, _ = row
            u, m = int(u), int(m)
            r = float(r)
            self.user_history.setdefault(u, []).append((m, r))

    @torch.no_grad()
    def _score_pair(self, user_idx: int, movie_idx: int) -> float:
        """Score a single (user, movie) pair with the real learned embedding."""
        u = torch.tensor([user_idx], dtype=torch.long, device=self.device)
        i = torch.tensor([movie_idx], dtype=torch.long, device=self.device)
        g = self.genre_matrix[movie_idx].unsqueeze(0)
        return float(self.model(u, i, g).item())

    @torch.no_grad()
    def _score_all_for_user(self, user_idx: int) -> np.ndarray:
        """Score every item in the catalog for this user."""
        scores = self.model.score_all_items(
            user_idx, self.genre_matrix, device=self.device
        )
        return scores.numpy()

    @torch.no_grad()
    def _score_with_perturbed_user(
        self, user_idx: int, movie_idx: int, removed_movie_idx: int
    ) -> float:
        """
        Re-score (user, movie) under a *perturbed* user profile that
        averages the learned user embedding with a subtraction of the
        removed item's embedding.

        Concretely, we compute:
            u_tilde_gmf = u_gmf - alpha * item_gmf(removed)
            u_tilde_mlp = u_mlp - alpha * item_mlp(removed)
        and feed those through the rest of the network with the target item.

        alpha is chosen so that removing an item the user rated highly
        meaningfully changes the prediction. We use alpha = 1/|history|,
        i.e. the "average contribution" of one history item.
        """
        model = self.model
        history = self.user_history.get(user_idx, [])
        alpha = 1.0 / max(len(history), 1)

        u = torch.tensor([user_idx], dtype=torch.long, device=self.device)
        i = torch.tensor([movie_idx], dtype=torch.long, device=self.device)
        rem = torch.tensor([removed_movie_idx], dtype=torch.long, device=self.device)

        # Original embeddings
        u_gmf = model.user_emb_gmf(u)
        u_mlp = model.user_emb_mlp(u)
        i_gmf = model.item_emb_gmf(i)
        i_mlp = model.item_emb_mlp(i)
        rem_gmf = model.item_emb_gmf(rem)
        rem_mlp = model.item_emb_mlp(rem)

        # Perturb user representation
        u_gmf_pert = u_gmf - alpha * rem_gmf
        u_mlp_pert = u_mlp - alpha * rem_mlp

        gmf = u_gmf_pert * i_gmf
        mlp_in = torch.cat([u_mlp_pert, i_mlp], dim=-1)
        mlp_out = model.mlp(mlp_in)
        g_vec = self.genre_matrix[movie_idx].unsqueeze(0)
        genre_out = model.genre_tower(g_vec)
        fused = torch.cat([gmf, mlp_out, genre_out], dim=-1)
        score = model.head(fused).squeeze(-1)
        return float(score.item())

    def _top_genres_for_target(self, movie_idx: int, top_k: int = 3) -> List[tuple]:
        """
        Gradient-based attribution: which genre input dims most affect the
        output for this item? We do a single-input forward with a fake
        'average' user, and take gradients of the output w.r.t. the
        genre-vector input.
        """
        g_vec = self.genre_matrix[movie_idx].detach().clone().unsqueeze(0)
        g_vec.requires_grad_(True)

        # Run the genre tower in isolation (since we want the local effect
        # of the genre signal on the final head)
        genre_out = self.model.genre_tower(g_vec)
        # Sum the output -- we just want which input dims influence it
        genre_out.sum().backward()
        grads = g_vec.grad.detach().cpu().numpy().flatten()
        # Only look at genres actually present in this movie
        present = (self.genre_matrix_np[movie_idx] > 0.5)
        contributions = grads * present
        ranked = np.argsort(-np.abs(contributions))
        out = []
        for idx in ranked:
            if present[idx]:
                out.append((self.genre_list[idx], float(np.abs(contributions[idx]))))
            if len(out) >= top_k:
                break
        # Normalize weights to percentages
        total = sum(w for _, w in out) or 1.0
        out = [(g, w / total) for g, w in out]
        return out

    def explain(
        self,
        user_idx: int,
        target_movie_idx: int,
        n_causes: int = 3,
        n_alternatives: int = 3,
        candidate_pool_size: int = 20,
    ) -> Explanation:
        """
        Generate a counterfactual explanation for a single recommendation.

        n_causes:            how many history items to report as causes
        n_alternatives:      how many "what you'd see instead" items to report
        candidate_pool_size: consider only the top-rated items from the user's
                             history as counterfactual causes (bigger = slower
                             but more thorough)
        """
        history = self.user_history.get(user_idx, [])
        if not history:
            return Explanation(
                user_idx=user_idx,
                target_movie_idx=target_movie_idx,
                target_title=self.movie_title.get(target_movie_idx, "unknown"),
                predicted_score=self._score_pair(user_idx, target_movie_idx),
            )

        # Candidate causes: user's highest-rated history items
        sorted_history = sorted(history, key=lambda x: -x[1])[:candidate_pool_size]

        # Baseline score
        base_score = self._score_pair(user_idx, target_movie_idx)

        # Score-drop per candidate cause
        cause_candidates = []
        for cand_movie, cand_rating in sorted_history:
            perturbed = self._score_with_perturbed_user(
                user_idx, target_movie_idx, cand_movie
            )
            delta = base_score - perturbed  # positive delta => cause
            cause_candidates.append((cand_movie, cand_rating, delta))

        # Sort by positive delta (largest drop when removed = biggest cause)
        cause_candidates.sort(key=lambda x: -x[2])
        top_causes = [
            CauseItem(
                movie_idx=m,
                title=self.movie_title.get(m, f"movie {m}"),
                user_rating=r,
                delta_score=d,
            )
            for m, r, d in cause_candidates[:n_causes]
        ]

        # --- Alternatives: what would the top-K look like if the top causes
        #     were all removed simultaneously?
        # Build the ablated user profile once, score all items, compare.
        alternatives = self._ablated_top_n(
            user_idx, [c.movie_idx for c in top_causes], k=n_alternatives + 1,
            exclude_target=target_movie_idx,
        )

        # Genre attribution for target
        top_genres = self._top_genres_for_target(target_movie_idx)

        return Explanation(
            user_idx=user_idx,
            target_movie_idx=target_movie_idx,
            target_title=self.movie_title.get(target_movie_idx, f"movie {target_movie_idx}"),
            predicted_score=base_score,
            cause_items=top_causes,
            alternatives=alternatives,
            top_genres=top_genres,
        )

    @torch.no_grad()
    def _ablated_top_n(
        self,
        user_idx: int,
        removed_items: List[int],
        k: int,
        exclude_target: Optional[int] = None,
    ) -> List[AlternativeItem]:
        """
        Score all items with user profile ablated by `removed_items`,
        return the top-k (excluding items the user already rated + the
        original recommendation target).
        """
        model = self.model
        history = self.user_history.get(user_idx, [])
        alpha = 1.0 / max(len(history), 1)

        u = torch.tensor([user_idx], dtype=torch.long, device=self.device)
        u_gmf = model.user_emb_gmf(u)
        u_mlp = model.user_emb_mlp(u)
        if removed_items:
            rem_t = torch.tensor(removed_items, dtype=torch.long, device=self.device)
            rem_gmf_sum = model.item_emb_gmf(rem_t).sum(dim=0, keepdim=True)
            rem_mlp_sum = model.item_emb_mlp(rem_t).sum(dim=0, keepdim=True)
            u_gmf = u_gmf - alpha * rem_gmf_sum
            u_mlp = u_mlp - alpha * rem_mlp_sum

        # Score every item
        items_full = torch.arange(self.n_movies, dtype=torch.long, device=self.device)
        batch_size = 4096
        scores = np.empty(self.n_movies, dtype=np.float32)
        for start in range(0, self.n_movies, batch_size):
            end = min(start + batch_size, self.n_movies)
            i_b = items_full[start:end]
            bs = end - start
            u_gmf_rep = u_gmf.expand(bs, -1)
            u_mlp_rep = u_mlp.expand(bs, -1)
            i_gmf = model.item_emb_gmf(i_b)
            i_mlp = model.item_emb_mlp(i_b)
            gmf = u_gmf_rep * i_gmf
            mlp_in = torch.cat([u_mlp_rep, i_mlp], dim=-1)
            mlp_out = model.mlp(mlp_in)
            genre_out = model.genre_tower(self.genre_matrix[i_b])
            fused = torch.cat([gmf, mlp_out, genre_out], dim=-1)
            s = model.head(fused).squeeze(-1).cpu().numpy()
            scores[start:end] = s

        # Mask user's history (and the original target)
        seen = set(m for m, _ in history)
        if exclude_target is not None:
            seen.add(exclude_target)
        if seen:
            scores[list(seen)] = -np.inf

        topk = np.argsort(-scores)[:k]
        out = []
        for new_rank, m in enumerate(topk):
            out.append(AlternativeItem(
                movie_idx=int(m),
                title=self.movie_title.get(int(m), f"movie {int(m)}"),
                original_rank=None,  # we don't compute this explicitly; could be added
                new_rank=new_rank,
                new_score=float(scores[m]),
            ))
        return out
