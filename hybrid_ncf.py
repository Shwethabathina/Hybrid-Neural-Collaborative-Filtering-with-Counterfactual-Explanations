"""
Hybrid Neural Collaborative Filtering (Hybrid-NCF).

Architecture:
  - GMF branch:   element-wise product of user/item embeddings (linear interaction)
  - MLP branch:   concat(user_emb, item_emb) -> MLP (non-linear interaction)
  - Genre tower: MLP over multi-hot genre vector (content signal)
  - Fusion:       concat(GMF, MLP_out, Genre_out) -> linear -> predicted rating

The genre tower is what makes this 'hybrid' and distinguishes it from vanilla NCF.
We predict ratings directly (regression against 1-5 stars) trained with MSE;
this lets us reuse the same model for both rating prediction (MAE/RMSE)
and top-N ranking (Precision/Recall/F1/NDCG).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridNCF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_movies: int,
        n_genres: int,
        gmf_dim: int = 32,
        mlp_dim: int = 32,
        mlp_hidden=(128, 64, 32),
        genre_hidden: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        # --- GMF embeddings (linear interaction) ---
        self.user_emb_gmf = nn.Embedding(n_users, gmf_dim)
        self.item_emb_gmf = nn.Embedding(n_movies, gmf_dim)

        # --- MLP embeddings (non-linear interaction) ---
        self.user_emb_mlp = nn.Embedding(n_users, mlp_dim)
        self.item_emb_mlp = nn.Embedding(n_movies, mlp_dim)

        # --- MLP tower ---
        mlp_layers = []
        in_dim = mlp_dim * 2
        for h in mlp_hidden:
            mlp_layers.append(nn.Linear(in_dim, h))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = h
        self.mlp = nn.Sequential(*mlp_layers)

        # --- Genre tower (the 'hybrid' part) ---
        self.genre_tower = nn.Sequential(
            nn.Linear(n_genres, genre_hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(genre_hidden * 2, genre_hidden),
            nn.ReLU(),
        )

        # --- Fusion + output head ---
        fusion_in = gmf_dim + mlp_hidden[-1] + genre_hidden
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for emb in [self.user_emb_gmf, self.item_emb_gmf,
                    self.user_emb_mlp, self.item_emb_mlp]:
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_idx, item_idx, genre_vec):
        """
        user_idx: (B,) long
        item_idx: (B,) long
        genre_vec: (B, n_genres) float
        returns: (B,) predicted rating (unclamped; clamp to [1,5] at inference)
        """
        # GMF branch: element-wise product
        u_gmf = self.user_emb_gmf(user_idx)
        i_gmf = self.item_emb_gmf(item_idx)
        gmf = u_gmf * i_gmf  # (B, gmf_dim)

        # MLP branch
        u_mlp = self.user_emb_mlp(user_idx)
        i_mlp = self.item_emb_mlp(item_idx)
        mlp_in = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # Genre branch
        genre_out = self.genre_tower(genre_vec)

        # Fuse
        fused = torch.cat([gmf, mlp_out, genre_out], dim=-1)
        pred = self.head(fused).squeeze(-1)
        return pred

    @torch.no_grad()
    def score_all_items(self, user_idx: int, genre_matrix: torch.Tensor,
                        batch_size: int = 4096, device=None):
        """
        Score every item for a single user. Used for top-N generation.
          genre_matrix: (n_movies, n_genres) float tensor on device
        returns: (n_movies,) predicted ratings tensor on CPU
        """
        self.eval()
        device = device or next(self.parameters()).device
        n_movies = genre_matrix.shape[0]
        u = torch.full((n_movies,), user_idx, dtype=torch.long, device=device)
        items = torch.arange(n_movies, dtype=torch.long, device=device)
        out = torch.empty(n_movies, dtype=torch.float32)
        for start in range(0, n_movies, batch_size):
            end = min(start + batch_size, n_movies)
            scores = self.forward(u[start:end], items[start:end],
                                  genre_matrix[start:end])
            out[start:end] = scores.detach().cpu()
        return out
