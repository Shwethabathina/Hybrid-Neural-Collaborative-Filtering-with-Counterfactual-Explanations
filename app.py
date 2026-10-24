"""
Streamlit demo for the Hybrid-NCF Explainable Recommender.

Launches with:
    streamlit run demo/app.py -- --data data/processed.pkl --ckpt checkpoints/hybrid_ncf.pt

(note the double dash -- which separates streamlit args from our own)

Tabs:
  1. Recommendations      -- pick a user, see their top-10 picks
  2. Why This?            -- counterfactual explanation for any recommendation
  3. Your Profile         -- user's rating history and genre preferences
  4. Evaluation           -- aggregate metrics MAE/RMSE/P/R/F1/NDCG

Design philosophy: editorial, magazine-like. Serif display type, clean
data tables, restrained accent color. Built to look good on a projector.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.hybrid_ncf import HybridNCF
from explain.recommend import Recommender
from explain.counterfactual import CounterfactualExplainer


# =============================================================================
# Page config + styling
# =============================================================================

st.set_page_config(
    page_title="Hybrid-NCF | Explainable Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&family=JetBrains+Mono:wght@400;500&family=Inter+Tight:wght@400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter Tight', sans-serif;
  }
  h1, h2, h3, h4 {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
  }
  h1 {
    font-size: 2.6rem !important;
    letter-spacing: -0.03em;
  }
  .accent { color: #c8553d; }
  .muted  { color: #6b7280; }
  .kicker {
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.75rem;
    color: #6b7280;
    font-weight: 500;
  }
  /* metric cards */
  .metric-card {
    padding: 1.1rem 1.3rem;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    background: #fafaf7;
  }
  .metric-card .label {
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    color: #6b7280;
    font-family: 'JetBrains Mono', monospace;
  }
  .metric-card .value {
    font-family: 'Fraunces', serif;
    font-size: 2rem;
    font-weight: 600;
    color: #111827;
    line-height: 1.1;
    margin-top: 0.3rem;
  }
  /* rec card */
  .rec-card {
    padding: 0.9rem 1.1rem;
    border-left: 3px solid #c8553d;
    background: #fafaf7;
    margin-bottom: 0.6rem;
    border-radius: 0 4px 4px 0;
  }
  .rec-card .rank {
    font-family: 'JetBrains Mono', monospace;
    color: #c8553d;
    font-weight: 500;
    font-size: 0.82rem;
  }
  .rec-card .title {
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #111827;
  }
  .rec-card .meta {
    font-size: 0.82rem;
    color: #6b7280;
  }
  .divider-thin {
    border: 0;
    border-top: 1px solid #e5e7eb;
    margin: 0.5rem 0 1rem 0;
  }
  .explain-box {
    padding: 1.2rem 1.4rem;
    background: linear-gradient(180deg, #fbf7f4 0%, #f5eee8 100%);
    border-left: 4px solid #c8553d;
    border-radius: 0 6px 6px 0;
    font-size: 1.02rem;
    line-height: 1.65;
    color: #1f2937;
  }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# Argument parsing (Streamlit needs sys.argv after --)
# =============================================================================

def parse_args():
    # Streamlit passes args after a "--" separator
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed.pkl")
    parser.add_argument("--ckpt", type=str, default="checkpoints/hybrid_ncf.pt")
    parser.add_argument("--metrics_json", type=str, default="results/eval_metrics.json")
    return parser.parse_args(argv)


ARGS = parse_args()


# =============================================================================
# Cached loaders
# =============================================================================

@st.cache_resource
def load_everything(data_path, ckpt_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model = HybridNCF(
        n_users=data["n_users"],
        n_movies=data["n_movies"],
        n_genres=data["n_genres"],
        gmf_dim=ckpt_args.get("gmf_dim", 32),
        mlp_dim=ckpt_args.get("mlp_dim", 32),
        dropout=ckpt_args.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    recommender = Recommender(model, data, device)
    explainer = CounterfactualExplainer(model, data, device)
    return data, model, device, recommender, explainer


@st.cache_data
def load_metrics(path):
    p = Path(path)
    if not p.exists():
        return None
    import json
    with open(p) as f:
        return json.load(f)


# =============================================================================
# Header
# =============================================================================

st.markdown(
    """
    <div style="margin-bottom: 0.4rem;">
      <span class="kicker">CS 550 · Explainable Recommender System</span>
    </div>
    <h1>Hybrid Neural Collaborative Filtering<br>
        <span class="accent">with Counterfactual Explanations</span></h1>
    <p class="muted" style="max-width: 820px; font-size: 1.02rem;">
      A deep-learning recommender trained on MovieLens 1M that predicts user ratings
      <em>and</em> explains each recommendation through a counterfactual lens: which
      past items most drove this recommendation, and what would be recommended instead
      in their absence.
    </p>
    <hr class="divider-thin" />
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Load data / model
# =============================================================================

try:
    data, model, device, recommender, explainer = load_everything(
        ARGS.data, ARGS.ckpt
    )
    metrics = load_metrics(ARGS.metrics_json)
except FileNotFoundError as e:
    st.error(
        f"Missing file: {e.filename}\n\n"
        "You need to run preprocessing + training first:\n"
        "```\n"
        "python data/preprocess.py --data_dir ./ml-1m --out_dir ./data\n"
        "python train.py --data data/processed.pkl --epochs 20\n"
        "python -m evaluation.evaluate --ckpt checkpoints/hybrid_ncf.pt\n"
        "```"
    )
    st.stop()


# =============================================================================
# Sidebar — user picker
# =============================================================================

with st.sidebar:
    st.markdown("### Explore a user")
    st.caption(
        f"{data['n_users']:,} users · {data['n_movies']:,} movies · "
        f"{len(data['train']):,} train ratings"
    )
    user_idx = st.number_input(
        "User index",
        min_value=0, max_value=data["n_users"] - 1, value=0, step=1,
        help="Pick any user from the training data (0 through N-1).",
    )
    k = st.slider("Top-K recommendations", 5, 20, 10)
    st.markdown("---")
    st.markdown("### Model")
    n_params = sum(p.numel() for p in model.parameters())
    st.caption(f"Hybrid NCF · {n_params:,} params · {device.type.upper()}")
    if metrics:
        st.markdown("---")
        st.markdown("### Test-set metrics")
        st.caption(f"MAE  {metrics.get('MAE', 0):.4f}")
        st.caption(f"RMSE {metrics.get('RMSE', 0):.4f}")
        if "Precision@10" in metrics:
            st.caption(f"P@10 {metrics['Precision@10']:.4f}")
            st.caption(f"R@10 {metrics['Recall@10']:.4f}")
            st.caption(f"NDCG@10 {metrics['NDCG@10']:.4f}")


# =============================================================================
# Tabs
# =============================================================================

tab_recs, tab_why, tab_profile, tab_eval = st.tabs([
    "Recommendations",
    "Why this?",
    "Your profile",
    "Model evaluation",
])


# -------- Tab 1: Recommendations ------------------------------------------
with tab_recs:
    st.markdown(f"### Top-{k} recommendations for user #{user_idx}")
    recs = recommender.recommend(user_idx=int(user_idx), k=int(k))
    # Store the chosen user's recs so the "Why" tab can reuse them
    st.session_state["current_recs"] = recs
    st.session_state["current_user"] = int(user_idx)

    cols = st.columns(2)
    for i, r in enumerate(recs):
        with cols[i % 2]:
            genres_html = ", ".join(g for g in r.genres.split("|") if g)
            st.markdown(
                f"""<div class="rec-card">
                      <div class="rank">#{r.rank:02d} · predicted {r.predicted_rating:.2f}/5.0</div>
                      <div class="title">{r.title}</div>
                      <div class="meta">{genres_html}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    # --- Test-item hit indication ---
    test_items = recommender.get_test_items(user_idx)
    test_item_ids = {t["movie_idx"] for t in test_items}
    hits = [r for r in recs if r.movie_idx in test_item_ids]
    st.markdown("<br>", unsafe_allow_html=True)
    hit_col1, hit_col2 = st.columns([1, 3])
    with hit_col1:
        st.markdown(
            f"""<div class="metric-card">
                  <div class="label">hits in top-{k}</div>
                  <div class="value">{len(hits)}<span style="font-size:1.1rem;color:#6b7280">/{len(test_items)}</span></div>
                </div>""",
            unsafe_allow_html=True,
        )
    with hit_col2:
        st.markdown(
            "<div class='kicker' style='margin-bottom:0.3rem'>evaluation</div>"
            "<p class='muted' style='font-size:0.92rem'>Each user's test set is their held-out 20% of ratings. "
            "A 'hit' is when a recommendation appears in that held-out set. "
            "Precision@K, Recall@K, F1@K and NDCG@K aggregate this across all users.</p>",
            unsafe_allow_html=True,
        )


# -------- Tab 2: Why this? ------------------------------------------------
with tab_why:
    st.markdown(f"### Counterfactual explanation")
    recs = st.session_state.get("current_recs")
    if not recs:
        st.info("Pick a user from the sidebar first.")
    else:
        titles = [f"#{r.rank}  {r.title}  (pred {r.predicted_rating:.2f})" for r in recs]
        chosen_idx = st.selectbox(
            "Explain this recommendation:",
            range(len(recs)),
            format_func=lambda i: titles[i],
        )
        target = recs[chosen_idx]

        with st.spinner("Generating counterfactual explanation..."):
            exp = explainer.explain(
                user_idx=int(user_idx),
                target_movie_idx=target.movie_idx,
                n_causes=3,
                n_alternatives=3,
            )

        # Summary box
        st.markdown(
            f"""<div class="explain-box">{exp.summary()}</div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # 3 columns: causes / alternatives / genres
        col_c, col_a, col_g = st.columns(3)

        with col_c:
            st.markdown("<div class='kicker'>Causal history items</div>", unsafe_allow_html=True)
            st.caption("Removing any of these would drop the predicted score by Δ.")
            if exp.cause_items:
                df_c = pd.DataFrame([
                    {
                        "Movie": c.title,
                        "Your rating": f"{c.user_rating:.0f}",
                        "Δ score": f"{c.delta_score:+.4f}",
                    }
                    for c in exp.cause_items
                ])
                st.dataframe(df_c, hide_index=True, use_container_width=True)
            else:
                st.info("No training history for this user.")

        with col_a:
            st.markdown("<div class='kicker'>Counterfactual alternatives</div>", unsafe_allow_html=True)
            st.caption("Top picks if the causal items were removed from your history.")
            if exp.alternatives:
                df_a = pd.DataFrame([
                    {
                        "Movie": a.title,
                        "New score": f"{a.new_score:.3f}",
                    }
                    for a in exp.alternatives
                ])
                st.dataframe(df_a, hide_index=True, use_container_width=True)
            else:
                st.info("No alternatives available.")

        with col_g:
            st.markdown("<div class='kicker'>Genre attribution</div>", unsafe_allow_html=True)
            st.caption("Which of the target's genres most influenced the prediction.")
            if exp.top_genres:
                df_g = pd.DataFrame([
                    {"Genre": g, "Weight": f"{w*100:.1f}%"}
                    for g, w in exp.top_genres
                ])
                st.dataframe(df_g, hide_index=True, use_container_width=True)
                # Bar chart
                chart_df = pd.DataFrame(
                    {"weight": [w for _, w in exp.top_genres]},
                    index=[g for g, _ in exp.top_genres],
                )
                st.bar_chart(chart_df)


# -------- Tab 3: Your profile --------------------------------------------
with tab_profile:
    st.markdown(f"### User #{user_idx} · rating history")
    hist = recommender.get_user_history(int(user_idx), top_rated_first=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<div class='kicker'>training ratings (top-rated first)</div>",
                    unsafe_allow_html=True)
        df_hist = pd.DataFrame([
            {
                "Rating": int(h["rating"]),
                "Movie": h["title"],
                "Genres": h["genres"],
            }
            for h in hist[:50]
        ])
        st.dataframe(df_hist, hide_index=True, use_container_width=True, height=400)

    with col2:
        # Genre preference histogram
        st.markdown("<div class='kicker'>genre preference (mean rating per genre)</div>",
                    unsafe_allow_html=True)
        genre_list = data["genre_list"]
        genre_matrix = data["genre_matrix"]
        weighted_sum = np.zeros(len(genre_list))
        weighted_count = np.zeros(len(genre_list))
        for h in hist:
            m = h["movie_idx"]
            r = h["rating"]
            for gi, present in enumerate(genre_matrix[m]):
                if present > 0.5:
                    weighted_sum[gi] += r
                    weighted_count[gi] += 1
        avg = np.divide(
            weighted_sum, weighted_count,
            out=np.zeros_like(weighted_sum), where=weighted_count > 0,
        )
        gdf = pd.DataFrame(
            {"avg_rating": avg, "count": weighted_count.astype(int)},
            index=genre_list,
        ).sort_values("avg_rating", ascending=False)
        # Filter out genres with very few ratings for stability
        gdf_plot = gdf[gdf["count"] >= 2]
        st.bar_chart(gdf_plot["avg_rating"])
        st.caption(f"Based on {len(hist)} training ratings.")


# -------- Tab 4: Model evaluation ----------------------------------------
with tab_eval:
    st.markdown("### Aggregate test-set metrics")
    if metrics is None:
        st.info(
            "No metrics file found. Run:\n"
            "```\npython -m evaluation.evaluate "
            "--ckpt checkpoints/hybrid_ncf.pt --data data/processed.pkl\n```"
        )
    else:
        # Rating prediction row
        st.markdown("<div class='kicker'>rating prediction</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""<div class="metric-card">
                      <div class="label">MAE</div>
                      <div class="value">{metrics['MAE']:.4f}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div class="metric-card">
                      <div class="label">RMSE</div>
                      <div class="value">{metrics['RMSE']:.4f}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

        # Top-N row
        st.markdown("<br><div class='kicker'>top-10 recommendation</div>",
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, key in [(c1, "Precision@10"), (c2, "Recall@10"),
                         (c3, "F1@10"), (c4, "NDCG@10")]:
            if key in metrics:
                with col:
                    st.markdown(
                        f"""<div class="metric-card">
                              <div class="label">{key}</div>
                              <div class="value">{metrics[key]:.4f}</div>
                            </div>""",
                        unsafe_allow_html=True,
                    )

        if "n_users_evaluated" in metrics:
            st.caption(
                f"Evaluated on {metrics['n_users_evaluated']:,} users. "
                f"Each user's 20% holdout set served as ground-truth relevance."
            )

        # Training curve
        curve_path = Path("results/training_curve.csv")
        if curve_path.exists():
            st.markdown("<br><div class='kicker'>training curve</div>",
                        unsafe_allow_html=True)
            curve = pd.read_csv(curve_path)
            chart_df = curve[["train_rmse", "val_rmse"]].copy()
            chart_df.index = curve["epoch"]
            st.line_chart(chart_df)


st.markdown(
    "<hr class='divider-thin'/>"
    "<p class='muted' style='font-size:0.8rem; text-align:center'>"
    "CS 550 · Massive Data Mining and Learning · Rutgers University"
    "</p>",
    unsafe_allow_html=True,
)
