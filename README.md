## How to run (fresh install, ~5 minutes)

```bash
# 1. Create environment
conda create -n cs550 python=3.11 -y
conda activate cs550

# 2. Install dependencies (versions pinned for NumPy 1.x compatibility)
pip install -r requirements.txt

# 3. Verify everything works (synthetic, ~1 min)
python notebooks/smoke_test.py       # expects: SMOKE TEST PASSED.
python notebooks/smoke_test_bpr.py   # expects: BPR SMOKE TEST PASSED.
```

If the smoke tests pass, the codebase is fully functional.

## Reproducing the results

Everything needed is already in the zip:
- `data/processed.pkl` — the MovieLens 1M train/test split used for training
- `checkpoints/hybrid_ncf.pt` — trained MSE model
- `checkpoints/hybrid_ncf_bpr.pt` — trained BPR model
- `results/*.json` — all computed metrics

### Launch the interactive demo

```bash
streamlit run demo/app.py -- \
    --data data/processed.pkl \
    --ckpt checkpoints/hybrid_ncf.pt \
    --metrics_json results/eval_metrics.json
```

Browser opens at `http://localhost:8501`. Four tabs: **Recommendations**, **Why this?** (counterfactual explanations), **Your profile**, **Model evaluation**.

### Re-evaluate the trained checkpoints

```bash
# MSE variant — rating prediction metrics
python -m evaluation.evaluate \
    --data data/processed.pkl \
    --ckpt checkpoints/hybrid_ncf.pt \
    --k 10 --out results/eval_metrics_reproduced.json

# BPR variant — top-N ranking metrics
python -m evaluation.evaluate \
    --data data/processed.pkl \
    --ckpt checkpoints/hybrid_ncf_bpr.pt \
    --k 10 --out results/eval_metrics_bpr_reproduced.json

# Baselines
python scripts/baselines.py \
    --data data/processed.pkl \
    --k 10 --out results/baseline_metrics_reproduced.json
```

### Re-train from scratch (requires GPU, optional)

```bash
# Re-download raw data
bash scripts/download_movielens.sh
python data/preprocess.py --data_dir ./ml-1m --out_dir ./data

# Train MSE variant (~3 min on RTX 3090, ~30 min on CPU)
python train.py --data data/processed.pkl --epochs 25 --ckpt checkpoints/hybrid_ncf.pt

# Train BPR variant (~2 min on RTX 3090)
python train_bpr.py --data data/processed.pkl --epochs 20 --ckpt checkpoints/hybrid_ncf_bpr.pt
```

### Run on Amarel as a SLURM job

```bash
mkdir -p slurm/logs
sbatch slurm/run_all.sbatch
squeue -u $USER
tail -f slurm/logs/hybrid_ncf-<JOBID>.out
```

Expect 15–25 minutes end-to-end.

## Troubleshooting

**`RuntimeError: Numpy is not available`** when running any Python script.
Your NumPy is newer than this project's pinned PyTorch supports. Fix:
```bash
pip install "numpy<2" --force-reinstall
```

**`ModuleNotFoundError: No module named 'numpy._core.numeric'`** when loading `data/processed.pkl`.
The pickle was created with a different NumPy major version than yours. Regenerate:
```bash
bash scripts/download_movielens.sh
python data/preprocess.py --data_dir ./ml-1m --out_dir ./data
```
The split is deterministic (seed=42), so the resulting `processed.pkl` matches what the checkpoints were trained on.

**SLURM `module load anaconda` fails on Amarel.**
Amarel's module system doesn't use that name. Skip the `module load` step and use `source ~/.bashrc && conda activate cs550` instead (see `slurm/run_all.sbatch`).

## The approach, in one paragraph

Hybrid Neural Collaborative Filtering combines three signal paths in a single model: a **Generalized Matrix Factorization (GMF) branch** (element-wise product of user and item embeddings, capturing linear collaborative signal), a **Multi-Layer Perceptron (MLP) branch** (concatenated embeddings through non-linear layers, capturing non-linear interactions), and a **genre content tower** (multi-hot genre encoding through a small MLP, adding content-based signal for sparse items). The three outputs are fused through a final head. We train two variants with the same architecture: MSE regression (for rating prediction) and Bayesian Personalized Ranking (BPR) (for top-N ranking). For the trustworthiness bonus, a **counterfactual explanation module** perturbs the learned user embedding by subtracting the contribution of individual history items, quantifying which past ratings most drove a given recommendation and surfacing the alternative recommendations the user would see if those items were removed. Genre-level gradient attribution provides a complementary content-side explanation.

## References

Key papers cited in the report:
- He et al., "Neural Collaborative Filtering" (WWW 2017)
- Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009)
- Tan et al., "Counterfactual Explainable Recommendation" (CIKM 2021, arxiv 2108.10539)
- Cañamares & Castells, "Should I Follow the Crowd? A Probabilistic Analysis of the Effectiveness of Popularity in Recommender Systems" (SIGIR 2018)
EOF
echo "--- README written ---"
wc -l README.md
head -20 README.md