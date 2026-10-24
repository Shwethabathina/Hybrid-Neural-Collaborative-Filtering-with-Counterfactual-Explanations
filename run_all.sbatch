#!/bin/bash
#SBATCH --job-name=hybrid_ncf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

# -----------------------------------------------------------------------------
# Hybrid-NCF full pipeline on Amarel
#
# Usage (from project root):
#   mkdir -p slurm/logs
#   sbatch slurm/run_all.sbatch
#
# This script:
#   1. Loads an anaconda module, activates the project env
#   2. Downloads MovieLens 1M (idempotent)
#   3. Preprocesses the raw data
#   4. Trains Hybrid-NCF
#   5. Evaluates and writes metrics JSON
#
# Before the first run, create the conda env:
#   module load anaconda
#   conda create -n cs550 python=3.11 -y
#   conda activate cs550
#   pip install -r requirements.txt
# -----------------------------------------------------------------------------

set -euo pipefail

echo "[job] Host: $(hostname)"
echo "[job] Started: $(date)"
echo "[job] CWD:   $(pwd)"
nvidia-smi || true

# --- Environment -------------------------------------------------------------
# Amarel uses lmod; adjust module names if your setup differs.
module purge || true
module load anaconda || true
# shellcheck disable=SC1091
source activate cs550 || conda activate cs550 || {
  echo "Could not activate conda env 'cs550'. Create it first:" >&2
  echo "  conda create -n cs550 python=3.11 -y && conda activate cs550 && pip install -r requirements.txt" >&2
  exit 1
}

python - <<'PY'
import torch
print(f"[env] torch={torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PY

# --- Step 1: download raw data ----------------------------------------------
bash scripts/download_movielens.sh

# --- Step 2: preprocess ------------------------------------------------------
if [ ! -f "data/processed.pkl" ]; then
  echo "[job] Preprocessing..."
  python data/preprocess.py --data_dir ./ml-1m --out_dir ./data
else
  echo "[job] data/processed.pkl already exists -- skipping preprocessing."
fi

# --- Step 3: train -----------------------------------------------------------
echo "[job] Training Hybrid-NCF..."
python train.py \
  --data data/processed.pkl \
  --epochs 25 \
  --batch_size 2048 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --gmf_dim 32 \
  --mlp_dim 32 \
  --dropout 0.2 \
  --patience 4 \
  --ckpt checkpoints/hybrid_ncf.pt \
  --curve_csv results/training_curve.csv

# --- Step 4: evaluate --------------------------------------------------------
echo "[job] Evaluating..."
python -m evaluation.evaluate \
  --data data/processed.pkl \
  --ckpt checkpoints/hybrid_ncf.pt \
  --k 10 \
  --out results/eval_metrics.json

# --- Step 5: baselines -------------------------------------------------------
echo "[job] Computing baselines..."
python scripts/baselines.py \
  --data data/processed.pkl \
  --k 10 \
  --out results/baseline_metrics.json

# --- Step 6: fill report placeholders ---------------------------------------
echo "[job] Filling report placeholders..."
python scripts/fill_report.py \
  --template report/report.tex \
  --metrics  results/eval_metrics.json \
  --curve_csv results/training_curve.csv \
  --out      report/report_filled.tex

echo "[job] Done: $(date)"
echo "[job] Next: scp results/ checkpoints/ data/processed.pkl back to your laptop,"
echo "[job]       then 'make demo' to launch the Streamlit app."
