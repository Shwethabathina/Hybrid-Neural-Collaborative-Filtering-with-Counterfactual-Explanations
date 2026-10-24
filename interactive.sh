#!/bin/bash
# Quick interactive GPU session on Amarel for debugging/iteration.
# Usage: bash slurm/interactive.sh
#
# Once the shell opens on a GPU node, do:
#   module load anaconda
#   conda activate cs550
#   python notebooks/smoke_test.py    # sanity check
#   # or
#   python train.py --data data/processed.pkl --epochs 3   # quick train

srun --partition=gpu \
     --gres=gpu:1 \
     --cpus-per-task=4 \
     --mem=16G \
     --time=02:00:00 \
     --pty bash
