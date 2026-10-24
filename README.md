# Hybrid Neural Collaborative Filtering for Movie Recommendations

An end-to-end movie recommendation system combining **collaborative filtering, neural networks, and genre-based content features**, with interpretable recommendations and an interactive Streamlit dashboard.

Built using the **MovieLens 1M dataset**, this project explores how hybrid neural recommendation models can improve rating prediction and personalized movie ranking while providing meaningful explanations for their recommendations.

## Project Overview

Traditional collaborative filtering methods primarily rely on user–item interactions, which can limit their ability to capture complex preferences and incorporate content information.

This project implements a **Hybrid Neural Collaborative Filtering (Hybrid NCF)** architecture that integrates three complementary recommendation signals:

1. **Generalized Matrix Factorization (GMF):** Captures linear collaborative relationships between users and movies.
2. **Multi-Layer Perceptron (MLP):** Learns complex, nonlinear user–movie interactions.
3. **Genre Content Tower:** Incorporates genre information to provide additional content-based recommendation signals.

The outputs of these components are combined through a final prediction head.

Two training objectives are explored:

- **Mean Squared Error (MSE):** Predicts explicit movie ratings.
- **Bayesian Personalized Ranking (BPR):** Optimizes personalized top-N recommendation rankings.

The system also includes counterfactual explanations, genre attribution, evaluation scripts, reproducible experiments, and an interactive demonstration.

---

## Key Features

- Hybrid neural collaborative filtering architecture
- GMF and MLP branches for modeling user–item interactions
- Genre-based content embeddings
- Explicit rating prediction using MSE
- Personalized ranking using BPR
- Reproducible training and evaluation
- Baseline recommender comparisons
- Counterfactual recommendation explanations
- Genre-level gradient attribution
- Interactive Streamlit dashboard
- Support for GPU training and SLURM execution on Amarel
- Pretrained checkpoints and evaluation results

---

## Model Architecture

The Hybrid NCF model combines three complementary paths.

**1. Generalized Matrix Factorization (GMF)**

User and movie embeddings are combined using element-wise multiplication to capture collaborative interaction signals.

**2. Multi-Layer Perceptron (MLP)**

User and movie embeddings are concatenated and processed through fully connected layers to learn nonlinear preference patterns.

**3. Genre Content Tower**

Movies are represented through multi-hot encoded genre vectors, which are transformed through a small neural network to generate content-aware representations.

**Fusion Layer**

The representations from GMF, MLP, and the genre tower are fused through a final prediction head.

The same overall architecture is trained using two objectives:

| Model variant | Training objective | Primary purpose |
|---|---|---|
| Hybrid NCF (MSE) | Mean Squared Error | Explicit rating prediction |
| Hybrid NCF (BPR) | Bayesian Personalized Ranking | Personalized top-N ranking |

---

## Dataset

This project uses the **MovieLens 1M** dataset, a widely used benchmark for recommender systems.

The dataset contains approximately:

| Attribute | Description |
|---|---|
| Ratings | 1 million |
| Users | 6,040 |
| Movies | Approximately 3,900 |
| Rating scale | 1–5 stars |
| Additional features | Movie genres |

Dataset source: [MovieLens 1M — GroupLens](https://grouplens.org/datasets/movielens/1m/)

The repository includes a preprocessed dataset:

`data/processed.pkl`

This contains the training and testing data used by the project.

The preprocessing split is deterministic, using **random seed 42**.

---

## Installation and Setup

### Prerequisites

- Python 3.11
- Conda (recommended)
- pip
- Git (optional)
- CUDA-compatible GPU (optional, for faster training)

### 1. Create the Conda environment

```bash
conda create -n cs550 python=3.11 -y
conda activate cs550
```

### 2. Install dependencies

From the project root directory, run:

```bash
pip install -r requirements.txt
```

Dependency versions are pinned for compatibility with NumPy 1.x and the project's PyTorch version.

### 3. Verify the installation

Run the provided smoke tests:

```bash
python notebooks/smoke_test.py
python notebooks/smoke_test_bpr.py
```

Expected output from the first test:

```text
SMOKE TEST PASSED.
```

Expected output from the second test:

```text
BPR SMOKE TEST PASSED.
```

Passing both tests verifies the functionality covered by the MSE and BPR smoke tests.

**Estimated fresh installation and verification time:** Approximately 5 minutes, depending on the system and dependency downloads.

---

## Reproducing the Results

The project archive includes the processed data, trained model checkpoints, and previously computed evaluation results.

No retraining is necessary to evaluate the supplied models or explore the interactive demo.

### Included artifacts

| File | Purpose |
|---|---|
| `data/processed.pkl` | Preprocessed MovieLens 1M training/testing data |
| `checkpoints/hybrid_ncf.pt` | Trained MSE model |
| `checkpoints/hybrid_ncf_bpr.pt` | Trained BPR model |
| `results/*.json` | Saved evaluation results |

### Evaluate the MSE model

```bash
python -m evaluation.evaluate \
  --data data/processed.pkl \
  --ckpt checkpoints/hybrid_ncf.pt \
  --k 10 \
  --out results/eval_metrics_reproduced.json
```

This evaluates the trained MSE checkpoint and writes the reproduced evaluation metrics to a new JSON file.

### Evaluate the BPR model

```bash
python -m evaluation.evaluate \
  --data data/processed.pkl \
  --ckpt checkpoints/hybrid_ncf_bpr.pt \
  --k 10 \
  --out results/eval_metrics_bpr_reproduced.json
```

This evaluates the BPR checkpoint and saves the reproduced ranking metrics.

### Evaluate baseline recommenders

```bash
python scripts/baselines.py \
  --data data/processed.pkl \
  --k 10 \
  --out results/baseline_metrics_reproduced.json
```

Baseline evaluations provide reference points for comparing the performance of the hybrid neural models.

The `--k 10` argument specifies a top-10 recommendation cutoff for applicable ranking metrics.

---

## Interactive Streamlit Demo

The project includes an interactive Streamlit application for exploring recommendations, explanations, user preferences, and model evaluation.

### Launch the application

Run the following command from the project root:

```bash
streamlit run demo/app.py -- \
  --data data/processed.pkl \
  --ckpt checkpoints/hybrid_ncf.pt \
  --metrics_json results/eval_metrics.json
```

Once the server starts, open:

**http://localhost:8501**

### Dashboard Features

| Tab | Description |
|---|---|
| Recommendations | Explore personalized movie recommendations |
| Why this? | Inspect counterfactual recommendation explanations |
| Your profile | Explore the user's movie preference profile |
| Model evaluation | View the saved model evaluation results |

The application uses the pretrained MSE checkpoint by default in the command above.

---

## Explainable and Trustworthy Recommendations

Beyond generating recommendations, the system provides mechanisms for understanding why particular movies are suggested.

### Counterfactual Explanations

The counterfactual explanation module investigates how individual movies in a user's rating history influence a recommendation.

The approach:

1. Identifies a user's historical movie interactions.
2. Estimates the contribution of selected history items to the learned user representation.
3. Perturbs the user embedding by subtracting the estimated contribution of a selected item.
4. Recomputes recommendation scores using the modified representation.
5. Compares the original and counterfactual recommendation outputs.

This provides a model-based approximation of how recommendations might change if a particular historical interaction had contributed differently.

**Important:** These are embedding-level counterfactual approximations, not exact retraining-based measures of causal influence.

### Genre-Level Gradient Attribution

The project also computes gradient-based attribution for movie genre features.

This offers a complementary content-side explanation by identifying genre inputs that influence the model's prediction.

Together, these components improve the interpretability of the system's recommendation behavior.

---

## Training Models from Scratch

Retraining is optional because pretrained checkpoints are included.

A GPU is recommended for faster experimentation, although CPU training is supported for the MSE variant.

### 1. Download the raw MovieLens dataset

```bash
bash scripts/download_movielens.sh
```

### 2. Preprocess the dataset

```bash
python data/preprocess.py \
  --data_dir ./ml-1m \
  --out_dir ./data
```

Preprocessing produces the dataset artifacts required for training and evaluation.

The train/test splitting procedure uses a fixed seed of 42.

### 3. Train the MSE variant

```bash
python train.py \
  --data data/processed.pkl \
  --epochs 25 \
  --ckpt checkpoints/hybrid_ncf.pt
```

Approximate training times:

| Hardware | Estimated time |
|---|---|
| NVIDIA RTX 3090 | 3 minutes |
| CPU | 30 minutes |

### 4. Train the BPR variant

```bash
python train_bpr.py \
  --data data/processed.pkl \
  --epochs 20 \
  --ckpt checkpoints/hybrid_ncf_bpr.pt
```

Estimated training time on an NVIDIA RTX 3090: approximately 2 minutes.

Training times depend on hardware, environment configuration, and resource availability.

**Note:** These commands overwrite the corresponding checkpoint paths if the training scripts save to the specified destinations. Use different checkpoint filenames to preserve the supplied pretrained models.

---

## Running on Amarel (SLURM)

The project supports execution on the Rutgers Amarel high-performance computing environment using SLURM.

### 1. Create the log directory

```bash
mkdir -p slurm/logs
```

### 2. Submit the SLURM job

```bash
sbatch slurm/run_all.sbatch
```

### 3. Check the job queue

```bash
squeue -u $USER
```

### 4. Monitor job output

Replace `<JOBID>` with the SLURM job ID returned by `sbatch`.

```bash
tail -f slurm/logs/hybrid_ncf-<JOBID>.out
```

**Estimated execution time:** Approximately 15–25 minutes end-to-end, depending on allocated resources and job execution conditions. Queue waiting time is additional.

### Conda activation on Amarel

If the command below fails:

```bash
module load anaconda
```

Use the available Conda installation directly:

```bash
source ~/.bashrc
conda activate cs550
```

The repository's `slurm/run_all.sbatch` contains the intended job configuration.

---

## Evaluation and Experiments

The project supports separate evaluation of rating prediction and recommendation ranking.

### Rating Prediction

The MSE model is trained to estimate users' explicit ratings.

Its primary objective is minimizing squared prediction error.

### Top-N Recommendation

The BPR model focuses on learning pairwise preferences, encouraging positively interacted items to rank above sampled negative items.

### Baseline Comparisons

Baseline recommender models provide reference performance for evaluating the hybrid architecture.

### Reproducibility

Reproducibility is supported through:

- Included processed training/testing data
- Fixed preprocessing seed
- Supplied pretrained checkpoints
- Saved evaluation metrics
- Re-evaluation scripts
- Smoke tests for both training variants
- Pinned dependency versions

The stored results are available under `results/`.

Exact metric values should be read from the corresponding JSON files rather than inferred from the architecture alone.

---

## Project Structure

The following structure highlights the key files and directories used by the supplied workflows.

```text
.
├── checkpoints/
│   ├── hybrid_ncf.pt
│   └── hybrid_ncf_bpr.pt
│
├── data/
│   ├── processed.pkl
│   └── preprocess.py
│
├── demo/
│   └── app.py
│
├── evaluation/
│   └── evaluate.py
│
├── notebooks/
│   ├── smoke_test.py
│   └── smoke_test_bpr.py
│
├── results/
│   ├── eval_metrics.json
│   └── *.json
│
├── scripts/
│   ├── baselines.py
│   └── download_movielens.sh
│
├── slurm/
│   ├── run_all.sbatch
│   └── logs/
│
├── train.py
├── train_bpr.py
├── requirements.txt
└── README.md
```

*This is a representative structure based on the provided run instructions; the actual repository may contain additional modules, figures, and experiment files.*

---

## Troubleshooting

### Issue 1: NumPy is not available

**Error:**

```text
RuntimeError: Numpy is not available
```

**Cause:** The installed NumPy version may be incompatible with the PyTorch version used by the project.

**Fix:**

```bash
pip install "numpy<2" --force-reinstall
```

After installation, rerun the smoke tests.

### Issue 2: NumPy core module not found

**Error:**

```text
ModuleNotFoundError: No module named 'numpy._core.numeric'
```

**Cause:** The serialized `processed.pkl` file was created using a different NumPy version than the one in the current environment.

**Fix:**

Regenerate the processed dataset:

```bash
bash scripts/download_movielens.sh
python data/preprocess.py \
  --data_dir ./ml-1m \
  --out_dir ./data
```

The preprocessing uses a deterministic seed of 42. For exact reproduction, the preprocessing implementation, raw data, dependencies, and split configuration must also remain consistent.

### Issue 3: Conda module not found on Amarel

**Error:**

```text
module load anaconda
```

fails.

**Fix:**

Use the installed Conda environment:

```bash
source ~/.bashrc
conda activate cs550
```

### Issue 4: Streamlit application does not start

Confirm that the environment is active and dependencies are installed:

```bash
conda activate cs550
pip install -r requirements.txt
```

Then run the Streamlit command again from the repository root.

---

## Research Background

The project builds on established research in neural recommendation systems, pairwise ranking, and explainable recommendations.

**Neural Collaborative Filtering**

He et al. introduced neural collaborative filtering as a framework for modeling user–item interactions through neural network architectures.

**Bayesian Personalized Ranking**

Rendle et al. proposed BPR, a pairwise ranking approach for learning personalized item rankings from implicit feedback.

**Counterfactual Explainable Recommendation**

Counterfactual recommendation research motivates understanding how recommendations might change under modifications to user interaction histories or model inputs.

**Popularity in Recommender Systems**

Research on popularity-based recommendation provides additional context for interpreting recommendation quality and comparing recommendation strategies.

---

## References

1. He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). **Neural Collaborative Filtering.** Proceedings of the 26th International Conference on World Wide Web (WWW). [Paper](https://arxiv.org/abs/1708.05031)

2. Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). **BPR: Bayesian Personalized Ranking from Implicit Feedback.** Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI). [Paper](https://arxiv.org/abs/1205.2618)

3. Tan et al. (2021). **Counterfactual Explainable Recommendation.** [arXiv:2108.10539](https://arxiv.org/abs/2108.10539)

4. Cañamares, R., & Castells, P. (2018). **Should I Follow the Crowd? A Probabilistic Analysis of the Effectiveness of Popularity in Recommender Systems.** Proceedings of SIGIR 2018.

---

## Quick Start

For users who want to run the application immediately:

```bash
# Create environment
conda create -n cs550 python=3.11 -y
conda activate cs550

# Install dependencies
pip install -r requirements.txt

# Verify installation
python notebooks/smoke_test.py
python notebooks/smoke_test_bpr.py

# Launch interactive demo
streamlit run demo/app.py -- \
  --data data/processed.pkl \
  --ckpt checkpoints/hybrid_ncf.pt \
  --metrics_json results/eval_metrics.json
```

Open **http://localhost:8501** to explore personalized movie recommendations and explanations.

---

## Project Summary

**Hybrid Neural Collaborative Filtering** combines linear collaborative interactions, nonlinear preference learning, and genre-based content features into a unified recommendation architecture.

By supporting both MSE-based rating prediction and BPR-based ranking, the system enables experimentation with different recommendation objectives. Its counterfactual explanation and gradient attribution modules provide additional insight into model behavior.

The project delivers a reproducible recommendation workflow spanning preprocessing, model training, evaluation, baseline comparisons, high-performance computing execution, and an interactive web demonstration.
