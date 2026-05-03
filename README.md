# CENG 463 Take-Home Midterm

Repository for Abdulbaki Çeltik's CENG 463 Machine Learning take-home midterm.

GitHub: <https://github.com/bakiceltik/ceng463-take-home-midterm-Abdulbaki-celtik>

The project contains reproducible Python pipelines, saved experiment artifacts, and a LaTeX report covering all five assignment questions.

## Questions Covered

- Q1 Regression: California Housing regression with linear, regularized, robust, and XGBoost models.
- Q2 Classification: credit-card fraud detection under extreme class imbalance with resampling, cost-sensitive learning, calibration, and threshold selection.
- Q3 Dimensionality Reduction: Fashion-MNIST embeddings with PCA, t-SNE, UMAP, and an autoencoder.
- Q4 Clustering: optdigits clustering with K-Means, GMM, DBSCAN, and agglomerative clustering.
- Q5 Neural Networks: CIFAR-10 MLP, CNN, ResNet18 transfer learning, Optuna-based compact hyperparameter search, Grad-CAM/LIME, and FGSM robustness.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The run scripts automatically use `./.venv/bin/python` when it exists, otherwise they fall back to `python3`.

## Data

- Q1 uses `sklearn.datasets.fetch_california_housing`.
- Q2 expects `data/external/creditcard.csv`.
- Q3 uses Fashion-MNIST through `torchvision`.
- Q4 uses the labelled `optdigits` setting through `sklearn.datasets.load_digits`.
- Q5 uses CIFAR-10 through `torchvision`, downloaded under `data/raw/`.

The `data/` directory keeps raw, interim, processed, and external data separated. Large external datasets should be placed locally rather than committed.

## Running Experiments

Run all questions:

```bash
bash scripts/init_homework.sh
```

Run individual questions:

```bash
bash scripts/run_q1.sh
bash scripts/run_q2.sh
bash scripts/run_q3.sh
bash scripts/run_q4.sh
bash scripts/run_q5.sh
```

Q5 also supports a quick smoke run:

```bash
./.venv/bin/python -m src.q5_neural_networks.pipeline --config configs/q5_neural_networks.yaml --smoke
```

## Report

The LaTeX report is in `reports/`.

Build the PDF:

```bash
cd reports
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The compiled report is `reports/main.pdf`. The repository link is included on the title page.

## Current Output Runs

Report-ready runs currently used by the LaTeX sections:

- `outputs/q1_regression/20260420-145313_california_housing_baseline`
- `outputs/q2_classification/20260420-135405_credit_card_fraud_baseline`
- `outputs/q3_dimensionality_reduction/20260501-165404_fashion_mnist_baseline`
- `outputs/q4_clustering/20260502-114853_optdigits_clustering_baseline`
- `outputs/q5_neural_networks/20260502-160350_cifar10_baseline`

Recent Q5 smoke-test outputs may also exist under `outputs/q5_neural_networks/`; these are for validation only and are not the report baseline.

Each run directory contains:

- `resolved_config.json`
- `figures/`
- `tables/`
- `logs/`
- `models/`

## Repository Layout

```text
ceng463-take-home-midterm/
├── configs/      # YAML configs for each question
├── data/         # raw, interim, processed, and external data locations
├── outputs/      # timestamped generated experiment artifacts
├── reports/      # LaTeX source, bibliography, and compiled PDF
├── scripts/      # per-question run scripts
├── src/          # shared utilities and question pipelines
├── README.md
└── requirements.txt
```

## Reproducibility Notes

- Config inheritance is handled through YAML files in `configs/`.
- Pipelines set fixed random seeds through shared helpers in `src/common/`.
- Final test-set reporting is separated from model selection where applicable.
- Generated tables and figures are written to timestamped folders under `outputs/`.
