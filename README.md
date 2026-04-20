# CENG 463 Take-Home Midterm Scaffold

This repository is a fresh starter for the CENG 463 Machine Learning take-home midterm due on 27 April 2026. It is organized so each question has its own config, runner, output folder, and report section, while sharing a single reproducible project layout.

## Default Dataset Choices

To reduce setup friction, the scaffold assumes the following defaults:

- Q1: California Housing via `sklearn.datasets`
- Q2: Credit Card Fraud CSV at `data/external/creditcard.csv`
- Q3: Fashion-MNIST via `torchvision`
- Q4: Wholesale Customers CSV at `data/external/wholesale_customers.csv`
- Q5: CIFAR-10 via `torchvision`

These choices match the assignment requirements well, but you can swap them later from the YAML configs.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/init_homework.sh
```

Running `scripts/init_homework.sh` creates one timestamped run folder per question under `outputs/` and saves:

- the resolved config
- a starter checklist
- empty artifact folders for figures, tables, logs, and models

## Running Individual Questions

```bash
bash scripts/run_q1.sh
bash scripts/run_q2.sh
bash scripts/run_q3.sh
bash scripts/run_q4.sh
bash scripts/run_q5.sh
```

## Repository Layout

```text
ceng463-take-home-midterm/
├── configs/
├── data/
├── notebooks/
├── outputs/
├── reports/
├── scripts/
├── src/
├── .gitignore
├── README.md
└── requirements.txt
```

Key conventions:

- `configs/` keeps one YAML config per question plus shared defaults.
- `data/` separates raw, interim, processed, and external assets.
- `outputs/` stores timestamped experiment runs and generated artifacts.
- `reports/` contains a LaTeX report template with one section per question.
- `src/` contains shared utilities and per-question starter pipelines.

## What The Starter Code Does

Each question pipeline currently handles the boring setup work for you:

- loads and resolves YAML config inheritance
- sets a global random seed
- creates a clean run directory
- writes a checklist for the exact deliverables that question needs

That means you can start implementing models and analysis inside the per-question packages without rebuilding the project skeleton first.

## Suggested Next Steps

1. Fill in your student ID and name in `reports/main.tex`.
2. Place external CSV datasets in the paths listed in [data/README.md](/Users/bakiceltik/Documents/GitHub/463/ceng463-take-home-midterm/data/README.md).
3. Start with one question at a time and extend the corresponding `pipeline.py`.
4. Push the repo once you are happy with the structure so you already have the required repository link ready.
