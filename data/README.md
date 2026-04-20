# Data Layout

Use this folder to keep dataset handling consistent across questions.

## Folder Meanings

- `raw/`: auto-downloaded or untouched original files
- `interim/`: cleaned intermediate outputs
- `processed/`: modeling-ready tables or tensors
- `external/`: manually downloaded files that should not be committed

## Default Dataset Paths

- Q1 uses California Housing from `sklearn`, so no manual file is required.
- Q2 expects `data/external/creditcard.csv`.
- Q3 uses Fashion-MNIST from `torchvision`, downloaded into `data/raw/`.
- Q4 expects `data/external/wholesale_customers.csv`.
- Q5 uses CIFAR-10 from `torchvision`, downloaded into `data/raw/`.

If you switch to a different allowed dataset from the assignment PDF, update the corresponding YAML file in `configs/`.
