# ELEC378 Final Project — Image Classification

This repository contains the code, data references, notebooks, and results for our ELEC378 image classification final project.

## Repo layout
- `train_images/`, `test_images/` — image datasets (also cropped variants in `_cropped` folders)
- `train.csv` — training labels / metadata
- `kfold_clf.ipynb, kernelsvm.py`, `sift_knn_classifier.py` — training/inference scripts for each model used
- `data_explorer.ipynb` — analysis notebook
<!-- - `results/` — outputs, predictions, ensemble CSVs, and `submission.csv` -->
- `report/` — final report source (`final_report.tex`) and figures
- `environment.yml` — Conda environment used for experiments

## Setup

Create the environment and install dependencies from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate elec378 
```
## Quick usage
Ensure you have downloaded the dataset in the main directory (i.e. the images are at [train_images/test_images]/train_000001.jpg, etc.).
```bash
python kernelsvm.py
python sift_knn_classifier.py
```

You can run the CNN by running each cell in `kfold_clf.ipynb`.

## Reproduce experiments

- Use the notebooks for step-by-step data exploration and model evaluation.
- Check `lightning_logs/` for training runs (if PyTorch Lightning was used).
- Many experiments save outputs in `results/folds/` — see these when re-running cross-validation.