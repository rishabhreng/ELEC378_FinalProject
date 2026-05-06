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

## Setup Dataset
Download all files in the dataset from [here](https://www.kaggle.com/competitions/elec-378-sp-26-final-project/data) to the main directory.
When extracting the files, ensure the dataset is in the main directory (i.e. the images are at [train_images/test_images]/train_000001.jpg, etc.); otherwise you will need to change many of the file paths in the files.

## Quick usage
```bash
python kernelsvm.py
python sift_knn_classifier.py
```

You can run the CNN by running each cell in `kfold_clf.ipynb`, or by executing each section marked by the Markdown cells.

To use the checkpoints we have saved for the CNN, download them from [here](https://drive.google.com/file/d/1H8-it7z9_Wvpi4iegeSbukO0wG-NAQwh/view?usp=sharing) and extract the folder into the main directory.