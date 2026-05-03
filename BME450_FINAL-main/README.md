# Grocery and Beverage Image Sorting for Food Pantry Applications

## Team members
1. Raymond Weng (`rayjweng`)
2. Joshua Zhang (`ZhangJoshcode`)

## Project description
This repository contains a deep learning project for the BME450 final. It trains convolutional neural networks on food label images to classify **beverage** and **food** products. It also compares several parameter choices across two model types.

The project uses the two Kaggle links provided for the assignment.
1. The food and beverage label image dataset from Tushar Sharma is used for model training.
2. The grocery basket dataset from umairaslam is used for background context and a short data summary.

Important note. The `umairaslam/grocery` dataset is **not** an image dataset. It contains transaction baskets in `groceries.csv`, so it cannot be used by itself for image classification. This repo says that clearly so the project stays honest and accurate.

## Repository layout
1. `configs/experiments.yaml` stores the experiment settings.
2. `scripts/download_datasets.py` downloads both Kaggle datasets through `kagglehub`.
3. `scripts/prepare_image_dataset.py` creates stratified train, validation, and test manifests from the labeled image dataset.
4. `scripts/summarize_grocery_dataset.py` summarizes the grocery transaction dataset.
5. `scripts/run_experiments.py` trains and evaluates all experiments.
6. `src/pantryvision/` contains reusable data, model, and training utilities.
7. `reports/final_report_template.md` is a simple report outline.

## Environment setup
Create and activate a local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Full workflow
1. Download both Kaggle datasets:

```bash
source .venv/bin/activate
python scripts/download_datasets.py
```

2. Prepare the labeled image manifests:

```bash
source .venv/bin/activate
python scripts/prepare_image_dataset.py
```

3. Summarize the grocery transaction dataset for context:

```bash
source .venv/bin/activate
python scripts/summarize_grocery_dataset.py
```

4. Run the deep learning experiments:

```bash
source .venv/bin/activate
python scripts/run_experiments.py
```

## Experiment design
The project satisfies the class requirement to modify architecture and parameters.

1. Model types used:
   `resnet18`
   `mobilenet_v3_small`
2. Parameter choices changed across five experiments:
   optimizer
   learning rate
   batch size
   augmentation strength

All experiments use pretrained transfer learning. The last feature block and the classifier stay trainable so the code can still run in a reasonable amount of time on a CPU.

## Outputs
Running the scripts creates:
1. `data/processed/food_beverage_binary/*.csv` with train, validation, and test manifests
2. `artifacts/grocery_summary.md` with a summary of the grocery dataset that does not contain images
3. `artifacts/experiments/results.csv` with the experiment comparison results
4. `artifacts/experiments/<experiment_name>/summary.json` with run metrics
5. `artifacts/experiments/<experiment_name>/learning_curves.png`
6. `artifacts/experiments/<experiment_name>/confusion_matrix.png`

These outputs stay outside version control so the submitted repository remains under the 15 MB limit.
