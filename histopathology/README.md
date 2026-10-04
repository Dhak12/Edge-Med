# EdgeMed — Breast Histopathology Classification

## Project Overview
EdgeMed is a deep learning project for classifying breast histopathology images into three categories:
- Normal
- Benign
- Malignant

The project uses **Swin Transformer Tiny (Swin-T)** with PyTorch and torchvision.

## Technology Stack
- Python
- PyTorch
- Torchvision
- MONAI
- Pandas
- Pillow
- Scikit-learn
- Matplotlib

## Project Structure
```text
histopathology/
├── scripts/
│   ├── create_splits.py
│   ├── test_dataset.py
│   ├── train_baseline.py
│   ├── evaluate_baseline.py
│   └── check_swin.py
├── checkpoints/
├── data/
│   ├── normal/
│   ├── benign/
│   └── malignant/
├── splits/
├── results/
├── requirements.txt
└── README.md
```

## Dataset
The final curated dataset will be provided separately and organized into Normal, Benign, and Malignant classes.

The dataset will use 200× magnification images. Duplicate and near-duplicate images should be removed before training. The final class distribution must be checked before finalizing the training configuration.

## Setup

Create and activate a virtual environment on Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with the system's NVIDIA GPU and CUDA environment.

## Dataset Preparation

Place the final dataset in the `data/` directory. Generate new training, validation, and test splits:

```bash
python scripts/create_splits.py
```

Verify the dataset and image loading:

```bash
python scripts/test_dataset.py
```

Ensure that the generated paths and class labels match the actual dataset structure before training.

## Model Training

Train the Swin-T model:

```bash
python scripts/train_baseline.py
```

Review the configuration in the training script before starting the final experiment. Select the best checkpoint using validation performance.

## Model Evaluation

Evaluate the trained model:

```bash
python scripts/evaluate_baseline.py
```

The evaluation script reports accuracy, macro precision, macro recall, macro F1-score, a classification report, and a confusion matrix.

## Initial Baseline Results

The initial baseline experiment used 1,500 images, with 500 images per class.

| Metric | Result |
|---|---:|
| Test Accuracy | 96.89% |
| Macro Precision | 96.96% |
| Macro Recall | 96.89% |
| Macro F1-score | 96.89% |

These results belong to the initial experiment and do not represent the performance of the final curated dataset.

## Important Notes
- The final dataset will be provided separately.
- Generate new data splits for the final dataset.
- Check class distribution and prevent data leakage between splits.
- Keep the test set separate from training and model selection.
- Save the final checkpoint and evaluation results.
- This project is an experimental research prototype and is not intended to replace professional medical diagnosis.
