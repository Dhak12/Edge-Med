# Edge-Med MRI Classification

MRI-based breast cancer classification module for the Edge-Med project.

The module classifies breast MRI images into three categories:

* Benign
* Malignant
* Healthy

## Overview

The MRI pipeline uses transfer learning with a pretrained ResNet-18 convolutional neural network.

The pipeline includes:

1. MRI dataset loading
2. Image preprocessing
3. Training-time data augmentation
4. Class-weighted loss
5. ResNet-18 transfer learning
6. Model validation
7. Best-model checkpointing
8. Test-set evaluation
9. Confusion matrix generation
10. Single-image prediction

The training configuration is fixed to a maximum of 10 epochs.

## Dataset Structure

The dataset should be stored separately from the Git repository.

The project expects the following structure:

```text
dataset/

│
├── train/
│   ├── Benign/
│   ├── Malignant/
│   └── Healthy/
│
├── validation/
│   ├── Benign/
│   ├── Malignant/
│   └── Healthy/
│
└── test/
    ├── Benign/
    ├── Malignant/
    └── Healthy/

The project uses torchvision.datasets.ImageFolder, so the class names are automatically obtained from the directory names.

The dataset is not included in this repository.

Dataset Statistics

The current dataset contains:

Split	Number of Images
Training	17,696
Validation	3,693
Testing	3,772
Total	25,161
Requirements
Python 3.14
PyTorch
Torchvision
NumPy
Pillow
Scikit-learn
Matplotlib
VS Code
CPU or NVIDIA GPU
Internet connection for downloading Python packages and pretrained ResNet-18 weights
Installation

Create a virtual environment:

python -m venv .venv

Activate it.

Windows
.venv\Scripts\activate
macOS/Linux
source .venv/bin/activate

Upgrade pip:

python -m pip install --upgrade pip

Install dependencies:

pip install -r requirements.txt
Dataset Path

The dataset path is configured in:

src/config.py

The project uses a project-relative dataset path:

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "dataset"

Therefore, the dataset should be placed inside the mri directory:

mri/
│
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/


The dataset/ directory is excluded from Git using .gitignore.

Training

From the mri directory, run:

python src/train.py

The model trains for exactly 10 epochs.

Training output includes:

training loss
training accuracy
validation loss
validation accuracy
best-model checkpoint

The best model is saved locally to:

outputs/checkpoints/best_model.pth

Model checkpoint files are excluded from the Git repository.

Training plots are saved to:

outputs/plots/
Evaluation

After training:

python src/evaluate.py

The evaluation uses the test dataset.

Generated results include:

outputs/

├── metrics/
│   └── metrics.json
│
└── plots/
    ├── loss_curve.png
    ├── accuracy_curve.png
    └── confusion_matrix.png

The test evaluation reports:

Accuracy
Precision
Recall
F1-score
Confusion matrix
Results

The current baseline ResNet-18 model achieved:

Best Validation Accuracy: 74.52%
Test Accuracy: 74.66%
Classification Performance
Class	Precision	Recall	F1-score
Benign	0.69	0.85	0.76
Healthy	1.00	0.99	0.99
Malignant	0.80	0.62	0.70

The current results represent the baseline MRI classification implementation.

Further improvements can be explored through fine-tuning, improved augmentation, class-balancing strategies, and model optimization.

Single Image Prediction

After training, run:

python src/predict.py "path/to/mri_image.jpg"

Example:

python src/predict.py "sample_mri.jpg"

The program displays:

predicted class
confidence
probability for each class
Model

The current implementation uses:

ResNet-18
      ↓
Pretrained ImageNet features
      ↓
Frozen feature extractor
      ↓
Dropout
      ↓
3-class classifier
      ↓
Benign / Malignant / Healthy

Preprocessing

Images are resized to:

224 × 224

Training augmentation includes:

Random horizontal flip
Small random rotation
Moderate brightness/contrast variation

Validation and test images are only resized and normalized.

Training Configuration
Parameter	Value
Model	ResNet-18
Input Size	224 × 224
Classes	3
Epochs	10
Optimizer	AdamW
Learning Rate	0.0001
Weight Decay	0.0001
Batch Size	16
Loss	Weighted Cross Entropy
Device	CPU/GPU automatically selected
Random Seed	42
Project Structure
mri/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── outputs/
│   ├── checkpoints/
│   │   └── best_model.pth
│   │
│   ├── plots/
│   │   ├── accuracy_curve.png
│   │   ├── loss_curve.png
│   │   └── confusion_matrix.png
│   │
│   └── metrics/
│       └── metrics.json
│
├── requirements.txt
├── README.md
└── .gitignore

The following are excluded from version control:

Dataset images
Virtual environment
Python cache files
Model checkpoint files
Reproducibility

A fixed random seed of 42 is used for reproducible experiments.

The main training configuration is:

Model: ResNet-18

Epochs: 10

Batch Size: 16

Learning Rate: 0.0001

Weight Decay: 0.0001

Optimizer: AdamW

Input Size: 224 × 224

Random Seed: 42
Version Control

The MRI dataset is not included in the GitHub repository.

The following files and directories are excluded using .gitignore:

dataset/

.venv/

venv/

env/

__pycache__/

*.pth

*.pt

*.onnx

outputs/checkpoints/

.vscode/

Training plots and evaluation metrics can be included in the GitHub repository.

Reproducibility Steps

To reproduce the baseline experiment:

Clone the Edge-Med repository.
Navigate to the MRI module:
cd Edge-Med/mri
Create a virtual environment:
python -m venv .venv
Activate the virtual environment.

Windows:

.venv\Scripts\activate

macOS/Linux:

source .venv/bin/activate
Install the dependencies:
pip install -r requirements.txt
Place the MRI dataset inside:
mri/dataset/
Verify the dataset structure.
Start training:
python src/train.py
Evaluate the trained model:
python src/evaluate.py
Perform single-image prediction:
python src/predict.py "path/to/mri_image.jpg"
Current Baseline

The current MRI implementation represents the baseline classification pipeline for the Edge-Med project.

The baseline uses:

ResNet-18 Transfer Learning
        +
Image Preprocessing
        +
Training Data Augmentation
        +
Class-Weighted Cross Entropy
        +
10 Epochs
        +
Test Evaluation

The baseline achieved:

Best Validation Accuracy: 74.52%

Test Accuracy: 74.66%
Limitations

The current implementation has several limitations:

The model uses only the MRI modality.
The current experiment is limited to 10 training epochs.
The ResNet-18 feature extractor is frozen in the baseline implementation.
Performance may vary depending on dataset characteristics.
The model shows lower recall for the Malignant class compared with the Healthy class.
The model has not been clinically validated.
The model should not be used as a standalone diagnostic system.
Future Improvements

Potential future improvements include:

Fine-tuning selected ResNet-18 layers
Improved data augmentation
Additional class-balancing techniques
Hyperparameter optimization
Learning-rate scheduling
Comparison with other CNN architectures
Improved feature extraction
Model explainability
GPU-based experimentation
Integration with other Edge-Med modalities
Edge-Med Integration

The MRI module forms one component of the broader Edge-Med multi-modal breast cancer detection system.

The overall project contains multiple imaging modalities, including:

                 Edge-Med
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
   Mammography  Ultrasound  Histopathology
        │           │           │
        └───────────┼───────────┘
                    │
                    ▼
                   MRI
                    │
                    ▼
          Multi-Modal Integration

The MRI component is maintained as an independent modality-specific pipeline so that it can be developed and evaluated separately before integration with the other Edge-Med components.