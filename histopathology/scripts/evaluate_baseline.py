import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import swin_t, Swin_T_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =========================
# Configuration
# =========================

TEST_CSV = "splits/test.csv"

MODEL_PATH = "checkpoints/swin_t_baseline_best.pth"

BATCH_SIZE = 8

NUM_CLASSES = 3

CLASS_NAMES = [
    "Normal",
    "Benign",
    "Malignant"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# Dataset
# =========================

class HistopathologyDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)

        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        image_path = self.data.iloc[idx]["path"]

        label = int(
            self.data.iloc[idx]["label"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return image, label


# =========================
# Transform
# =========================

weights = Swin_T_Weights.DEFAULT

test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])


# =========================
# Test Dataset
# =========================

test_dataset = HistopathologyDataset(
    TEST_CSV,
    transform=test_transform
)


# =========================
# Test DataLoader
# =========================

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)


# =========================
# Load Model
# =========================

print("\nLoading pretrained Swin-T...")

model = swin_t(
    weights=None
)

in_features = model.head.in_features

model.head = nn.Linear(
    in_features,
    NUM_CLASSES
)


# =========================
# Load Checkpoint
# =========================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


print("\n========================================")
print("EdgeMed Swin-T Baseline Evaluation")
print("========================================")

print("Device:", DEVICE)

print("Test samples:", len(test_dataset))

print("Checkpoint:", MODEL_PATH)

print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)

print(
    "Best validation loss:",
    f"{checkpoint['val_loss']:.4f}"
)

print(
    "Best validation accuracy:",
    f"{checkpoint['val_accuracy']:.4f}"
)

print("========================================\n")


# =========================
# Inference
# =========================

all_predictions = []

all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# =========================
# Metrics
# =========================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


# =========================
# Overall Results
# =========================

print("\n========================================")
print("Overall Test Results")
print("========================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"Macro F1 : {f1:.4f}"
)


# =========================
# Classification Report
# =========================

print("\n========================================")
print("Classification Report")
print("========================================")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n========================================")
print("Confusion Matrix")
print("========================================")

print(
    "Rows    = Actual"
)

print(
    "Columns = Predicted"
)

print()

print(
    "              Normal  Benign  Malignant"
)

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:<12}"
        f"{cm[i, 0]:>7}"
        f"{cm[i, 1]:>8}"
        f"{cm[i, 2]:>11}"
    )


print("\n========================================")
print("Evaluation Complete")
print("========================================")