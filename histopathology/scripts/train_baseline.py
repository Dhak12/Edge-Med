import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import swin_t, Swin_T_Weights


# =========================
# Configuration
# =========================

TRAIN_CSV = "splits/train.csv"
VAL_CSV = "splits/val.csv"

BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

NUM_CLASSES = 3

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "swin_t_baseline_best.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "Normal",
    "Benign",
    "Malignant"
]


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

        # CSV column is "path"
        image_path = self.data.iloc[idx]["path"]

        # CSV column is "label"
        label = int(self.data.iloc[idx]["label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================
# Pretrained weights
# =========================

weights = Swin_T_Weights.DEFAULT


# =========================
# Transforms
# =========================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomVerticalFlip(p=0.5),

    transforms.RandomRotation(15),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])


# =========================
# Datasets
# =========================

train_dataset = HistopathologyDataset(
    TRAIN_CSV,
    transform=train_transform
)

val_dataset = HistopathologyDataset(
    VAL_CSV,
    transform=val_transform
)


# =========================
# DataLoaders
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================
# Model
# =========================

print("\nLoading pretrained Swin-T...")

model = swin_t(weights=weights)

# Original ImageNet classifier is replaced
# with a 3-class classifier.

in_features = model.head.in_features

model.head = nn.Linear(
    in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)


# =========================
# Loss
# =========================

criterion = nn.CrossEntropyLoss()


# =========================
# Optimizer
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# =========================
# Checkpoint directory
# =========================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# =========================
# Initial information
# =========================

print("\n========================================")
print("EdgeMed Swin-T Baseline")
print("========================================")

print("Device:", DEVICE)

print("Train samples:", len(train_dataset))

print("Validation samples:", len(val_dataset))

print("Classes:", CLASS_NAMES)

print("Batch size:", BATCH_SIZE)

print("Epochs:", NUM_EPOCHS)

print("Learning rate:", LEARNING_RATE)

print("========================================\n")


# =========================
# Training
# =========================

best_val_loss = float("inf")


for epoch in range(NUM_EPOCHS):

    # ====================================
    # Training phase
    # ====================================

    model.train()

    running_train_loss = 0.0

    correct_train = 0

    total_train = 0


    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        # Clear previous gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        # Accumulate loss
        running_train_loss += (
            loss.item() * images.size(0)
        )


        # Calculate predictions
        predictions = torch.argmax(
            outputs,
            dim=1
        )


        # Calculate correct predictions
        correct_train += (
            predictions == labels
        ).sum().item()


        total_train += labels.size(0)


    # Average training loss
    train_loss = (
        running_train_loss /
        total_train
    )


    # Training accuracy
    train_accuracy = (
        correct_train /
        total_train
    )


    # ====================================
    # Validation phase
    # ====================================

    model.eval()

    running_val_loss = 0.0

    correct_val = 0

    total_val = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            # Forward pass
            outputs = model(images)


            # Validation loss
            loss = criterion(
                outputs,
                labels
            )


            running_val_loss += (
                loss.item() *
                images.size(0)
            )


            # Predictions
            predictions = torch.argmax(
                outputs,
                dim=1
            )


            # Correct predictions
            correct_val += (
                predictions == labels
            ).sum().item()


            total_val += labels.size(0)


    # Average validation loss
    val_loss = (
        running_val_loss /
        total_val
    )


    # Validation accuracy
    val_accuracy = (
        correct_val /
        total_val
    )


    # ====================================
    # Epoch results
    # ====================================

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.4f}"
    )


    # ====================================
    # Save best model
    # ====================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_loss":
                    val_loss,

                "val_accuracy":
                    val_accuracy,

                "class_names":
                    CLASS_NAMES
            },
            BEST_MODEL_PATH
        )

        print(
            f"✓ Best model saved to:"
        )

        print(
            f"  {BEST_MODEL_PATH}"
        )


    print("----------------------------------------")


# =========================
# Training complete
# =========================

print("\n========================================")

print("Training complete.")

print("========================================")

print(
    "Best model:"
)

print(
    BEST_MODEL_PATH
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print("========================================")