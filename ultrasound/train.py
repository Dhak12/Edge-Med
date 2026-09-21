import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from torch.utils.data import Dataset, DataLoader

from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate,
    RandAdjustContrast,
    RandGaussianNoise,
    RandZoom
)

from monai.networks.nets import DenseNet121


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"

IMG_SIZE = 224
BATCH_SIZE = 8

EPOCHS = 10
LEARNING_RATE = 0.00005

PATIENCE = 3

NUM_CLASSES = 3

CLASSES = [
    "benign",
    "malignant",
    "normal"
]

CLASS_TO_INDEX = {
    "benign": 0,
    "malignant": 1,
    "normal": 2
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# LOAD IMAGE PATHS
# ============================================================

image_paths = []
labels = []

for class_name in CLASSES:

    folder = os.path.join(
        DATA_DIR,
        class_name
    )

    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Missing folder: {folder}"
        )

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".tif",
                ".tiff"
            )
        ):

            image_paths.append(
                os.path.join(
                    folder,
                    filename
                )
            )

            labels.append(
                CLASS_TO_INDEX[class_name]
            )


# ============================================================
# DATASET INFORMATION
# ============================================================

print(
    f"\nDataset: {len(image_paths)} images"
)

print(
    f"Benign: {labels.count(0)} | "
    f"Malignant: {labels.count(1)} | "
    f"Normal: {labels.count(2)}"
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_paths, temp_paths, train_labels, temp_labels = (
    train_test_split(
        image_paths,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=SEED
    )
)

val_paths, test_paths, val_labels, test_labels = (
    train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        stratify=temp_labels,
        random_state=SEED
    )
)


print(
    f"Split: Train {len(train_paths)} | "
    f"Val {len(val_paths)} | "
    f"Test {len(test_paths)}"
)


# ============================================================
# MONAI AUGMENTATION
# ============================================================

train_transform = Compose([

    RandFlip(
        prob=0.5,
        spatial_axis=1
    ),

    RandRotate(
        range_x=np.pi / 18,
        prob=0.4
    ),

    RandAdjustContrast(
        gamma=(0.8, 1.2),
        prob=0.3
    ),

    RandGaussianNoise(
        mean=0.0,
        std=0.01,
        prob=0.2
    ),

    RandZoom(
        min_zoom=0.9,
        max_zoom=1.1,
        prob=0.2
    )
])


# ============================================================
# DATASET CLASS
# ============================================================

class BreastUltrasoundDataset(Dataset):

    def __init__(
        self,
        paths,
        labels,
        training=False
    ):

        self.paths = paths
        self.labels = labels
        self.training = training


    def __len__(self):

        return len(self.paths)


    def __getitem__(self, index):

        path = self.paths[index]

        label = self.labels[index]

        # Load ultrasound image as grayscale
        image = Image.open(path).convert("L")

        image = np.asarray(
            image,
            dtype=np.float32
        )

        # Normalize pixels
        image = image / 255.0

        # Convert to tensor
        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        # Add channel and resize
        image = image.unsqueeze(0)

        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False
        )

        image = image.squeeze(0)

        # MONAI augmentation
        if self.training:

            image = train_transform(
                image
            )

        # ----------------------------------------------------
        # Convert grayscale to 3 channels
        # Required for ImageNet pretrained DenseNet
        # ----------------------------------------------------

        image = image.repeat(
            3,
            1,
            1
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = BreastUltrasoundDataset(
    train_paths,
    train_labels,
    training=True
)

val_dataset = BreastUltrasoundDataset(
    val_paths,
    val_labels,
    training=False
)

test_dataset = BreastUltrasoundDataset(
    test_paths,
    test_labels,
    training=False
)


# ============================================================
# DATA LOADERS
# ============================================================

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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Device: {device}"
)


# ============================================================
# MODEL
# ============================================================

print(
    "Loading MONAI DenseNet121..."
)

model = DenseNet121(
    spatial_dims=2,
    in_channels=3,
    out_channels=NUM_CLASSES,
    pretrained=True
)

model = model.to(device)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

class_weights = (
    len(train_labels)
    /
    (
        NUM_CLASSES *
        class_counts
    )
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)


# ============================================================
# METRIC STORAGE
# ============================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

val_f1_scores = []


# ============================================================
# BEST MODEL VARIABLES
# ============================================================

best_f1 = 0.0

epochs_without_improvement = 0


# ============================================================
# TRAINING
# ============================================================

print("\nTraining Version 3...\n")


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAINING MODE
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    train_predictions = []
    train_targets = []


    for images, targets in train_loader:

        images = images.to(device)

        targets = targets.to(device)


        # Clear gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Calculate loss
        loss = criterion(
            outputs,
            targets
        )


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        running_loss += loss.item()


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        train_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        train_targets.extend(
            targets.cpu()
            .numpy()
        )


    # Average training loss
    train_loss = (
        running_loss /
        len(train_loader)
    )


    # Training accuracy
    train_accuracy = accuracy_score(
        train_targets,
        train_predictions
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    validation_loss = 0.0

    val_predictions = []
    val_targets = []


    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(device)

            targets = targets.to(device)


            outputs = model(images)


            loss = criterion(
                outputs,
                targets
            )


            validation_loss += loss.item()


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_predictions.extend(
                predictions.cpu()
                .numpy()
            )

            val_targets.extend(
                targets.cpu()
                .numpy()
            )


    # Average validation loss
    val_loss = (
        validation_loss /
        len(val_loader)
    )


    # Validation accuracy
    val_accuracy = accuracy_score(
        val_targets,
        val_predictions
    )


    # Macro F1
    val_f1 = f1_score(
        val_targets,
        val_predictions,
        average="macro"
    )


    # --------------------------------------------------------
    # STORE METRICS
    # --------------------------------------------------------

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )

    val_f1_scores.append(
        val_f1
    )


    # --------------------------------------------------------
    # LEARNING RATE UPDATE
    # --------------------------------------------------------

    scheduler.step(
        val_f1
    )


    # --------------------------------------------------------
    # SHORT OUTPUT
    # --------------------------------------------------------

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}% | "
        f"Val F1: {val_f1:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_f1:

        best_f1 = val_f1

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            "models/best_model_v3.pth"
        )

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print(
            "Early stopping."
        )

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        "models/best_model_v3.pth",
        map_location=device
    )
)

model.eval()


# ============================================================
# TESTING
# ============================================================

test_predictions = []
test_targets = []


with torch.no_grad():

    for images, targets in test_loader:

        images = images.to(device)


        outputs = model(
            images
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        test_predictions.extend(
            predictions.cpu()
            .numpy()
        )

        test_targets.extend(
            targets.numpy()
        )


# ============================================================
# FINAL METRICS
# ============================================================

test_accuracy = accuracy_score(
    test_targets,
    test_predictions
)

test_macro_f1 = f1_score(
    test_targets,
    test_predictions,
    average="macro"
)


print("\n==============================")
print("VERSION 3 - TEST RESULTS")
print("==============================")


print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Macro F1: "
    f"{test_macro_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    test_targets,
    test_predictions,
    target_names=CLASSES,
    digits=4,
    zero_division=0
)

print("\nClassification Report:")
print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_targets,
    test_predictions
)


print("Confusion Matrix:")
print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES
)

display.plot()

plt.title(
    "Version 3 - Breast Ultrasound"
)

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix_v3.png"
)

plt.close()


# ============================================================
# SAVE LOSS GRAPH
# ============================================================

plt.figure()

plt.plot(
    train_losses,
    label="Train Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Version 3 Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/loss_graph_v3.png"
)

plt.close()


# ============================================================
# SAVE ACCURACY GRAPH
# ============================================================

plt.figure()

plt.plot(
    train_accuracies,
    label="Train Accuracy"
)

plt.plot(
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Version 3 Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/accuracy_graph_v3.png"
)

plt.close()


# ============================================================
# SAVE F1 GRAPH
# ============================================================

plt.figure()

plt.plot(
    val_f1_scores,
    label="Validation Macro F1"
)

plt.xlabel("Epoch")

plt.ylabel("Macro F1")

plt.title(
    "Version 3 Macro F1"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/f1_graph_v3.png"
)

plt.close()


# ============================================================
# SAVE TEXT RESULTS
# ============================================================

with open(
    "results/results_v3.txt",
    "w"
) as file:

    file.write(
        "EdgeMed Version 3\n"
    )

    file.write(
        "=================\n\n"
    )

    file.write(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Test Macro F1: "
        f"{test_macro_f1:.4f}\n\n"
    )

    file.write(
        "Classification Report:\n"
    )

    file.write(
        report
    )

    file.write(
        "\nConfusion Matrix:\n"
    )

    file.write(
        str(cm)
    )


# ============================================================
# COMPLETED
# ============================================================

print(
    "\nBest model: "
    "models/best_model_v3.pth"
)

print(
    "Results: results/"
)

print(
    "\nVersion 3 completed."
)