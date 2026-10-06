from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BEST_MODEL_PATH,
    CHECKPOINT_DIR,
    PLOTS_DIR,
    SEED,
)

from dataset import create_dataloaders
from model import create_model
from utils import set_seed, plot_training_history


def calculate_class_weights(dataset):

    labels = np.array(dataset.targets)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )

    return torch.tensor(
        class_weights,
        dtype=torch.float32
    ).to(DEVICE)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def main():

    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("EDGE-MED MRI CLASSIFICATION")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders()

    print("\nClasses:")
    print(train_dataset.class_to_idx)

    print(f"\nTraining images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model()

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # CLASS-WEIGHTED LOSS
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_dataset
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # Save best model
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": train_dataset.classes,
                    "val_accuracy": val_accuracy,
                    "epoch": epoch + 1
                },
                BEST_MODEL_PATH
            )

            print(
                f"✓ Best model saved "
                f"(validation accuracy: "
                f"{val_accuracy:.4f})"
            )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    plot_training_history(
        history,
        PLOTS_DIR
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Model saved to: "
        f"{BEST_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()