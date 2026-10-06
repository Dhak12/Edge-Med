import json

import torch

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DEVICE,
    BEST_MODEL_PATH,
    METRICS_DIR,
    PLOTS_DIR
)

from dataset import create_dataloaders
from model import create_model


def main():

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders()

    model = create_model()

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            ).cpu()

            all_predictions.extend(
                predictions.numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        output_dict=True
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(
        f"\nTest Accuracy: {accuracy:.4f}"
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=test_dataset.classes
        )
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics = {
        "test_accuracy": accuracy,
        "classification_report": report
    }

    with open(
        METRICS_DIR / "metrics.json",
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=test_dataset.classes,
        yticklabels=test_dataset.classes
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("MRI Classification Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "confusion_matrix.png",
        dpi=200
    )

    plt.close()


if __name__ == "__main__":
    main()