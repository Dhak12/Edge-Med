import sys

import torch
from PIL import Image
from torchvision import transforms

from config import (
    DEVICE,
    IMAGE_SIZE,
    BEST_MODEL_PATH
)

from model import create_model


def predict(image_path):

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

    transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = probabilities.argmax(
            dim=1
        ).item()

    class_names = checkpoint[
        "class_names"
    ]

    predicted_label = class_names[
        predicted_class
    ]

    confidence = probabilities[
        0,
        predicted_class
    ].item()

    print("\n" + "=" * 50)
    print("MRI PREDICTION")
    print("=" * 50)

    print(
        f"Prediction: {predicted_label}"
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print("\nClass probabilities:")

    for name, probability in zip(
        class_names,
        probabilities[0]
    ):

        print(
            f"{name}: "
            f"{probability.item() * 100:.2f}%"
        )


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "python src/predict.py "
            "path/to/image.jpg"
        )

        sys.exit(1)

    predict(sys.argv[1])
