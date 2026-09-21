import torch
from PIL import Image
import numpy as np

from monai.transforms import Resize
from monai.networks.nets import DenseNet121


# ==========================================
# SETTINGS
# ==========================================
IMAGE_PATH = "../data/benign/00001.bmp"
MODEL_PATH = "../models/best_model.pth"

IMG_SIZE = 224

CLASSES = [
    "benign",
    "malignant",
    "normal"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# LOAD MODEL
# ==========================================

model = DenseNet121(
    spatial_dims=2,
    in_channels=1,
    out_channels=3
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

image = Image.open(IMAGE_PATH).convert("L")

image = np.array(image).astype(np.float32) / 255.0

# Add channel dimension
image = np.expand_dims(image, axis=0)

# Resize to 224 x 224
resize = Resize((IMG_SIZE, IMG_SIZE))
image = resize(image)

# Convert to tensor
image = torch.as_tensor(
    image,
    dtype=torch.float32
)

# Add batch dimension
image = image.unsqueeze(0)

image = image.to(DEVICE)


# ==========================================
# PREDICTION
# ==========================================

with torch.no_grad():

    output = model(image)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    predicted_index = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = (
        probabilities[0][predicted_index].item()
        * 100
    )


# ==========================================
# RESULT
# ==========================================

predicted_class = CLASSES[predicted_index]

print()
print("==============================")
print("   BREAST ULTRASOUND RESULT")
print("==============================")

print(f"Prediction : {predicted_class.upper()}")
print(f"Confidence : {confidence:.2f}%")

print("==============================")