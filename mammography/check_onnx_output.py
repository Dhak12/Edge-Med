"""
EdgeMed :: Mammography — ONNX Sanity Check
============================================
Loads the exported mammography_densenet121.onnx and runs it against a real
image from your dataset to confirm the model actually produces sensible
output: one score per class (benign / malignant / normal).
"""

import os
import glob
import numpy as np
import onnxruntime as ort
from PIL import Image

# ==============================================================================
# CONFIG
# ==============================================================================
ONNX_PATH = "./checkpoints/mammography_densenet121.onnx"
DATA_ROOT = "C:/Users/DELL/Desktop/dhakshita/7th SEM/Edge Med/mamography/archive (2)/CLAHE_images"   # <-- FILL THIS IN, same path as train_mammography_monai.py
IMG_SIZE = 96     # 96 for QUICK_TEST_MODE runs, 224 for full runs
CLASS_NAMES = ["benign", "malignant", "normal"]
# ==============================================================================


def preprocess_image(img_path, img_size):
    img = Image.open(img_path).convert("L")
    img = img.resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.stack([arr, arr, arr], axis=0)
    arr = np.expand_dims(arr, axis=0)
    return arr


def pick_sample_image(data_root):
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_root, class_name)
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            matches = glob.glob(os.path.join(class_dir, ext))
            if matches:
                return matches[0], class_name
    return None, None


def main():
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(f"{ONNX_PATH} not found. Run train_mammography_monai.py first.")
    if not DATA_ROOT:
        raise ValueError("Fill in DATA_ROOT near the top of this file.")

    sample_path, true_class = pick_sample_image(DATA_ROOT)
    if sample_path is None:
        raise FileNotFoundError(f"No sample image found under {DATA_ROOT}. Check DATA_ROOT.")

    print(f"Testing with: {sample_path}")
    print(f"(True label, from folder name): {true_class}")

    input_tensor = preprocess_image(sample_path, IMG_SIZE)

    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    logits = session.run([output_name], {input_name: input_tensor})[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    predicted_idx = int(np.argmax(probs))

    print("\n=== Prediction ===")
    for name, p in zip(CLASS_NAMES, probs):
        marker = " <-- predicted" if name == CLASS_NAMES[predicted_idx] else ""
        print(f"  {name:>10}: {p:.4f}{marker}")

    print(f"\nPredicted class: {CLASS_NAMES[predicted_idx]}")
    print(f"Actual class (from folder): {true_class}")
    print("\nNOTE: trained only 2 quick-test epochs on 20 images/class, so a "
          "wrong prediction here is expected, not a bug.")


if __name__ == "__main__":
    main()