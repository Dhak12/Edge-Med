"""
================================================================================
EdgeMed :: Mammography Branch — MONAI Training Script (folder-per-class data)
================================================================================
Training target : local machine (CPU now, RTX 4070 later)
Deployment target: Jetson Orin Nano (ONNX -> TensorRT, later)
Framework        : MONAI (Medical Open Network for AI), built on PyTorch

Dataset layout expected (matches your pendrive):
    DATA_ROOT/
        benign/     *.png / *.jpg
        malignant/  *.png / *.jpg
        normal/     *.png / *.jpg

--------------------------------------------------------------------------------
REQUIREMENTS  (copy into requirements.txt, or pip install directly)
--------------------------------------------------------------------------------
torch>=2.1.0             # CPU now: pip install torch torchvision
                          # RTX 4070 later: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
torchvision>=0.16.0
monai>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
Pillow>=10.0.0
scikit-learn>=1.3.0
tqdm>=4.66.0
onnx>=1.15.0
onnxruntime>=1.16.0      # swap for onnxruntime-gpu once on the RTX 4070
================================================================================
"""

import os
import glob
import time
import copy
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from monai.data import Dataset as MonaiDataset
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized,
    RandFlipd, RandRotate90d, RandAdjustContrastd, EnsureTyped, RepeatChanneld,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

# ==============================================================================
# >>> ADD YOUR DATASET HERE <<<
# ------------------------------------------------------------------------------
# DATA_ROOT must point directly at the folder that CONTAINS benign/ malignant/
# normal/ subfolders — i.e. your "CLAHE_images" folder.
#
# From your screenshot, on Windows this looks like:
#   DATA_ROOT = "D:/MAMOGRPHY/archive (2)/CLAHE_images"
# ==============================================================================
ONNX_PATH = "./checkpoints/mammography_densenet121.onnx"
DATA_ROOT = "C:/Users/DELL/Desktop/dhakshita/7th SEM/Edge Med/mamography/archive (2)/CLAHE_images"   # <-- FILL THIS IN, e.g. "D:/MAMOGRPHY/archive (2)/CLAHE_images"
# ==============================================================================

CLASS_NAMES = ["benign", "malignant", "normal"]   # folder names = class names, in label order

# ==============================================================================
# QUICK_TEST_MODE — leave True while on CPU. Shrinks everything so you can
# verify the pipeline runs end-to-end in minutes. Flip to False on the RTX 4070.
# ==============================================================================
QUICK_TEST_MODE = True

CONFIG = {
    "data_root": DATA_ROOT,
    "img_size": 96 if QUICK_TEST_MODE else 224,
    "batch_size": 8 if QUICK_TEST_MODE else 64,
    "epochs": 2 if QUICK_TEST_MODE else 25,
    "max_samples_per_class": 20 if QUICK_TEST_MODE else None,  # caps data for the smoke test
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "val_split": 0.15,
    "test_split": 0.15,
    "num_workers": 0 if QUICK_TEST_MODE else 4,
    "seed": 42,
    "checkpoint_dir": "./checkpoints",
    "onnx_export_path": "./checkpoints/mammography_densenet121.onnx",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# ------------------------------------------------------------------------------
# DATA PREP — scan benign/ malignant/ normal/ subfolders directly (no CSV)
# ------------------------------------------------------------------------------
def build_data_list(data_root, max_per_class=None):
    data = []
    exts = ("*.png", "*.jpg", "*.jpeg")
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_root, class_name)
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(class_dir, ext)))
        if not files:
            print(f"[!] WARNING: no images found in {class_dir} — check DATA_ROOT and folder names.")
        if max_per_class:
            np.random.shuffle(files)
            files = files[:max_per_class]
        for f in files:
            data.append({"img": f, "label": label_idx})
        print(f"  {class_name}: {len(files)} images")
    return data


# ------------------------------------------------------------------------------
# MONAI TRANSFORMS
# ------------------------------------------------------------------------------
# NOTE: CLAHE mammography images are grayscale (1 channel). DenseNet121 is
# pretrained on 3-channel ImageNet, so RepeatChanneld duplicates the single
# channel into 3 identical channels — the standard fix for feeding grayscale
# medical images into an RGB-pretrained backbone.
train_transforms = Compose([
    LoadImaged(keys=["img"], image_only=True),
    EnsureChannelFirstd(keys=["img"]),
    ScaleIntensityd(keys=["img"]),
    Resized(keys=["img"], spatial_size=(CONFIG["img_size"], CONFIG["img_size"])),
    RepeatChanneld(keys=["img"], repeats=3),
    RandFlipd(keys=["img"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["img"], prob=0.5),
    RandAdjustContrastd(keys=["img"], prob=0.3),
    EnsureTyped(keys=["img"], dtype=torch.float32),
])

eval_transforms = Compose([
    LoadImaged(keys=["img"], image_only=True),
    EnsureChannelFirstd(keys=["img"]),
    ScaleIntensityd(keys=["img"]),
    Resized(keys=["img"], spatial_size=(CONFIG["img_size"], CONFIG["img_size"])),
    RepeatChanneld(keys=["img"], repeats=3),
    EnsureTyped(keys=["img"], dtype=torch.float32),
])


# ------------------------------------------------------------------------------
# MODEL — MONAI DenseNet121, 3-class (benign / malignant / normal)
# ------------------------------------------------------------------------------
def build_model(num_classes=3):
    return DenseNet121(spatial_dims=2, in_channels=3, out_channels=num_classes, pretrained=True)


# ------------------------------------------------------------------------------
# TRAIN / EVAL LOOP (multi-class metrics)
# ------------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    torch.set_grad_enabled(is_train)

    for batch in tqdm(loader, leave=False):
        imgs = batch["img"].to(CONFIG["device"])
        labels = batch["label"].to(CONFIG["device"])

        if is_train:
            optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        preds = outputs.argmax(dim=1).detach().cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return {"loss": epoch_loss, "acc": acc, "precision": precision,
            "recall": recall, "f1": f1, "auc": auc}


def main():
    if not CONFIG["data_root"]:
        raise ValueError(
            "DATA_ROOT is empty. Fill in the '>>> ADD YOUR DATASET HERE <<<' "
            "block near the top of this file before running."
        )

    print(f"Device: {CONFIG['device']}")
    if CONFIG["device"] == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Scanning dataset...")
    all_data = build_data_list(CONFIG["data_root"], CONFIG["max_samples_per_class"])
    labels = [d["label"] for d in all_data]

    train_data, temp_data, train_labels, temp_labels = train_test_split(
        all_data, labels, test_size=CONFIG["val_split"] + CONFIG["test_split"],
        stratify=labels, random_state=CONFIG["seed"]
    )
    relative_test = CONFIG["test_split"] / (CONFIG["val_split"] + CONFIG["test_split"])
    val_data, test_data = train_test_split(
        temp_data, test_size=relative_test, stratify=temp_labels, random_state=CONFIG["seed"]
    )

    train_ds = MonaiDataset(train_data, transform=train_transforms)
    val_ds = MonaiDataset(val_data, transform=eval_transforms)
    test_ds = MonaiDataset(test_data, transform=eval_transforms)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                               num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                             num_workers=CONFIG["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                              num_workers=CONFIG["num_workers"], pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model = build_model(num_classes=len(CLASS_NAMES)).to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    best_f1 = 0.0
    best_state = None
    history = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
              f"train_loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
              f"val_loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} "
              f"f1 {val_metrics['f1']:.4f} auc {val_metrics['auc']:.4f} | {elapsed:.1f}s")

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(CONFIG["checkpoint_dir"], "best_model.pt"))

    with open(os.path.join(CONFIG["checkpoint_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, test_loader, criterion, optimizer=None)
    print("\n=== Final Test Metrics (best val checkpoint) ===")
    for k, v in test_metrics.items():
        print(f"{k:>10}: {v:.4f}")

    model.eval()
    dummy_input = torch.randn(1, 3, CONFIG["img_size"], CONFIG["img_size"]).to(CONFIG["device"])
    torch.onnx.export(
        model, dummy_input, CONFIG["onnx_export_path"],
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )
    onnx_size_mb = os.path.getsize(CONFIG["onnx_export_path"]) / (1024 * 1024)
    print(f"\nONNX model exported to {CONFIG['onnx_export_path']} ({onnx_size_mb:.2f} MB)")
    print("Next: convert ONNX -> TensorRT engine (trtexec --onnx=... --fp16) for Jetson Orin Nano.")


if __name__ == "__main__":
    main()


"""
================================================================================
SAMPLE / PROBABLE OUTPUT  (illustrative — real numbers depend on your data)
================================================================================
Device: cpu
Scanning dataset...
  benign: 20 images
  malignant: 20 images
  normal: 20 images
Train: 42 | Val: 9 | Test: 9

Epoch 01/2 | train_loss 1.0821 acc 0.4286 | val_loss 1.0512 acc 0.4444 f1 0.3981 auc 0.5920 | 12.4s
Epoch 02/2 | train_loss 0.9247 acc 0.5714 | val_loss 0.9884 acc 0.5556 f1 0.5312 auc 0.6540 | 11.9s

=== Final Test Metrics (best val checkpoint) ===
      loss: 0.9765
       acc: 0.5556
 precision: 0.5417
    recall: 0.5556
        f1: 0.5312
       auc: 0.6472

ONNX model exported to ./checkpoints/mammography_densenet121.onnx (27.14 MB)
Next: convert ONNX -> TensorRT engine (trtexec --onnx=... --fp16) for Jetson Orin Nano.

NOTE: This is QUICK_TEST_MODE output (2 epochs, 20 images/class) — it's only
meant to confirm the pipeline runs end-to-end. Numbers are close to random
guessing (33% for 3 classes) because there isn't enough data/epochs to learn
anything yet. Once on the RTX 4070, set QUICK_TEST_MODE = False to train on
the full dataset for 25 epochs — expect accuracy/AUC to climb well above this.
================================================================================
"""