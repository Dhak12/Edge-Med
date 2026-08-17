"""
================================================================================
EdgeMed :: Mammography Branch (CBIS-DDSM) — MONAI Training Script
================================================================================
Training target : local RTX 4070 (CUDA)
Deployment target: Jetson Orin Nano (ONNX -> TensorRT, later)
Framework        : MONAI (Medical Open Network for AI), built on PyTorch

--------------------------------------------------------------------------------
REQUIREMENTS  (save this block as requirements.txt, or just pip install below)
--------------------------------------------------------------------------------
torch>=2.1.0            # install the CUDA 12.x build for RTX 4070:
                         # pip install torch --index-url https://download.pytorch.org/whl/cu121
torchvision>=0.16.0
monai>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
Pillow>=10.0.0
scikit-learn>=1.3.0
tqdm>=4.66.0
onnx>=1.15.0
onnxruntime-gpu>=1.16.0

Quick install (RTX 4070, CUDA 12.1):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install monai pandas numpy pillow scikit-learn tqdm onnx onnxruntime-gpu

--------------------------------------------------------------------------------
DATASET — fill this in once your data is available
--------------------------------------------------------------------------------
"""

import os
import time
import copy
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from monai.data import Dataset as MonaiDataset
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized,
    RandFlipd, RandRotate90d, RandAdjustContrastd, ToTensord, EnsureTyped,
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

# ==============================================================================
# >>> ADD YOUR DATASET HERE <<<
# ------------------------------------------------------------------------------
# DATA_ROOT -> folder containing your images + csv/ subfolder
# TRAIN_CSV -> training case-description CSV, relative to DATA_ROOT
# TEST_CSV  -> test case-description CSV, relative to DATA_ROOT
#
# Example (local, RTX 4070 machine):
#   DATA_ROOT = "D:/EdgeMed/mammography_data"
#   TRAIN_CSV = "csv/calc_case_description_train_set.csv"
#   TEST_CSV  = "csv/calc_case_description_test_set.csv"
# ==============================================================================
DATA_ROOT = ""   # <-- FILL THIS IN
TRAIN_CSV = ""   # <-- FILL THIS IN
TEST_CSV  = ""   # <-- FILL THIS IN
# ==============================================================================

CONFIG = {
    "data_root": DATA_ROOT,
    "train_csv": TRAIN_CSV,
    "test_csv": TEST_CSV,
    "image_col": "cropped image file path",
    "label_col": "pathology",           # BENIGN / BENIGN_WITHOUT_CALLBACK / MALIGNANT
    "img_size": 224,
    "batch_size": 64,                   # RTX 4070 (12GB) comfortably handles 64 at 224x224
    "epochs": 25,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "val_split": 0.15,
    "num_workers": 4,
    "seed": 42,
    "checkpoint_dir": "./checkpoints",
    "onnx_export_path": "./checkpoints/mammography_densenet121.onnx",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# ------------------------------------------------------------------------------
# DATA PREP — build MONAI-style list-of-dicts and CSV loader
# ------------------------------------------------------------------------------
def build_data_list(csv_path, root):
    df = pd.read_csv(csv_path)
    df["image_path"] = df[CONFIG["image_col"]].astype(str).str.strip()
    df = df.dropna(subset=["image_path", CONFIG["label_col"]])
    data = []
    for _, row in df.iterrows():
        label = 1 if "MALIGNANT" in str(row[CONFIG["label_col"]]).upper() else 0
        data.append({"img": os.path.join(root, row["image_path"]), "label": label})
    return data


# ------------------------------------------------------------------------------
# MONAI TRANSFORMS
# ------------------------------------------------------------------------------
train_transforms = Compose([
    LoadImaged(keys=["img"], image_only=True),
    EnsureChannelFirstd(keys=["img"]),
    ScaleIntensityd(keys=["img"]),
    Resized(keys=["img"], spatial_size=(CONFIG["img_size"], CONFIG["img_size"])),
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
    EnsureTyped(keys=["img"], dtype=torch.float32),
])


# ------------------------------------------------------------------------------
# MODEL — MONAI DenseNet121 (2D, 3-channel, transfer-learned)
# ------------------------------------------------------------------------------
def build_model(num_classes=2):
    model = DenseNet121(
        spatial_dims=2,
        in_channels=3,
        out_channels=num_classes,
        pretrained=True,
    )
    return model


# ------------------------------------------------------------------------------
# TRAIN / EVAL LOOP
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
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        preds = outputs.argmax(dim=1).detach().cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(all_labels, all_probs)
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

    full_train_data = build_data_list(
        os.path.join(CONFIG["data_root"], CONFIG["train_csv"]), CONFIG["data_root"]
    )
    test_data = build_data_list(
        os.path.join(CONFIG["data_root"], CONFIG["test_csv"]), CONFIG["data_root"]
    )

    np.random.shuffle(full_train_data)
    val_size = int(len(full_train_data) * CONFIG["val_split"])
    val_data = full_train_data[:val_size]
    train_data = full_train_data[val_size:]

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

    model = build_model(num_classes=2).to(CONFIG["device"])
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
Device: cuda
GPU: NVIDIA GeForce RTX 4070
Train: 2136 | Val: 377 | Test: 645

Epoch 01/25 | train_loss 0.6598 acc 0.6041 | val_loss 0.6301 acc 0.6259 f1 0.6014 auc 0.6488 | 18.3s
Epoch 02/25 | train_loss 0.5872 acc 0.6733 | val_loss 0.5714 acc 0.6870 f1 0.6602 auc 0.7091 | 17.9s
Epoch 03/25 | train_loss 0.5241 acc 0.7228 | val_loss 0.5203 acc 0.7347 f1 0.7145 auc 0.7622 | 18.1s
...
Epoch 12/25 | train_loss 0.3327 acc 0.8544 | val_loss 0.3798 acc 0.8330 f1 0.8188 auc 0.8710 | 18.0s
...
Epoch 25/25 | train_loss 0.1721 acc 0.9336 | val_loss 0.2914 acc 0.8827 f1 0.8691 auc 0.9203 | 17.8s

=== Final Test Metrics (best val checkpoint) ===
      loss: 0.2984
       acc: 0.8798
 precision: 0.8672
    recall: 0.8531
        f1: 0.8601
       auc: 0.9147

ONNX model exported to ./checkpoints/mammography_densenet121.onnx (27.14 MB)
Next: convert ONNX -> TensorRT engine (trtexec --onnx=... --fp16) for Jetson Orin Nano.

NOTE: RTX 4070 training is roughly 2x faster per epoch than a Kaggle T4/P100
for this batch size. DenseNet121 is heavier than MobileNetV3 (~27MB vs ~6MB
ONNX), so for the final Jetson Orin Nano deployment you'll likely want to
either (a) prune/quantize this DenseNet121 to INT8 via TensorRT, or (b) swap
the backbone to something smaller (MobileNetV3, EfficientNet-B0) once you've
confirmed DenseNet121's accuracy ceiling on your actual data. Train on the
4070 with the heavier model first to establish a strong baseline, then
decide whether the accuracy/size tradeoff is worth optimizing further.
================================================================================
"""
