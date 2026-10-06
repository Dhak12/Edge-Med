from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
)


def get_transforms():

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(degrees=10),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, eval_transform


def create_dataloaders():

    train_transform, eval_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    )