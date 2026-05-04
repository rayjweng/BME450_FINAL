from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms 


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ManifestImageDataset(Dataset):
    def __init__(self, manifest_path: str | Path, transform=None) -> None:
        self.frame = pd.read_csv(manifest_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(image_size: int, augmentation: str):
    train_steps = [transforms.Resize((image_size, image_size))]
    if augmentation == "light":
        train_steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
        )
    elif augmentation == "heavy":
        train_steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.02,
                ),
            ]
        )
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_steps = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(train_steps), eval_steps


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_size: int
    val_size: int
    test_size: int


def build_dataloaders(
    train_manifest: str | Path,
    val_manifest: str | Path,
    test_manifest: str | Path,
    image_size: int,
    augmentation: str,
    batch_size: int,
    num_workers: int,
) -> DataBundle:
    train_transform, eval_transform = build_transforms(image_size, augmentation)

    train_dataset = ManifestImageDataset(train_manifest, transform=train_transform)
    val_dataset = ManifestImageDataset(val_manifest, transform=eval_transform)
    test_dataset = ManifestImageDataset(test_manifest, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
    )

