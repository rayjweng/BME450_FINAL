from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn 
from tqdm import tqdm

from pantryvision.utils import ensure_dir, write_json


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    f1: float


def build_optimizer(name: str, parameters, learning_rate: float, weight_decay: float):
    lowered = name.lower()
    if lowered == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if lowered == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if lowered == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    predictions = []
    targets = []

    for inputs, labels in tqdm(loader, leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        predictions.extend(outputs.argmax(dim=1).detach().cpu().numpy())
        targets.extend(labels.detach().cpu().numpy())

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average="weighted")
    return EpochMetrics(loss=average_loss, accuracy=accuracy, f1=f1)


def train_model(
    model,
    data_bundle,
    device,
    epochs: int,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    early_stopping_patience: int,
    output_dir: str | Path,
):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = build_optimizer(
        optimizer_name,
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1": [],
        "val_f1": [],
    }

    best_val_loss = float("inf")
    best_path = Path(output_dir) / "best_model.pt"
    patience = 0

    for epoch_index in range(epochs):
        train_metrics = run_epoch(model, data_bundle.train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, data_bundle.val_loader, criterion, device)

        history["train_loss"].append(train_metrics.loss)
        history["val_loss"].append(val_metrics.loss)
        history["train_accuracy"].append(train_metrics.accuracy)
        history["val_accuracy"].append(val_metrics.accuracy)
        history["train_f1"].append(train_metrics.f1)
        history["val_f1"].append(val_metrics.f1)

        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break

        print(
            f"epoch={epoch_index + 1} "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f}"
        )

    model.load_state_dict(torch.load(best_path, map_location=device))
    return history, best_path


def evaluate_model(model, loader, class_names, device):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            targets.extend(labels.numpy())

    accuracy = accuracy_score(targets, predictions)
    weighted_f1 = f1_score(targets, predictions, average="weighted")
    matrix = confusion_matrix(targets, predictions)
    report = classification_report(
        targets,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


def save_learning_curves(history: dict[str, list[float]], output_dir: str | Path) -> None:
    ensure_dir(output_dir)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["train_accuracy"], label="Train Accuracy")
    axes[1].plot(history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(Path(output_dir) / "learning_curves.png", dpi=200)
    plt.close(figure)


def save_confusion_matrix(matrix, class_names, output_dir: str | Path) -> None:
    ensure_dir(output_dir)
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        np.array(matrix),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix")
    figure.tight_layout()
    figure.savefig(Path(output_dir) / "confusion_matrix.png", dpi=200)
    plt.close(figure)


def save_experiment_summary(
    output_dir: str | Path,
    experiment_config: dict,
    dataset_sizes: dict,
    history: dict[str, list[float]],
    test_metrics: dict,
) -> None:
    summary = {
        "experiment": experiment_config,
        "dataset_sizes": dataset_sizes,
        "history": history,
        "test_metrics": test_metrics,
    }
    write_json(Path(output_dir) / "summary.json", summary)

