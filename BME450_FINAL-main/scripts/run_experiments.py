#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pantryvision.data import build_dataloaders
from pantryvision.models import build_model
from pantryvision.training import (
    evaluate_model,
    save_confusion_matrix,
    save_experiment_summary,
    save_learning_curves,
    train_model,
)
from pantryvision.utils import ensure_dir, load_yaml, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--output-root", default="artifacts/experiments")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def verify_manifests(dataset_config: dict) -> None:
    required = [
        dataset_config["train_manifest"],
        dataset_config["val_manifest"],
        dataset_config["test_manifest"],
    ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        missing_list = "\n".join(missing)
        raise FileNotFoundError(
            "Missing dataset manifests. Run scripts/prepare_image_dataset.py first.\n"
            f"{missing_list}"
        )


def build_results_table(records: list[dict], output_root: Path) -> None:
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame = frame.sort_values(by=["test_accuracy", "test_weighted_f1"], ascending=False)
    frame.to_csv(output_root / "results.csv", index=False)
    print(frame.to_string(index=False))


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    dataset_config = config["dataset"]
    output_root = ensure_dir(args.output_root)
    verify_manifests(dataset_config)

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
        output_root = ensure_dir(args.output_root)

    experiments = config["experiments"]
    if args.limit is not None:
        experiments = experiments[: args.limit]

    results = []
    for experiment in experiments:
        print(f"running {experiment['name']} on {device}")
        experiment_dir = ensure_dir(output_root / experiment["name"])

        data_bundle = build_dataloaders(
            train_manifest=dataset_config["train_manifest"],
            val_manifest=dataset_config["val_manifest"],
            test_manifest=dataset_config["test_manifest"],
            image_size=dataset_config["image_size"],
            augmentation=experiment["augmentation"],
            batch_size=experiment["batch_size"],
            num_workers=config["num_workers"],
        )

        model = build_model(
            architecture=experiment["architecture"],
            num_classes=len(dataset_config["class_names"]),
            pretrained=True,
            freeze_backbone=config["training_defaults"]["freeze_backbone"],
        )
        model = model.to(device)

        history, best_path = train_model(
            model=model,
            data_bundle=data_bundle,
            device=device,
            epochs=config["training_defaults"]["epochs"],
            optimizer_name=experiment["optimizer"],
            learning_rate=experiment["learning_rate"],
            weight_decay=config["training_defaults"]["weight_decay"],
            label_smoothing=config["training_defaults"]["label_smoothing"],
            early_stopping_patience=config["training_defaults"]["early_stopping_patience"],
            output_dir=experiment_dir,
        )

        test_metrics = evaluate_model(
            model,
            data_bundle.test_loader,
            dataset_config["class_names"],
            device,
        )

        save_learning_curves(history, experiment_dir)
        save_confusion_matrix(
            test_metrics["confusion_matrix"],
            dataset_config["class_names"],
            experiment_dir,
        )
        save_experiment_summary(
            experiment_dir,
            experiment,
            {
                "train": data_bundle.train_size,
                "val": data_bundle.val_size,
                "test": data_bundle.test_size,
                "best_model_path": str(best_path),
            },
            history,
            test_metrics,
        )

        results.append(
            {
                "name": experiment["name"],
                "architecture": experiment["architecture"],
                "optimizer": experiment["optimizer"],
                "learning_rate": experiment["learning_rate"],
                "batch_size": experiment["batch_size"],
                "augmentation": experiment["augmentation"],
                "test_accuracy": test_metrics["accuracy"],
                "test_weighted_f1": test_metrics["weighted_f1"],
            }
        )

    build_results_table(results, output_root)


if __name__ == "__main__":
    main()

