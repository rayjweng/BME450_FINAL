#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        default=None,
    )
    parser.add_argument("--output-dir", default="data/processed/food_beverage_binary")
    parser.add_argument("--seed", type=int, default=450)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    return parser.parse_args()


def resolve_source_dir(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value)

    metadata_path = Path("artifacts/data_sources.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        dataset_path = metadata["food_beverage_labels"]["path"]
        return Path(dataset_path)

    return Path(
        "/Users/elidubizh/.cache/kagglehub/datasets/tushar5harma/food-and-beverage-labels/versions/1"
    )


def main() -> None:
    args = parse_args()
    source_dir = resolve_source_dir(args.source_dir)
    labels_path = source_dir / "labels_(train).csv"
    image_dir = source_dir / "train"

    frame = pd.read_csv(labels_path)
    frame["image_path"] = frame["image_name"].map(lambda name: str(image_dir / name))

    missing = frame.loc[~frame["image_path"].map(lambda path: Path(path).exists())]
    if not missing.empty:
        raise FileNotFoundError(f"Missing {len(missing)} training images from {image_dir}")

    train_val, test = train_test_split(
        frame,
        test_size=args.test_size,
        stratify=frame["label"],
        random_state=args.seed,
    )

    relative_val_size = args.val_size / (1.0 - args.test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val["label"],
        random_state=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    class_names = ["beverage", "food"]
    (output_dir / "class_names.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    print(
        f"prepared manifests in {output_dir} "
        f"(train={len(train)}, val={len(val)}, test={len(test)})"
    )


if __name__ == "__main__":
    main()
