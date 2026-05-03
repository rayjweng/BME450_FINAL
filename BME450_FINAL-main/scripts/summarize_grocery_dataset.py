#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path",
        default=None,
    )
    parser.add_argument("--output-path", default="artifacts/grocery_summary.md")
    return parser.parse_args()


def resolve_csv_path(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value)

    metadata_path = Path("artifacts/data_sources.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return Path(metadata["grocery_transactions"]["path"]) / "groceries.csv"

    return Path("/Users/elidubizh/.cache/kagglehub/datasets/umairaslam/grocery/versions/1/groceries.csv")


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(resolve_csv_path(args.csv_path))

    transactions = frame["Items"].fillna("").tolist()
    counts = Counter()
    basket_sizes = []
    for row in transactions:
        items = [item.strip() for item in row.split(",") if item.strip()]
        basket_sizes.append(len(items))
        counts.update(items)

    top_items = counts.most_common(15)
    lines = [
        "# Grocery Dataset Summary",
        "",
        "This dataset is included because it was supplied for the project, but it is not used for CNN training.",
        "It contains transaction baskets rather than labeled images, so it supports domain context instead of image classification.",
        "",
        f"- Transactions: {len(transactions)}",
        f"- Unique items: {len(counts)}",
        f"- Average basket size: {sum(basket_sizes) / len(basket_sizes):.2f}",
        "",
        "## Top 15 items",
        "",
    ]
    lines.extend([f"- {item}: {count}" for item, count in top_items])

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
