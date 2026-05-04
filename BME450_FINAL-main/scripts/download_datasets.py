#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import kagglehub 


DATASETS = {
    "grocery_transactions": "umairaslam/grocery",
    "food_beverage_labels": "tushar5harma/food-and-beverage-labels",
}


def main() -> None:
    output = {}
    for name, slug in DATASETS.items():
        path = kagglehub.dataset_download(slug)
        output[name] = {"slug": slug, "path": path}
        print(f"{name}: {path}")

    destination = Path("artifacts") / "data_sources.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"saved metadata to {destination}")


if __name__ == "__main__":
    main()

