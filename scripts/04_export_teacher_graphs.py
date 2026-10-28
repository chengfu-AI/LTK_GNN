#!/usr/bin/env python
"""Export ERA5-predicted dynamic transport kernels for the 2021–2022 samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import default_config
from ltkg_2021_2022.config import load_settings
from ltkg_2021_2022.dataset import load_manifest
from ltkg_2021_2022.io import write_csv, write_json
from ltkg_2021_2022.model import load_teacher_run, predict_kernel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config()); parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pm10-root", type=Path); parser.add_argument("--output-root", type=Path)
    parser.add_argument("--variant", default="full"); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("all", "train", "valid", "test"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    settings = load_settings(args.config, data_root=args.data_root, pm10_root=args.pm10_root, output_root=args.output_root)
    model, variables, mean, std, _ = load_teacher_run(settings, args.variant, args.seed)
    rows = load_manifest(settings)
    if args.split != "all": rows = [row for row in rows if row["split"] == args.split]
    output = settings.output_root / "teacher_graphs" / args.variant / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for number, row in enumerate(rows, start=1):
        name = f"{row['weather_time_key']}.npz"; path = output / name
        if args.overwrite or not path.is_file():
            np.savez_compressed(path, transport_weight=predict_kernel(settings, row, model, variables, mean, std))
        records.append({"end_time_utc": row["end_time_utc"], "split": row["split"], "graph_file": name})
        if number % 100 == 0 or number == len(rows): print(f"Exported {number}/{len(rows)}")
    write_csv(output / "graph_manifest.csv", records)
    write_json(output / "export_metadata.json", {"variant": args.variant, "seed": args.seed, "rows": len(rows), "split": args.split})
    print(f"Graphs written to: {output}")


if __name__ == "__main__":
    main()
