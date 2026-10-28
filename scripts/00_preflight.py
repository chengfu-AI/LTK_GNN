#!/usr/bin/env python
"""Audit all 2021–2022 inputs before any model or dataset is written."""

from __future__ import annotations

import argparse
import csv
import gzip
from collections import Counter
from datetime import datetime
from pathlib import Path

from _bootstrap import default_config
from ltkg_2021_2022.config import format_utc, load_settings, parse_utc
from ltkg_2021_2022.dataset import _label_files
from ltkg_2021_2022.io import write_json


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pm10-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def label_time_counts(settings) -> dict:
    result = {}
    for height, path in _label_files(settings):
        times: set[str] = set()
        bad_rows = 0
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    end = parse_utc(row["end_time_utc"])
                except (KeyError, ValueError):
                    bad_rows += 1
                    continue
                if settings.contains(end):
                    times.add(format_utc(end))
        result[str(height)] = {"file": str(path), "distinct_2021_2022_times": len(times), "bad_rows": bad_rows}
    return result


def main() -> None:
    args = arguments()
    settings = load_settings(
        args.config,
        data_root=args.data_root,
        pm10_root=args.pm10_root,
        output_root=args.output_root,
        allow_missing_roots=True,
    )
    required = {
        "station_file": settings.station_file,
        "node_terrain_file": settings.node_terrain_file,
        "edge_terrain_file": settings.edge_terrain_file,
    }
    files = {name: {"path": str(path), "exists": path.is_file()} for name, path in required.items()}
    weather = {}
    for variable in settings.weather_variables:
        directory = settings.weather_root / variable
        matching = sum(1 for path in directory.glob("*.npy") if len(path.stem) == 10 and path.stem.isdigit() and settings.contains(datetime.strptime(path.stem, "%Y%m%d%H"))) if directory.is_dir() else 0
        weather[variable] = {"path": str(directory), "exists": directory.is_dir(), "hourly_files_2021_2022": matching}
    try:
        labels = label_time_counts(settings)
    except FileNotFoundError as exc:
        labels = {
            str(height): {"file": str(settings.labels_root / f"dynamic_edges_{height:05d}m.csv.gz"), "distinct_2021_2022_times": 0, "bad_rows": 0}
            for height in settings.release_heights_agl_m
        }
        label_error = str(exc)
    else:
        label_error = None
    report = {
        "study_period_utc": [format_utc(settings.start), format_utc(settings.end)],
        "inputs": files,
        "weather": weather,
        "labels": labels,
        "pm10_root": str(settings.pm10_root),
        "status": "ready",
        "notes": [
            "This audit never accepts 2023–2024 timestamps as valid study samples.",
            "A label time can only become a graph sample when all weather fields and all 20 PM10 values are available.",
        ],
    }
    write_json(settings.output_root / "preflight_report.json", report)
    print(f"Wrote: {settings.output_root / 'preflight_report.json'}")
    print("Label time counts:", {height: row["distinct_2021_2022_times"] for height, row in labels.items()})
    missing = [name for name, result in files.items() if not result["exists"]]
    missing += [f"weather/{name}" for name, result in weather.items() if not result["exists"] or result["hourly_files_2021_2022"] == 0]
    missing += [f"hysplit_label_{height}m" for height, result in labels.items() if result["distinct_2021_2022_times"] == 0]
    if not settings.pm10_root.is_dir():
        missing.append("pm10_root")
    if missing:
        report["status"] = "incomplete"
        report["missing_inputs"] = missing
        if label_error:
            report["label_error"] = label_error
        write_json(settings.output_root / "preflight_report.json", report)
        print("预检完成，但当前数据尚不完整：" + ", ".join(missing))
        print("详情见：" + str(settings.output_root / "preflight_report.json"))
    else:
        print("预检通过：2021–2022 年建模输入完整。")


if __name__ == "__main__":
    main()
