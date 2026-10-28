#!/usr/bin/env python
"""Build and publish the complete 2021-01 to 2022-10 graph dataset.

Run this file directly after 00_cdump_to_graph_labels.py and 00_preflight.py.
It uses the paths in config/study_2021_2022.json, verifies every expected
label/ERA5/PM10 input, and replaces outputs only after a complete new dataset
has been built successfully.  No command-line suffixes are needed.
"""

from __future__ import annotations

from _bootstrap import default_config
from ltkg_2021_2022.config import load_settings
from ltkg_2021_2022.dataset import build_graph_dataset


def main() -> None:
    settings = load_settings(default_config())
    print("Starting strict graph-dataset build for 2021-01 to 2022-10.")
    print(f"Output directory: {settings.dataset_root}")
    summary = build_graph_dataset(settings)
    print("Dataset build completed:", summary)


if __name__ == "__main__":
    main()
