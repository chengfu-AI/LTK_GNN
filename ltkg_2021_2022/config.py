"""Configuration loading and hard time-boundary checks."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def parse_utc(value: str) -> datetime:
    """Parse the project's unambiguous UTC representation."""
    try:
        return datetime.strptime(value, TIME_FORMAT)
    except ValueError as exc:
        raise ValueError(f"Expected UTC time in {TIME_FORMAT!r} form, got {value!r}.") from exc


def format_utc(value: datetime) -> str:
    return value.strftime(TIME_FORMAT)


def _path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _automatic_path(label: str, base: Path, *, allow_missing: bool = False) -> Path:
    """Resolve ordinary local locations so the shipped scripts need no flags."""
    environment_name = f"LTKG_{label.upper()}"
    if os.environ.get(environment_name):
        return Path(os.environ[environment_name]).expanduser().resolve()
    candidates = {
        "data_root": (
            base / "data",
            base / "input_data",
            Path(r"D:\paper2_数据下载\data"),
            Path(r"D:\paper2_数据下载"),
            Path(r"D:\LTK-GNN\data_procress\Station_jjj\data"),
        ),
        "pm10_root": (
            base / "PM10",
            base / "PM10_Kinsert",
            base / "input_pm10",
            base / "data" / "PM10",
            base / "data" / "PM10_Kinsert",
            Path(r"D:\paper2_数据下载\PM10"),
            Path(r"D:\LTK-GNN\data_procress\Station_jjj\PM10_Kinsert_2024"),
        ),
    }.get(label, ())
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    suggested = base / ("input_data" if label == "data_root" else "input_pm10")
    if allow_missing:
        return suggested.resolve()
    raise ValueError(
        f"Cannot automatically locate {label}. Put the data in {suggested}, "
        f"set {environment_name}, or edit config/study_2021_2022.json once."
    )


def _configured_path(
    value: str | None, label: str, base: Path, override: str | Path | None, *, allow_missing: bool = False
) -> Path:
    chosen = override if override is not None else value
    if chosen == "AUTO":
        return _automatic_path(label, base, allow_missing=allow_missing)
    if chosen is None or str(chosen).startswith("REPLACE_WITH_"):
        raise ValueError(f"{label} is not configured. Edit config/study_2021_2022.json once.")
    return _path(chosen, base)


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_root: Path
    pm10_root: Path
    output_root: Path
    start: datetime
    end: datetime
    train_end: datetime
    valid_end: datetime
    sampling_interval_hours: int
    pm10_utc_offset_hours: int
    lag_hours: int
    release_heights_agl_m: tuple[int, ...]
    weather_variables: tuple[str, ...]
    teacher_variants: dict[str, dict[str, Any]]
    ecm2arl: Path
    hycs: Path
    con2asc: Path
    hysplit_workers: int

    @property
    def labels_root(self) -> Path:
        return self.data_root / "hysplit_footprints" / "graph_labels"

    @property
    def weather_root(self) -> Path:
        return self.data_root / "era5_by_variable"

    @property
    def station_file(self) -> Path:
        return self.data_root / "core_station_selection" / "core_20_stations.csv"

    @property
    def node_terrain_file(self) -> Path:
        return self.data_root / "terrain_features" / "node_terrain_features.csv"

    @property
    def edge_terrain_file(self) -> Path:
        return self.data_root / "terrain_features" / "edge_terrain_features.csv"

    @property
    def dataset_root(self) -> Path:
        return self.output_root / "graph_dataset"

    @property
    def teachers_root(self) -> Path:
        return self.output_root / "teacher_runs"

    @property
    def pm10_runs_root(self) -> Path:
        return self.output_root / "pm10_runs"

    def contains(self, value: datetime) -> bool:
        return self.start <= value <= self.end

    def split_name(self, value: datetime) -> str:
        if not self.contains(value):
            raise ValueError(f"Timestamp outside the declared 2021-01 to 2022-10 study period: {format_utc(value)}")
        if value <= self.train_end:
            return "train"
        if value <= self.valid_end:
            return "valid"
        return "test"

    def teacher_run_dir(self, variant: str, seed: int) -> Path:
        return self.teachers_root / variant / f"seed_{seed}"

    def pm10_run_dir(self, source: str, seed: int) -> Path:
        return self.pm10_runs_root / source / f"seed_{seed}"


def load_settings(
    config_path: str | Path,
    *,
    data_root: str | Path | None = None,
    pm10_root: str | Path | None = None,
    output_root: str | Path | None = None,
    allow_missing_roots: bool = False,
) -> Settings:
    """Load settings and enforce the available 2021-01 through 2022-10 data boundary."""
    config_file = Path(config_path).resolve()
    payload = json.loads(config_file.read_text(encoding="utf-8"))
    project_root = config_file.parent.parent
    configured_data = _configured_path(
        payload.get("data_root"), "data_root", project_root, data_root, allow_missing=allow_missing_roots
    )
    configured_pm10 = _configured_path(
        payload.get("pm10_root"), "pm10_root", project_root, pm10_root, allow_missing=allow_missing_roots
    )
    configured_output = _path(output_root or payload.get("output_root", "outputs"), project_root)

    period = payload["period"]
    splits = payload["splits"]
    start, end = parse_utc(period["start_utc"]), parse_utc(period["end_utc"])
    train_end, valid_end = parse_utc(splits["train_end_utc"]), parse_utc(splits["valid_end_utc"])
    expected = (datetime(2021, 1, 1), datetime(2022, 10, 31, 23))
    if (start, end) != expected:
        raise ValueError("This project is intentionally fixed to 2021-01-01 through 2022-10-31 UTC.")
    if not start <= train_end < valid_end < end:
        raise ValueError("Require start <= train_end < valid_end < end.")

    variables = tuple(payload["weather_variables"])
    variants = payload["teacher_variants"]
    executables = payload.get("executables", {})
    execution = payload.get("execution", {})
    requested_workers = int(execution.get("hysplit_workers", 1))
    if requested_workers < 1:
        raise ValueError("execution.hysplit_workers must be at least 1.")
    hysplit_workers = min(requested_workers, os.cpu_count() or requested_workers)
    if not variables or not variants:
        raise ValueError("weather_variables and teacher_variants must not be empty.")
    if any(variable not in variables for item in variants.values() for variable in item["weather"]):
        raise ValueError("A teacher variant requests a weather variable absent from weather_variables.")

    return Settings(
        project_root=project_root,
        data_root=configured_data,
        pm10_root=configured_pm10,
        output_root=configured_output,
        start=start,
        end=end,
        train_end=train_end,
        valid_end=valid_end,
        sampling_interval_hours=int(payload["sampling_interval_hours"]),
        pm10_utc_offset_hours=int(payload["pm10_utc_offset_hours"]),
        lag_hours=int(payload["lag_hours"]),
        release_heights_agl_m=tuple(int(value) for value in payload["release_heights_agl_m"]),
        weather_variables=variables,
        teacher_variants=variants,
        ecm2arl=_path(executables.get("ecm2arl", r"C:\hysplit\exec\ecm2arl.exe"), project_root),
        hycs=_path(executables.get("hycs", r"C:\hysplit\exec\hycs_std.exe"), project_root),
        con2asc=_path(executables.get("con2asc", r"C:\hysplit\exec\con2asc.exe"), project_root),
        hysplit_workers=hysplit_workers,
    )
