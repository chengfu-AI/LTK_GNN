"""Build a date-bounded graph dataset from terrain, HYSPLIT and PM10 inputs."""

from __future__ import annotations

import csv
import gzip
import os
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np

from .config import Settings, format_utc, parse_utc
from .io import read_csv, write_csv, write_json


LABEL_RE = re.compile(r"^dynamic_edges_(?P<height>\d+)m\.csv\.gz$")


def expected_sample_times(settings: Settings) -> list[datetime]:
    """Return the release times that have a complete backward transport window.

    HYSPLIT uses a 24-hour backward footprint in this study.  Consequently,
    the first UTC day (2021-01-01) is input history rather than a supervised
    graph sample; the first valid release/sample is 2021-01-02 00:00 UTC.
    Keeping this rule here makes the dataset stage agree exactly with
    ``00_run_hysplit_footprints.py`` and ``00_cdump_to_graph_labels.py``.
    """
    first = settings.start + timedelta(hours=settings.lag_hours)
    times: list[datetime] = []
    current = first
    while current <= settings.end:
        times.append(current)
        current += timedelta(hours=settings.sampling_interval_hours)
    if not times:
        raise RuntimeError("The configured study period contains no complete transport-window sample times.")
    return times


def _required_columns(rows: list[dict[str, str]], path: Path, columns: set[str]) -> None:
    if not rows or not columns.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns: {sorted(columns)}")


def read_stations(settings: Settings) -> list[dict[str, str | float]]:
    rows = read_csv(settings.station_file)
    _required_columns(rows, settings.station_file, {"site_code", "longitude", "latitude"})
    stations = [
        {"site_code": row["site_code"].strip(), "longitude": float(row["longitude"]), "latitude": float(row["latitude"])}
        for row in rows
    ]
    codes = [str(row["site_code"]) for row in stations]
    if len(codes) != len(set(codes)):
        raise ValueError("core_20_stations.csv contains duplicate site_code values.")
    if len(stations) != 20:
        raise ValueError(f"Expected exactly 20 core stations, found {len(stations)}.")
    return stations


def write_static_graph(settings: Settings, stations: list[dict[str, str | float]], output: Path) -> None:
    """Write static node/edge features once, in the source-to-target edge order."""
    node_rows, edge_rows = read_csv(settings.node_terrain_file), read_csv(settings.edge_terrain_file)
    _required_columns(node_rows, settings.node_terrain_file, {"site_code", "elevation_m"})
    _required_columns(
        edge_rows,
        settings.edge_terrain_file,
        {"source_site", "target_site", "distance_km", "bearing_deg", "delta_z_m", "terrain_barrier_m"},
    )
    elevations = {row["site_code"].strip(): float(row["elevation_m"]) for row in node_rows}
    terrain = {(row["source_site"].strip(), row["target_site"].strip()): row for row in edge_rows}
    sites = [str(row["site_code"]) for row in stations]
    absent = [site for site in sites if site not in elevations]
    if absent:
        raise ValueError(f"Node terrain data omit station(s): {absent}")

    index = {site: number for number, site in enumerate(sites)}
    edge_index, edge_static = [], []
    for source in sites:
        for target in sites:
            edge_index.append((index[source], index[target]))
            if source == target:
                edge_static.append([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                continue
            row = terrain.get((source, target))
            if row is None:
                raise ValueError(f"Missing directed terrain edge {source} -> {target}")
            bearing = np.deg2rad(float(row["bearing_deg"]))
            edge_static.append([
                float(row["distance_km"]), np.sin(bearing), np.cos(bearing),
                float(row["delta_z_m"]), float(row["terrain_barrier_m"]), 0.0,
            ])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        site_code=np.asarray(sites),
        longitude=np.asarray([row["longitude"] for row in stations], dtype=np.float32),
        latitude=np.asarray([row["latitude"] for row in stations], dtype=np.float32),
        node_static=np.asarray([[elevations[site]] for site in sites], dtype=np.float32),
        edge_index=np.asarray(edge_index, dtype=np.int64).T,
        edge_static=np.asarray(edge_static, dtype=np.float32),
        node_feature_names=np.asarray(["elevation_m"]),
        edge_feature_names=np.asarray([
            "distance_km", "bearing_sin", "bearing_cos", "delta_z_m", "terrain_barrier_m", "is_self_edge",
        ]),
    )


def _label_files(settings: Settings) -> list[tuple[int, Path]]:
    available: dict[int, Path] = {}
    for path in settings.labels_root.glob("dynamic_edges_*m.csv.gz"):
        match = LABEL_RE.fullmatch(path.name)
        if match:
            available[int(match["height"])] = path
    missing = [height for height in settings.release_heights_agl_m if height not in available]
    if missing:
        raise FileNotFoundError(f"Missing HYSPLIT label file(s) for release height(s): {missing}")
    return [(height, available[height]) for height in settings.release_heights_agl_m]


def _save_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, path)


def stage_transport_kernels(
    settings: Settings,
    site_index: dict[str, int],
    staging: Path,
    expected_times: list[datetime],
) -> int:
    """Validate and stage every transport matrix without loading all labels.

    A valid time/height block has one value for every
    ``lag × source station × target station`` combination.  Zero is a valid
    footprint weight, so merely filling a zero array is not sufficient: the
    separate ``seen`` array below detects truncated and duplicate label rows.
    """
    files = _label_files(settings)
    nodes, heights, lags = len(site_index), len(files), settings.lag_hours
    expected_by_text = {format_utc(value): value for value in expected_times}
    expected_texts = set(expected_by_text)
    staging.mkdir(parents=True, exist_ok=True)

    for height_index, (height, path) in enumerate(files):
        print(f"Validating and staging {height:05d} m labels: {path}")
        current_time: str | None = None
        layer = np.zeros((lags, nodes, nodes), dtype=np.float32)
        seen = np.zeros((lags, nodes, nodes), dtype=bool)
        staged_for_height: set[str] = set()

        def flush() -> None:
            if current_time is None:
                return
            if current_time not in expected_texts:
                raise ValueError(
                    f"Unexpected label release time {current_time} in {path}; "
                    "re-create labels with 00_cdump_to_graph_labels.py."
                )
            absent = int(seen.size - np.count_nonzero(seen))
            if absent:
                raise ValueError(
                    f"Incomplete label block in {path.name} at {current_time}: "
                    f"{absent}/{seen.size} lag/source/target entries are absent."
                )
            if current_time in staged_for_height:
                raise ValueError(f"Duplicate label time {current_time} in {path}")

            staged = staging / f"{expected_by_text[current_time]:%Y%m%d%H}.npy"
            if staged.is_file():
                full = np.load(staged)
                if full.shape != (heights, lags, nodes, nodes):
                    raise ValueError(f"Unexpected staged label shape in {staged}")
            else:
                full = np.zeros((heights, lags, nodes, nodes), dtype=np.float32)
            full[height_index] = layer
            _save_array(staged, full)
            staged_for_height.add(current_time)
            if len(staged_for_height) % 100 == 0 or len(staged_for_height) == len(expected_times):
                print(f"{height:05d} m label blocks: {len(staged_for_height)}/{len(expected_times)}")

        previous_time: datetime | None = None
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"end_time_utc", "lag_hours", "source_site", "target_site", "footprint_weight"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"Invalid label columns in {path}; need {sorted(required)}")
            for row in reader:
                end = parse_utc(row["end_time_utc"])
                if previous_time is not None and end < previous_time:
                    raise ValueError(f"{path} is not chronological near {row['end_time_utc']}")
                previous_time = end
                time_text = format_utc(end)
                if current_time is None:
                    current_time = time_text
                if time_text != current_time:
                    flush()
                    current_time = time_text
                    layer = np.zeros((lags, nodes, nodes), dtype=np.float32)
                    seen = np.zeros((lags, nodes, nodes), dtype=bool)

                lag = int(row["lag_hours"])
                source, target = row["source_site"].strip(), row["target_site"].strip()
                if not 1 <= lag <= lags:
                    raise ValueError(f"Unexpected lag {lag} in {path}")
                if source not in site_index or target not in site_index:
                    raise ValueError(f"Unknown station in {path}: {source} -> {target}")
                source_index, target_index = site_index[source], site_index[target]
                if seen[lag - 1, source_index, target_index]:
                    raise ValueError(
                        f"Duplicate label row in {path.name}: {time_text}, lag={lag}, "
                        f"{source}->{target}."
                    )
                layer[lag - 1, source_index, target_index] = float(row["footprint_weight"])
                seen[lag - 1, source_index, target_index] = True
        flush()

        missing_times = expected_texts - staged_for_height
        if missing_times:
            preview = ", ".join(sorted(missing_times)[:3])
            raise ValueError(
                f"{path.name} is missing {len(missing_times)} required label time block(s), "
                f"including {preview}."
            )
    return len(expected_times)


def _weather_available(settings: Settings, end: datetime) -> bool:
    key = end.strftime("%Y%m%d%H")
    return all((settings.weather_root / variable / f"{key}.npy").is_file() for variable in settings.weather_variables)


def _read_pm10(settings: Settings, end: datetime, site_index: dict[str, int]) -> tuple[np.ndarray, Path]:
    local_time = end + timedelta(hours=settings.pm10_utc_offset_hours)
    path = settings.pm10_root / f"{local_time:%Y%m%d%H}.csv"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    rows = read_csv(path)
    _required_columns(rows, path, {"site_code", "PM10"})
    values = np.full(len(site_index), np.nan, dtype=np.float32)
    for row in rows:
        site, value = row["site_code"].strip(), row["PM10"].strip()
        if site in site_index and value:
            values[site_index[site]] = float(value)
    if not np.isfinite(values).all():
        missing = [site for site, number in site_index.items() if not np.isfinite(values[number])]
        raise ValueError(f"Incomplete PM10 in {path.name}: {missing}")
    return values, path


def _publish_dataset(staging_root: Path, destination: Path) -> None:
    """Replace the old generated dataset only after a complete new build.

    Directory replacement is not atomic on Windows.  The short two-rename
    sequence below preserves the old dataset if publishing the new one fails;
    the old files are removed only after the new directory is in place.
    """
    previous = destination.with_name(f".{destination.name}_previous_{uuid4().hex}")
    moved_old = False
    try:
        if destination.exists():
            destination.replace(previous)
            moved_old = True
        staging_root.replace(destination)
    except Exception:
        if moved_old and previous.exists() and not destination.exists():
            previous.replace(destination)
        raise
    else:
        if previous.exists():
            shutil.rmtree(previous)


def build_graph_dataset(settings: Settings) -> dict:
    """Build and publish one complete configured-period graph dataset.

    This is deliberately strict.  A missing label, ERA5 field, or PM10 value
    aborts the build, leaves a pre-existing published dataset untouched, and
    reports the number of missing inputs.  Therefore the script can be run
    directly without an ``--overwrite`` or ``--fail-on-missing-inputs`` flag.
    """
    for required in (settings.station_file, settings.node_terrain_file, settings.edge_terrain_file):
        if not required.is_file():
            raise FileNotFoundError(f"Required input is absent: {required}")

    expected_times = expected_sample_times(settings)
    settings.output_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".graph_dataset_build_", dir=settings.output_root))
    print(f"Building {len(expected_times)} graph samples in temporary directory: {staging_root}")

    try:
        samples_dir, label_staging = staging_root / "samples", staging_root / ".label_staging"
        samples_dir.mkdir()
        stations = read_stations(settings)
        sites = [str(row["site_code"]) for row in stations]
        site_index = {site: number for number, site in enumerate(sites)}
        write_static_graph(settings, stations, staging_root / "static_graph.npz")
        staged = stage_transport_kernels(settings, site_index, label_staging, expected_times)

        manifest, skipped = [], Counter()
        for number, end in enumerate(expected_times, start=1):
            staged_file = label_staging / f"{end:%Y%m%d%H}.npy"
            if not staged_file.is_file():
                skipped["missing_staged_label"] += 1
                continue
            if not _weather_available(settings, end):
                skipped["missing_era5"] += 1
                continue
            try:
                pm10, pm10_path = _read_pm10(settings, end, site_index)
            except (FileNotFoundError, ValueError):
                skipped["missing_or_incomplete_pm10"] += 1
                continue
            kernel = np.load(staged_file)
            if kernel.shape != (
                len(settings.release_heights_agl_m), settings.lag_hours, len(sites), len(sites)
            ) or not np.isfinite(kernel).all():
                skipped["invalid_label"] += 1
                continue
            sample_name = f"{end:%Y%m%d%H}.npz"
            np.savez_compressed(samples_dir / sample_name, pm10=pm10, transport_weight=kernel)
            manifest.append({
                "end_time_utc": format_utc(end), "split": settings.split_name(end),
                "sample_file": f"samples/{sample_name}", "weather_time_key": f"{end:%Y%m%d%H}",
                "pm10_file": pm10_path.name, "num_heights": len(settings.release_heights_agl_m),
                "num_lags": settings.lag_hours,
            })
            if number % 100 == 0 or number == len(expected_times):
                print(f"Graph samples written: {number}/{len(expected_times)}")

        if skipped:
            raise RuntimeError(
                "Input gaps prevented a complete graph dataset; the existing published dataset was kept. "
                f"Details: {dict(skipped)}. Run 00_preflight.py and repair the reported inputs."
            )
        if len(manifest) != len(expected_times):
            raise RuntimeError(f"Expected {len(expected_times)} samples but built {len(manifest)}.")

        counts = Counter(row["split"] for row in manifest)
        if any(counts[split] == 0 for split in ("train", "valid", "test")):
            raise RuntimeError(f"One or more time splits are empty: {dict(counts)}")

        write_csv(staging_root / "sample_manifest.csv", manifest)
        metadata = {
            "study_period_utc": [format_utc(settings.start), format_utc(settings.end)],
            "sample_period_utc": [format_utc(expected_times[0]), format_utc(expected_times[-1])],
            "first_day_role": "2021-01-01 UTC is 24-hour HYSPLIT history; graph samples begin 2021-01-02T00:00:00Z.",
            "splits": {"train_end_utc": format_utc(settings.train_end), "valid_end_utc": format_utc(settings.valid_end)},
            "site_codes": sites, "release_heights_agl_m": list(settings.release_heights_agl_m),
            "lag_hours": list(range(1, settings.lag_hours + 1)),
            "weather_variables": list(settings.weather_variables), "pm10_utc_offset_hours": settings.pm10_utc_offset_hours,
            "transport_shape": "[height, lag, source, target]",
            "normalization": "source weights sum to one for each non-empty height/lag/target label",
        }
        write_json(staging_root / "metadata.json", metadata)
        summary = {
            "expected_graph_samples": len(expected_times),
            "staged_label_times": staged,
            "written_samples": len(manifest),
            "samples_by_split": dict(counts),
            "skipped": {},
        }
        write_json(staging_root / "build_summary.json", summary)

        # Staged label arrays are an intermediate representation; downstream
        # training uses the compressed samples and does not need this extra copy.
        shutil.rmtree(label_staging)
        _publish_dataset(staging_root, settings.dataset_root)
        print(f"Published complete graph dataset: {settings.dataset_root}")
        return summary
    except Exception:
        # A failed build must neither expose partial outputs nor alter a
        # previously published dataset.
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def load_manifest(settings: Settings) -> list[dict[str, str]]:
    rows = read_csv(settings.dataset_root / "sample_manifest.csv")
    if not rows:
        raise RuntimeError("sample_manifest.csv is empty.")
    rows.sort(key=lambda row: row["end_time_utc"])
    for row in rows:
        if not settings.contains(parse_utc(row["end_time_utc"])):
            raise ValueError(f"Dataset contains an out-of-period sample: {row['end_time_utc']}")
        if row["split"] != settings.split_name(parse_utc(row["end_time_utc"])):
            raise ValueError(f"Incorrect split recorded for {row['end_time_utc']}")
    return rows


def load_sample(settings: Settings, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(settings.dataset_root / row["sample_file"]) as sample:
        return sample["pm10"].astype(np.float32), sample["transport_weight"].astype(np.float32)


def load_weather(settings: Settings, time_key: str, variables: tuple[str, ...]) -> np.ndarray:
    arrays = [np.load(settings.weather_root / variable / f"{time_key}.npy").astype(np.float32) for variable in variables]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError(f"ERA5 fields at {time_key} must be equal-size 2D arrays; got {shapes}")
    return np.stack(arrays, axis=0)
