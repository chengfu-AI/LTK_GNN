#!/usr/bin/env python
"""Convert completed HYSPLIT cdump footprints into lagged graph labels.

The full study has 320,640 cdump files.  This runner uses the same strategy as
``00_run_hysplit_footprints.py``: each con2asc conversion runs in an isolated,
ASCII-only directory on the C: SSD, while the authoritative cdump files remain
in the data directory.  Conversions run concurrently, but output rows are
written in deterministic time/station order to bounded-memory gzip files.

Run this file directly.  Existing label files are replaced only after each new
height file has been completed successfully, so earlier Pilot label files no
longer require an ``--overwrite`` argument.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from _bootstrap import default_config
from ltkg_2021_2022.config import format_utc, load_settings
from ltkg_2021_2022.dataset import read_stations
from ltkg_2021_2022.io import write_csv


JOB_RE = re.compile(r"^(?P<time>\d{10})_(?P<site>.+)_(?P<height>\d+)m$")

# Match the high-throughput HYSPLIT runner.  The requested count still comes
# from config/study_2021_2022.json and is capped by usable logical CPUs.
SSD_SCRATCH_ROOT = Path(r"C:\LTK_GNN_HYSPLIT_SCRATCH")
MIN_SCRATCH_FREE_GB = 10
MAX_CON2ASC_WORKERS = 24
GZIP_COMPRESSLEVEL = 6


@dataclass(frozen=True)
class Grid:
    center_lat: float
    center_lon: float
    dlat: float
    dlon: float
    span_lat: float
    span_lon: float

    @property
    def lower_lat(self) -> float:
        return self.center_lat - self.span_lat / 2

    @property
    def lower_lon(self) -> float:
        return self.center_lon - self.span_lon / 2


@dataclass(frozen=True)
class ConversionResult:
    rows: list[tuple[str, int, int, str, str, str, str]]
    manifest: tuple[str, int, str, int, str]


def parse_control(path: Path) -> tuple[Grid, int]:
    """Read only the concentration-grid geometry and backward duration."""
    lines = [line.strip() for line in path.read_text(encoding="ascii", errors="replace").splitlines()]
    locations = int(lines[1])
    index = 2 + locations
    hours = abs(int(float(lines[index])))
    index += 3
    meteorology_files = int(lines[index])
    index += 1 + 2 * meteorology_files
    pollutants = int(lines[index])
    index += 1 + 3 * pollutants
    index += 1
    if int(lines[index]) != 1:
        raise ValueError(f"Only one concentration grid is supported: {path}")
    index += 1
    center_lat, center_lon = map(float, lines[index].split())
    dlat, dlon = map(float, lines[index + 1].split())
    span_lat, span_lon = map(float, lines[index + 2].split())
    return Grid(center_lat, center_lon, dlat, dlon, span_lat, span_lon), hours


def snap(latitude: float, longitude: float, grid: Grid) -> tuple[float, float]:
    """Return the concentration-grid cell that contains a station."""
    lat = grid.lower_lat + round((latitude - grid.lower_lat) / grid.dlat) * grid.dlat
    lon = grid.lower_lon + round((longitude - grid.lower_lon) / grid.dlon) * grid.dlon
    if not (
        grid.lower_lat <= lat <= grid.lower_lat + grid.span_lat
        and grid.lower_lon <= lon <= grid.lower_lon + grid.span_lon
    ):
        raise ValueError("Station outside HYSPLIT concentration grid")
    return round(lat, 4), round(lon, 4)


def completed_cdump(directory: Path) -> bool:
    """Use only cdump files that the HYSPLIT runner recorded as complete."""
    cdump = directory / "cdump"
    stdout = directory / "stdout.txt"
    if not cdump.is_file() or not cdump.stat().st_size or not stdout.is_file():
        return False
    return "Complete Hysplit" in stdout.read_text(encoding="utf-8", errors="replace")


def effective_worker_count(requested_workers: int) -> int:
    """Mirror the safe CPU reservation used by the HYSPLIT footprint runner."""
    logical_cpus = os.cpu_count() or requested_workers
    reserve = 2 if logical_cpus >= 6 else 1
    usable_cpus = max(1, logical_cpus - reserve)
    return max(1, min(requested_workers, usable_cpus, MAX_CON2ASC_WORKERS))


def ensure_scratch_root() -> Path:
    """Create the SSD workspace and ensure C: has enough safety headroom."""
    SSD_SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(SSD_SCRATCH_ROOT).free
    required_bytes = MIN_SCRATCH_FREE_GB * 1024**3
    if free_bytes < required_bytes:
        available = free_bytes / 1024**3
        raise OSError(
            f"con2asc SSD scratch drive has only {available:.1f} GB free; at least "
            f"{MIN_SCRATCH_FREE_GB} GB is required at {SSD_SCRATCH_ROOT}."
        )
    return SSD_SCRATCH_ROOT


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy a file without exposing a half-written target."""
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def collect_jobs(settings, sites: list[str], backward_hours: int) -> dict[int, list[tuple[datetime, str, Path]]]:
    """Find and strictly validate the full HYSPLIT result set before conversion."""
    jobs_root = settings.data_root / "hysplit_footprints" / "jobs"
    if not jobs_root.is_dir():
        raise FileNotFoundError(f"HYSPLIT job directory does not exist: {jobs_root}")
    first_end = settings.start + timedelta(hours=backward_hours)
    groups: dict[int, list[tuple[datetime, str, Path]]] = defaultdict(list)
    for directory in jobs_root.iterdir():
        if not directory.is_dir() or not completed_cdump(directory):
            continue
        match = JOB_RE.fullmatch(directory.name)
        if match is None:
            continue
        end = datetime.strptime(match["time"], "%Y%m%d%H")
        height = int(match["height"])
        site = match["site"]
        if (
            first_end <= end <= settings.end
            and site in sites
            and height in settings.release_heights_agl_m
        ):
            groups[height].append((end, site, directory))

    times = []
    current = first_end
    while current <= settings.end:
        times.append(current)
        current += timedelta(hours=settings.sampling_interval_hours)
    expected_per_height = len(times) * len(sites)
    for height in settings.release_heights_agl_m:
        unique = {(end, site) for end, site, _ in groups[height]}
        if len(groups[height]) != expected_per_height or len(unique) != expected_per_height:
            raise RuntimeError(
                f"Height {height} m has {len(unique)}/{expected_per_height} completed cdump jobs. "
                "Re-run 00_run_hysplit_footprints.py until its summary has failed=0 before creating labels."
            )
        groups[height].sort(key=lambda item: (item[0], item[1], item[2].name))
    return groups


def convert_footprint(
    source_job: Path,
    scratch_job: Path,
    end: datetime,
    hours: int,
    con2asc: Path,
    station_cells: set[tuple[float, float]],
    keep_ascii: bool,
) -> dict[tuple[int, float, float], float]:
    """Run con2asc on C: and retain only the 20 station cells that are needed."""
    scratch_job.mkdir(parents=True, exist_ok=True)
    try:
        atomic_copy(source_job / "cdump", scratch_job / "cdump")
        output = scratch_job / "footprint_ascii"
        output_txt = scratch_job / "footprint_ascii.txt"
        output.unlink(missing_ok=True)
        output_txt.unlink(missing_ok=True)
        result = subprocess.run(
            [str(con2asc), "-icdump", "-ofootprint_ascii", "-s", "-d", "-x"],
            cwd=scratch_job,
            text=True,
            capture_output=True,
            check=False,
        )
        ascii_file = output_txt if output_txt.is_file() else output
        if result.returncode or not ascii_file.is_file():
            message = (result.stdout + result.stderr).strip()
            raise RuntimeError(f"con2asc failed: {source_job.name}: {message}")

        values: dict[tuple[int, float, float], float] = {}
        with ascii_file.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) < 7:
                    continue
                year, month, day, hour = map(int, row[:4])
                cell = (round(float(row[4]), 4), round(float(row[5]), 4))
                if cell not in station_cells:
                    continue
                time = datetime(year + 2000 if year < 100 else year, month, day, hour)
                lag = round((end - time).total_seconds() / 3600)
                if 1 <= lag <= hours:
                    values[(lag, *cell)] = float(row[-1])
        if keep_ascii:
            atomic_copy(ascii_file, source_job / "footprint_ascii.txt")
        return values
    finally:
        # The cdump and ASCII output are transient.  Deleting per job keeps C:
        # usage bounded even for the 320,640-job full study.
        shutil.rmtree(scratch_job, ignore_errors=True)


def build_result(
    end: datetime,
    target: str,
    source_job: Path,
    height: int,
    hours: int,
    sites: list[str],
    station_points: dict[str, tuple[float, float]],
    con2asc: Path,
    scratch_jobs: Path,
    keep_ascii: bool,
) -> ConversionResult:
    """Convert one cdump and build its 20 source x 24 lag graph-label rows."""
    values = convert_footprint(
        source_job,
        scratch_jobs / source_job.name,
        end,
        hours,
        con2asc,
        set(station_points.values()),
        keep_ascii,
    )
    end_text = format_utc(end)
    rows: list[tuple[str, int, int, str, str, str, str]] = []
    nonzero = 0
    for lag in range(1, hours + 1):
        raw = {source: max(0.0, values.get((lag, *station_points[source]), 0.0)) for source in sites}
        total = sum(raw.values())
        for source, value in raw.items():
            nonzero += value > 0.0
            rows.append((
                end_text,
                height,
                lag,
                source,
                target,
                f"{value:.8e}",
                f"{value / total if total else 0.0:.8e}",
            ))
    return ConversionResult(rows, (end_text, height, target, nonzero, "done"))


def ordered_results(tasks, workers: int, **conversion_kwargs):
    """Yield parallel conversions in input order with at most 2x workers buffered."""
    task_iterator = iter(enumerate(tasks, start=1))
    pending: dict[Future[ConversionResult], int] = {}
    ready: dict[int, ConversionResult] = {}
    next_to_write = 1
    exhausted = False
    window = max(workers * 2, workers)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="con2asc") as executor:
        while pending or not exhausted:
            while not exhausted and len(pending) < window:
                try:
                    number, (end, target, source_job) = next(task_iterator)
                except StopIteration:
                    exhausted = True
                    break
                future = executor.submit(build_result, end, target, source_job, **conversion_kwargs)
                pending[future] = number
            if not pending:
                continue
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                ready[pending.pop(future)] = future.result()
            while next_to_write in ready:
                yield next_to_write, ready.pop(next_to_write)
                next_to_write += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pm10-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--con2asc", type=Path, help="Optional override; default is config.executables.con2asc.")
    parser.add_argument("--keep-ascii", action="store_true", help="Copy diagnostic ASCII output back to every job directory.")
    parser.add_argument("--backward-hours", type=int, default=24)
    args = parser.parse_args()

    settings = load_settings(
        args.config,
        data_root=args.data_root,
        pm10_root=args.pm10_root,
        output_root=args.output_root,
        allow_missing_roots=True,
    )
    con2asc = args.con2asc or settings.con2asc
    if not con2asc.is_file():
        raise FileNotFoundError(con2asc)

    stations = read_stations(settings)
    sites = [str(row["site_code"]) for row in stations]
    groups = collect_jobs(settings, sites, args.backward_hours)
    reference, hours = parse_control(groups[settings.release_heights_agl_m[0]][0][2] / "CONTROL")
    station_points = {
        site: snap(float(row["latitude"]), float(row["longitude"]), reference)
        for site, row in zip(sites, stations)
    }

    settings.labels_root.mkdir(parents=True, exist_ok=True)
    write_csv(
        settings.labels_root / "station_grid_points.csv",
        [
            {"site_code": site, "grid_latitude": point[0], "grid_longitude": point[1]}
            for site, point in station_points.items()
        ],
    )

    workers = effective_worker_count(settings.hysplit_workers)
    scratch_base = ensure_scratch_root()
    scratch_session = scratch_base / f"con2asc_session_{os.getpid()}"
    scratch_jobs = scratch_session / "jobs"
    scratch_session.mkdir(parents=True, exist_ok=False)
    station_cells = len(set(station_points.values()))
    total = sum(len(groups[height]) for height in settings.release_heights_agl_m)
    print(f"Scheduled con2asc conversions: {total} (parallel workers: {workers})")
    print(f"SSD scratch workspace: {scratch_session}")
    print(f"Reading {station_cells} unique station grid cell(s) per cdump; existing label files will be replaced atomically.")

    labels_header = [
        "end_time_utc", "height_agl_m", "lag_hours", "source_site", "target_site", "footprint_raw", "footprint_weight",
    ]
    manifest_header = ["end_time_utc", "height_agl_m", "target_site", "nonzero_station_edges", "status"]
    manifest_destination = settings.labels_root / "graph_build_manifest.csv"
    manifest_temporary = manifest_destination.with_suffix(manifest_destination.suffix + ".tmp")
    completed = 0
    try:
        with manifest_temporary.open("w", encoding="utf-8-sig", newline="") as manifest_handle:
            manifest_writer = csv.writer(manifest_handle)
            manifest_writer.writerow(manifest_header)
            for height in settings.release_heights_agl_m:
                destination = settings.labels_root / f"dynamic_edges_{height:05d}m.csv.gz"
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                kwargs = {
                    "height": height,
                    "hours": hours,
                    "sites": sites,
                    "station_points": station_points,
                    "con2asc": con2asc,
                    "scratch_jobs": scratch_jobs,
                    "keep_ascii": args.keep_ascii,
                }
                with gzip.open(temporary, "wt", encoding="utf-8", newline="", compresslevel=GZIP_COMPRESSLEVEL) as handle:
                    writer = csv.writer(handle)
                    writer.writerow(labels_header)
                    for number, result in ordered_results(groups[height], workers, **kwargs):
                        writer.writerows(result.rows)
                        manifest_writer.writerow(result.manifest)
                        completed += 1
                        if number % 100 == 0:
                            print(f"{height:05d} m: {number}/{len(groups[height])}; total {completed}/{total}")
                os.replace(temporary, destination)
                manifest_handle.flush()
                print(f"Wrote: {destination}")
        os.replace(manifest_temporary, manifest_destination)
    finally:
        shutil.rmtree(scratch_session, ignore_errors=True)

    print(f"Wrote: {manifest_destination}")
    print("Graph-label conversion completed successfully.")


if __name__ == "__main__":
    main()
