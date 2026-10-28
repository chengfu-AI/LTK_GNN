#!/usr/bin/env python
"""Run the full configured HYSPLIT footprint calculation efficiently.

This runner is designed for the local machine used for this project:
large RAM, a C: SSD, and an HDD-based data directory. HYSPLIT itself is a CPU
program, so the RTX 2060 is not used. For every release time the two required
ARL files are cached on C:, then independent station/height jobs run in
parallel from an ASCII-only SSD workspace. Only validated final files are
archived to the data directory.

Run the file directly for the complete configured 2021-01 to 2022-10 study
period. ``--pilot-only`` remains available only for a small diagnostic run.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

from _bootstrap import default_config
from ltkg_2021_2022.config import format_utc, load_settings, parse_utc
from ltkg_2021_2022.dataset import read_stations


# This path deliberately contains ASCII only: HYSPLIT's Windows executables
# can fail when CONTROL or meteorology paths contain Chinese text.
SSD_SCRATCH_ROOT = Path(r"C:\LTK_GNN_HYSPLIT_SCRATCH")
MIN_SCRATCH_FREE_GB = 10
MAX_WORKERS_FOR_THIS_MACHINE = 24


def arl_path(root: Path, timestamp) -> Path:
    return root / f"ERA5_{timestamp:%Y%m%d}.arl"


def hysplit_path(path: Path) -> str:
    """Return an ASCII-compatible HYSPLIT directory path with a trailing slash."""
    return path.absolute().as_posix().rstrip("/") + "/"


def completed_cdump(directory: Path) -> bool:
    """Only reuse a cdump when HYSPLIT recorded a normal completion."""
    cdump = directory / "cdump"
    stdout = directory / "stdout.txt"
    if not cdump.is_file() or not cdump.stat().st_size or not stdout.is_file():
        return False
    return "Complete Hysplit" in stdout.read_text(encoding="utf-8", errors="replace")


def require_arl_files(times, arl_root: Path, backward_hours: int) -> None:
    """Fail before starting if any required daily ARL input is absent."""
    required = {
        arl_path(arl_root, day)
        for end in times
        for day in (end.date(), (end - timedelta(hours=backward_hours)).date())
    }
    missing = sorted(path for path in required if not path.is_file())
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:10])
        remainder = "" if len(missing) <= 10 else f"\n  ... plus {len(missing) - 10} more"
        raise FileNotFoundError(
            f"Missing {len(missing)} required daily ARL file(s). Complete 00_convert_era5_to_arl.py first:\n"
            f"{preview}{remainder}"
        )


def find_ascdata(hycs: Path) -> Path | None:
    """Locate HYSPLIT's optional land-use/terrain configuration file."""
    candidates = (hycs.parent.parent / "working" / "ASCDATA.CFG", hycs.parent / "ASCDATA.CFG")
    return next((path for path in candidates if path.is_file()), None)


def effective_worker_count(requested_workers: int) -> int:
    """Reserve CPU capacity for Windows and cap I/O contention safely."""
    logical_cpus = os.cpu_count() or requested_workers
    reserve = 2 if logical_cpus >= 6 else 1
    usable_cpus = max(1, logical_cpus - reserve)
    return max(1, min(requested_workers, usable_cpus, MAX_WORKERS_FOR_THIS_MACHINE))


def ensure_scratch_root() -> Path:
    """Create the SSD workspace and reject a run that could fill C:."""
    SSD_SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(SSD_SCRATCH_ROOT).free
    minimum = MIN_SCRATCH_FREE_GB * 1024**3
    if free_bytes < minimum:
        available = free_bytes / 1024**3
        raise OSError(
            f"HYSPLIT SSD scratch drive has only {available:.1f} GB free; at least "
            f"{MIN_SCRATCH_FREE_GB} GB is required at {SSD_SCRATCH_ROOT}. Free C: space first."
        )
    return SSD_SCRATCH_ROOT


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy a file without exposing a half-written destination file."""
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def stage_meteorology(actual_arl_root: Path, cache_root: Path, end, backward_hours: int) -> Path:
    """Keep exactly the two ARL files needed by the current release time on C:."""
    sources = [
        arl_path(actual_arl_root, day)
        for day in sorted({end.date(), (end - timedelta(hours=backward_hours)).date()})
    ]
    cache_root.mkdir(parents=True, exist_ok=True)
    needed_names = {source.name for source in sources}
    for stale in cache_root.glob("ERA5_*.arl"):
        if stale.name not in needed_names:
            stale.unlink()
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        cached = cache_root / source.name
        if not cached.is_file() or cached.stat().st_size != source.stat().st_size:
            atomic_copy(source, cached)
    return cache_root


def write_control(
    job: Path,
    station,
    end,
    height: int,
    cached_arl_root: Path,
    backward_hours: int,
    ascdata: Path | None,
) -> None:
    """Write CONTROL and SETUP.CFG into one isolated SSD job directory."""
    met = [
        arl_path(cached_arl_root, date)
        for date in sorted({end.date(), (end - timedelta(hours=backward_hours)).date()})
    ]
    absent = [path for path in met if not path.is_file()]
    if absent:
        raise FileNotFoundError("Missing staged ARL: " + ", ".join(map(str, absent)))
    stop = end - timedelta(hours=backward_hours)
    lines = [
        end.strftime("%y %m %d %H"), "1",
        f"{station['latitude']:.5f} {station['longitude']:.5f} {height:.1f}",
        str(-backward_hours), "0", "10000.0", str(len(met)),
    ]
    for path in met:
        lines += [hysplit_path(cached_arl_root), path.name]
    lines += [
        "1", "TEST", "1.0", "1.0", end.strftime("%y %m %d %H 00"), "1",
        "39.250 116.300", "0.250 0.250", "12.000 14.000", "./", "cdump", "1", "10000",
        end.strftime("%y %m %d %H 00"), stop.strftime("%y %m %d %H 00"), "1 01 00", "1",
        "0.0 0.0 0.0", "0.0 0.0 0.0 0.0 0.0", "0.0 0.0 0.0", "0.0", "0.0",
    ]
    (job / "CONTROL").write_text("\n".join(lines) + "\n", encoding="ascii")
    setup = f"""&SETUP
 INITD = 0,
 KHMAX = {backward_hours},
 NUMPAR = 10000,
 MAXPAR = 50000,
 KMSL = 0,
 ICHEM = 0,
 NDUMP = 0,
 NCYCL = 0,
/
"""
    (job / "SETUP.CFG").write_text(setup, encoding="ascii")
    if ascdata is not None:
        shutil.copy2(ascdata, job / "ASCDATA.CFG")


def archive_artifacts(scratch_job: Path, final_job: Path, *, include_cdump: bool) -> None:
    """Archive diagnostics and, only after success, the final cdump to data disk."""
    final_job.mkdir(parents=True, exist_ok=True)
    for name in ("CONTROL", "SETUP.CFG", "ASCDATA.CFG", "MESSAGE", "stdout.txt", "stderr.txt"):
        source = scratch_job / name
        if source.is_file():
            atomic_copy(source, final_job / name)
    if include_cdump:
        source = scratch_job / "cdump"
        if not source.is_file() or not source.stat().st_size:
            raise FileNotFoundError(f"HYSPLIT did not produce a usable cdump: {source}")
        # cdump is copied last: it is the restart-safe completion signal.
        atomic_copy(source, final_job / "cdump")


def run_one_job(
    final_directory: Path,
    scratch_directory: Path,
    station,
    end,
    height: int,
    cached_arl_root: Path,
    backward_hours: int,
    ascdata: Path | None,
    hycs: Path,
    prepare_only: bool,
) -> dict[str, str | int]:
    """Run one independent HYSPLIT process from the SSD workspace."""
    if completed_cdump(final_directory):
        status = "skipped_existing"
    else:
        try:
            scratch_directory.mkdir(parents=True, exist_ok=True)
            # Never reuse a cdump that was left by an interrupted temporary job.
            (scratch_directory / "cdump").unlink(missing_ok=True)
            write_control(scratch_directory, station, end, height, cached_arl_root, backward_hours, ascdata)
            if prepare_only:
                archive_artifacts(scratch_directory, final_directory, include_cdump=False)
                status = "prepared"
            else:
                result = subprocess.run(
                    [str(hycs)], cwd=scratch_directory, text=True, capture_output=True, check=False,
                )
                (scratch_directory / "stdout.txt").write_text(result.stdout or "", encoding="utf-8")
                (scratch_directory / "stderr.txt").write_text(result.stderr or "", encoding="utf-8")
                succeeded = result.returncode == 0 and (scratch_directory / "cdump").is_file()
                if succeeded:
                    succeeded = bool((scratch_directory / "cdump").stat().st_size)
                archive_artifacts(scratch_directory, final_directory, include_cdump=succeeded)
                status = "done" if succeeded else f"failed (return code {result.returncode})"
        except Exception as exc:
            status = f"failed: {exc}"
    return {
        "end_time_utc": format_utc(end),
        "site_code": str(station["site_code"]),
        "height_agl_m": height,
        "status": status,
        "job_directory": str(final_directory),
    }


def make_times(settings, backward_hours: int, pilot_only: bool, pilot_time: str) -> list:
    if pilot_only:
        time = parse_utc(pilot_time)
        if not settings.contains(time):
            raise ValueError("--pilot-time must fall within the configured study period")
        return [time]
    times = []
    current = settings.start + timedelta(hours=backward_hours)
    while current <= settings.end:
        times.append(current)
        current += timedelta(hours=settings.sampling_interval_hours)
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pm10-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--hycs", type=Path, help="Optional override; default is config.executables.hycs.")
    parser.add_argument("--pilot-only", action="store_true", help="Run one site and one time only, for diagnostics.")
    parser.add_argument("--pilot-time", default="2021-01-02T00:00:00Z")
    parser.add_argument("--pilot-site")
    parser.add_argument("--backward-hours", type=int, default=24)
    parser.add_argument("--prepare-only", action="store_true", help="Only generate CONTROL/SETUP.CFG; do not launch HYSPLIT.")
    args = parser.parse_args()

    settings = load_settings(
        args.config, data_root=args.data_root, pm10_root=args.pm10_root,
        output_root=args.output_root, allow_missing_roots=True,
    )
    hycs = args.hycs or settings.hycs
    if not args.prepare_only and not hycs.is_file():
        raise FileNotFoundError(hycs)
    ascdata = None if args.prepare_only else find_ascdata(hycs)
    if ascdata is None and not args.prepare_only:
        print("Warning: ASCDATA.CFG was not found; HYSPLIT may report a terrain-data warning.")
    elif ascdata is not None:
        print(f"Using ASCDATA.CFG: {ascdata}")

    stations = read_stations(settings)
    if args.pilot_only:
        stations = [station for station in stations if args.pilot_site is None or station["site_code"] == args.pilot_site][:1]
    if not stations:
        raise ValueError("No matching pilot station.")

    times = make_times(settings, args.backward_hours, args.pilot_only, args.pilot_time)
    arl_root = settings.data_root / "raw" / "era5_hysplit" / "arl"
    final_jobs = settings.data_root / "hysplit_footprints" / "jobs"
    final_jobs.mkdir(parents=True, exist_ok=True)
    require_arl_files(times, arl_root, args.backward_hours)

    workers = effective_worker_count(settings.hysplit_workers)
    scratch_base = ensure_scratch_root()
    # A PID-specific session prevents conflicts with leftovers from an
    # interrupted run; it is removed after normal exit.
    scratch_session = scratch_base / f"session_{os.getpid()}"
    scratch_jobs = scratch_session / "jobs"
    scratch_met = scratch_session / "met"
    scratch_session.mkdir(parents=True, exist_ok=False)

    destination = settings.data_root / "hysplit_footprints" / (
        "run_manifest_pilot.csv" if args.pilot_only else "run_manifest_full.csv"
    )
    total = len(times) * len(stations) * len(settings.release_heights_agl_m)
    print(f"Scheduled HYSPLIT jobs: {total} (parallel workers: {workers})")
    print(f"SSD scratch workspace: {scratch_session}")
    print("ARL files are cached per release time on C:; completed cdump files are archived to the data directory.")

    counts, completed = Counter(), 0
    fieldnames = ["end_time_utc", "site_code", "height_agl_m", "status", "job_directory"]
    try:
        with destination.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hysplit") as executor:
                for end in times:
                    cached_arl_root = stage_meteorology(arl_root, scratch_met, end, args.backward_hours)
                    futures = []
                    for station in stations:
                        for height in settings.release_heights_agl_m:
                            job_name = f"{end:%Y%m%d%H}_{station['site_code']}_{height:05d}m"
                            futures.append(executor.submit(
                                run_one_job,
                                final_jobs / job_name,
                                scratch_jobs / job_name,
                                station,
                                end,
                                height,
                                cached_arl_root,
                                args.backward_hours,
                                ascdata,
                                hycs,
                                args.prepare_only,
                            ))
                    for future in as_completed(futures):
                        row = future.result()
                        writer.writerow(row)
                        status = str(row["status"])
                        counts["failed" if status.startswith("failed") else status] += 1
                        completed += 1
                        if args.pilot_only or status.startswith("failed"):
                            print(f"{Path(str(row['job_directory'])).name} | {status}")
                        elif completed % 100 == 0:
                            print(f"Progress: {completed}/{total} jobs")
                    handle.flush()
                    print(f"Processed {end:%Y-%m-%d %H}:00 UTC")
    finally:
        shutil.rmtree(scratch_session, ignore_errors=True)

    print(f"Wrote: {destination}")
    print("Job summary:", ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))
    if counts["failed"]:
        raise RuntimeError(
            f"{counts['failed']} HYSPLIT job(s) failed. See stdout.txt and stderr.txt in the job_directory listed in {destination}."
        )


if __name__ == "__main__":
    main()
