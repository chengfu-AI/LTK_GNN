#!/usr/bin/env python
"""Convert the downloaded monthly ERA5 GRIB stream to daily HYSPLIT ARL files.

This wrapper runs HYSPLIT's external ``ecm2arl.exe``.  It only processes the
months within the configured study period and splits each monthly DATA.ARL into 24-hour files named
``ERA5_YYYYMMDD.arl``.  Test with ``--months 202101`` before a full run.
"""

from __future__ import annotations

import argparse
import calendar
import ctypes
import re
import shutil
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

from _bootstrap import default_config
from ltkg_2021_2022.config import load_settings


PRESSURE_RE = re.compile(r"^ERA5_PL_(?P<month>\d{6})_(?P<start>\d{2})-(?P<end>\d{2})\.grib$", re.I)


def parse_layout(path: Path) -> tuple[int, int]:
    nx = ny = None; variable_records = 0
    level = re.compile(r"^Level\s+\d+:\s+\S+\s+(\d+)\b")
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        text = line.strip()
        if text.startswith("Numb X pt:"): nx = int(text.split()[-1])
        elif text.startswith("Numb Y pt:"): ny = int(text.split()[-1])
        else:
            match = level.match(text)
            if match: variable_records += int(match.group(1))
    if nx is None or ny is None or not variable_records: raise RuntimeError(f"Cannot parse ARL layout: {path}")
    return 50 + nx * ny, 1 + variable_records


def copy_exact(source, target, byte_count: int) -> None:
    remaining = byte_count
    while remaining:
        chunk = source.read(min(16 * 1024 * 1024, remaining))
        if not chunk: raise RuntimeError("DATA.ARL ended before a full daily block.")
        target.write(chunk); remaining -= len(chunk)


def available_drive_letter() -> str:
    """Find a free drive letter for a temporary ASCII-only path mapping."""
    mask = ctypes.windll.kernel32.GetLogicalDrives()
    for letter in "ZYXWVUTSRQPONMLKJIHGFED":
        if not mask & (1 << (ord(letter) - ord("A"))):
            return f"{letter}:"
    raise RuntimeError("No unused drive letter is available for the temporary SUBST mapping.")


@contextmanager
def short_path_mapping(root: Path):
    """Map a potentially Chinese/long ERA5 root to a short path for ecm2arl."""
    drive = available_drive_letter()
    try:
        subprocess.run(["subst", drive, str(root)], check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Windows subst.exe is unavailable; cannot create a short path for ecm2arl.") from exc
    try:
        yield Path(f"{drive}\\")
    finally:
        subprocess.run(["subst", drive, "/D"], check=False)


def mapped_path(actual: Path, actual_root: Path, mapped_root: Path) -> Path:
    return mapped_root / actual.resolve().relative_to(actual_root.resolve())


def months(root: Path, start_month: str, end_month: str) -> dict[str, list[Path]]:
    result = defaultdict(list)
    for path in (root / "pressure").glob("ERA5_PL_*.grib"):
        match = PRESSURE_RE.fullmatch(path.name)
        if match and start_month <= match["month"] <= end_month: result[match["month"]].append(path)
    return {month: sorted(parts, key=lambda item: int(PRESSURE_RE.fullmatch(item.name)["start"])) for month, parts in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config()); parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pm10-root", type=Path); parser.add_argument("--output-root", type=Path)
    parser.add_argument("--ecm2arl", type=Path, help="Optional override; default is config.executables.ecm2arl.")
    parser.add_argument("--months", nargs="*", help="Optional YYYYMM subset for a pilot conversion.")
    parser.add_argument("--pressure-level-count", type=int, default=16); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args(); settings = load_settings(
        args.config, data_root=args.data_root, pm10_root=args.pm10_root,
        output_root=args.output_root, allow_missing_roots=True,
    )
    ecm2arl = args.ecm2arl or settings.ecm2arl
    if not ecm2arl.is_file(): raise FileNotFoundError(ecm2arl)
    root = settings.data_root / "raw" / "era5_hysplit"
    start_month, end_month = settings.start.strftime("%Y%m"), settings.end.strftime("%Y%m")
    groups = months(root, start_month, end_month)
    selected = sorted(set(args.months or groups))
    if not selected: raise FileNotFoundError(f"No pressure files found for {start_month}–{end_month} in {root / 'pressure'}")
    outside = [month for month in selected if not start_month <= month <= end_month]
    if outside: parser.error(f"--months must fall within {start_month}–{end_month}: {', '.join(outside)}")
    arl, work = root / "arl", root / "_ecm2arl_work"; arl.mkdir(parents=True, exist_ok=True)
    with short_path_mapping(root) as short_root:
        print(f"ecm2arl temporary short path: {short_root}")
        for month in selected:
            parts = groups.get(month)
            single = root / "single" / f"ERA5_SFC_{month}.grib"
            if not parts or not single.is_file(): raise FileNotFoundError(f"Missing pressure chunks or surface GRIB for {month}")
            year, mon = int(month[:4]), int(month[4:]); days = calendar.monthrange(year, mon)[1]
            destinations = [arl / f"ERA5_{month}{day:02d}.arl" for day in range(1, days + 1)]
            if not args.overwrite and all(path.is_file() and path.stat().st_size for path in destinations):
                print(f"Skip {month}: all daily ARL files exist."); continue
            run = work / month; run.mkdir(parents=True, exist_ok=True); merged = run / f"ERA5_PL_{month}_all.grib"
            with merged.open("wb") as target:
                for part in parts:
                    with part.open("rb") as source: shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
            short_merged = mapped_path(merged, root, short_root)
            short_single = mapped_path(single, root, short_root)
            short_run = mapped_path(run, root, short_root)
            command = [str(ecm2arl), f"-i{short_merged}", f"-s{short_single}", "-g3", f"-k{args.pressure_level_count}", "-r1", f"-t{days * 24}", "-z1"]
            print("Running", month); subprocess.run(command, cwd=short_run, check=True)
            data, cfg = run / "DATA.ARL", run / "CFG_ARL"
            if not data.is_file() or not cfg.is_file(): raise RuntimeError(f"ecm2arl did not create DATA.ARL and CFG_ARL for {month}")
            record_bytes, records_per_hour = parse_layout(cfg); daily_bytes = record_bytes * records_per_hour * 24
            if data.stat().st_size != daily_bytes * days: raise RuntimeError(f"Unexpected DATA.ARL size for {month}")
            with data.open("rb") as source:
                for destination in destinations:
                    if destination.is_file() and destination.stat().st_size and not args.overwrite:
                        source.seek(daily_bytes, 1); continue
                    temporary = destination.with_suffix(".arl.tmp")
                    with temporary.open("wb") as target: copy_exact(source, target, daily_bytes)
                    temporary.replace(destination)
            shutil.copy2(cfg, arl / f"ERA5_{month}_CFG_ARL.txt")
            if not args.keep_work: shutil.rmtree(run)
            print(f"Finished {month}")


if __name__ == "__main__": main()
