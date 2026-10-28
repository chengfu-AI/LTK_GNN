#!/usr/bin/env python
"""Extract model-ready hourly ERA5 fields from the 2021–2024 GRIB download layout.

Only 2021–2022 messages are written.  It accepts the `single/ERA5_SFC_YYYYMM.grib`
and `pressure/ERA5_PL_YYYYMM_DD-DD.grib` layout produced by the supplied downloader.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import default_config
from ltkg_2021_2022.config import load_settings

SINGLE_NAMES = {"10u": "u10", "10v": "v10", "2t": "t2m", "2d": "d2m", "sp": "sp", "blh": "blh", "tp": "tp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config()); parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pm10-root", type=Path); parser.add_argument("--output-root", type=Path); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(); settings = load_settings(
        args.config, data_root=args.data_root, pm10_root=args.pm10_root,
        output_root=args.output_root, allow_missing_roots=True,
    )
    try: from eccodes import codes_get, codes_get_array, codes_get_values, codes_grib_new_from_file, codes_release
    except ImportError as exc: raise SystemExit("Install eccodes first: pip install eccodes") from exc
    # Keep imported functions in a local namespace so the loop works with both eccodes Python APIs.
    source = settings.data_root / "raw" / "era5_hysplit"; inputs = sorted((source / "single").glob("ERA5_SFC_*.grib")) + sorted((source / "pressure").glob("ERA5_PL_*.grib"))
    if not inputs: raise FileNotFoundError(f"No GRIB inputs found under {source}")
    written, coordinates = 0, False
    for path in inputs:
        with path.open("rb") as handle:
            while True:
                message = codes_grib_new_from_file(handle)
                if message is None: break
                try:
                    short = str(codes_get(message, "shortName")); level = int(codes_get(message, "level"))
                    date, hour = int(codes_get(message, "validityDate")), int(codes_get(message, "validityTime"))
                    from datetime import datetime
                    stamp = f"{date:08d}{hour:04d}"[:10]
                    timestamp = datetime.strptime(stamp, "%Y%m%d%H")
                    name = SINGLE_NAMES.get(short)
                    if short in {"u", "v"} and level in {850, 925}: name = f"{short}_{level}"
                    if name is None or not settings.contains(timestamp): continue
                    ni, nj = int(codes_get(message, "Ni")), int(codes_get(message, "Nj")); values = np.asarray(codes_get_values(message), dtype=np.float32).reshape(nj, ni)
                    target = settings.weather_root / name / f"{stamp}.npy"; target.parent.mkdir(parents=True, exist_ok=True)
                    if args.overwrite or not target.is_file(): np.save(target, values); written += 1
                    if not coordinates:
                        lat = np.asarray(codes_get_array(message, "latitudes"), dtype=np.float32).reshape(nj, ni); lon = np.asarray(codes_get_array(message, "longitudes"), dtype=np.float32).reshape(nj, ni)
                        (settings.weather_root / "latitude").mkdir(parents=True, exist_ok=True); (settings.weather_root / "longitude").mkdir(parents=True, exist_ok=True)
                        np.save(settings.weather_root / "latitude" / "values.npy", lat[:, 0]); np.save(settings.weather_root / "longitude" / "values.npy", lon[0, :]); coordinates = True
                finally: codes_release(message)
        print(f"Read: {path.name}")
    print(f"Wrote {written} hourly ERA5 fields under {settings.weather_root}")


if __name__ == "__main__": main()
