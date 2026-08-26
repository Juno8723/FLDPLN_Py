"""Entry point for the gauge-based event flood map. The only stage meant to
be re-run repeatedly (e.g. every few minutes against fresh gauge readings,
for a live dashboard) -- it only reads the already-built, already-tiled
library from cli_library_reorganization.py, never touches
cli_segmentation.py's or cli_library_reorganization.py's inputs.
Deliberately kept as its own entry point (rather than a subcommand of
cli_library_reorganization.py) so a process embedding this stage on a
schedule only needs floodmap/gauge_map.py + io/raster.py, not the whole
segmentation/model/library-building stack.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import PyfldplnConfig, load_config
from .io.raster import read_raster_info
from .floodmap.gauge_map import create_gauge_flood_map


def run_stage_Floodmap(
    cfg: PyfldplnConfig,
    gauges_csv: str,
    snap_dist: float,
    out_path: Optional[str] = None,
    x_field: str = "x",
    y_field: str = "y",
) -> None:
    """Event-specific gauge-based flood map, reading gauge stages from a
    plain CSV (export AmiteGauges.xlsx's 'Available' sheet, or equivalent,
    to CSV first -- see floodmap/gauge_map.py's module docstring for why no
    Excel/live-fetch machinery is built in here). `gauges_csv` must have an
    'x'/'y' (or --x-field/--y-field) column pair in the library's own
    coordinate system, and a 'stage_elevation' column in the same vertical
    datum/units as fsp_info.csv's FilledElev (feet).
    """
    lib_dir = Path(cfg.output.resolve(cfg.paths.home_dir).library_dir)
    fdr_info = read_raster_info(cfg.paths.fdr_path, crs=cfg.paths.crs)
    gauges = pd.read_csv(gauges_csv)

    if out_path is None:
        out_path = str(lib_dir / "gauge_flood_map.tif")

    create_gauge_flood_map(
        lib_dir=lib_dir,
        gauges=gauges,
        snap_dist=snap_dist,
        raster_info=fdr_info,
        out_path=out_path,
        x_field=x_field,
        y_field=y_field,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pyfldpln-floodmap",
        description="Build an event-specific, gauge-based flood map from a "
        "library already built and tiled by pyfldpln-library-reorganization. "
        "Safe to call repeatedly (e.g. on a schedule) with fresh gauge readings.",
    )
    parser.add_argument("config", help="path to YAML config file")
    parser.add_argument("gauges_csv", help="CSV of gauge x,y,stage_elevation (see run_stage_Floodmap() docstring)")
    parser.add_argument("--snap-dist", type=float, required=True,
                         help="max gauge-to-FSP snap distance, in the library's own CRS units")
    parser.add_argument("--out", help="output GeoTIFF path (default: <library_dir>/gauge_flood_map.tif)")
    parser.add_argument("--x-field", default="x")
    parser.add_argument("--y-field", default="y")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_stage_Floodmap(
        cfg, args.gauges_csv, args.snap_dist,
        out_path=args.out, x_field=args.x_field, y_field=args.y_field,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
