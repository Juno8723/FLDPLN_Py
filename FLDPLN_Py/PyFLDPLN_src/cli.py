"""Command-line entry points for the PyFLDPLN_src pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

from .config import PyfldplnConfig, load_config
from .io.raster import read_raster, read_raster_info
from .io.vector import read_shapefile_fields
from .mapping.create_map import create_fldpln_map
from .model.runner import run_library
from .segmentation.auto import build_auto_segments
from .segmentation.network_core import write_segment_raster
from .segmentation.seedpoints import build_seedpoint_segments
from .segmentation.units import miles_to_pixels, sqmiles_to_pixels


def build_segments(cfg: PyfldplnConfig) -> None:
    """Build the stream segment library (Steps 0-4), replacing the
    create_segdb_facthr_maxlen[.m|_seedpts.m] call in fldpln_wrapper_v8.m.
    """
    fdr, fdr_info = read_raster(cfg.paths.fdr_path)
    fac, fac_info = read_raster(cfg.paths.fac_path)
    fac = fac.astype(np.float64)

    pxsz = fdr_info.pixel_size_x
    strthr = sqmiles_to_pixels(cfg.segmentation.stream_threshold_sqmi, pxsz)
    facthr = sqmiles_to_pixels(cfg.segmentation.break_threshold_sqmi, pxsz)
    seglen = miles_to_pixels(cfg.segmentation.max_segment_length_mi, pxsz)

    if cfg.segmentation.method == "auto":
        seg_table = build_auto_segments(fac, fdr, fac_info.nodata, strthr, facthr, seglen)
    else:
        shp_path = cfg.paths.seed_points_shapefile_path
        points_xy = read_shapefile_fields(shp_path, ["X", "Y"])
        seg_table = build_seedpoint_segments(points_xy, fdr, fac, fdr_info, facthr, seglen)

    print(
        f"{len(seg_table)} stream segments identified "
        f"(strthr={strthr}px, facthr={facthr}px, seglen={seglen}px)"
    )

    seg_info_path = cfg.paths.seg_info_path
    seg_info_path.parent.mkdir(parents=True, exist_ok=True)
    # seg_table.attrs carries transient numpy arrays (pixel_segment_ids,
    # reach_pixels) used only by write_segment_raster below; drop them on a
    # copy before persisting, since they aren't JSON/parquet-serializable.
    seg_info_only = seg_table.copy()
    seg_info_only.attrs = {}  # rebind (not mutate) so seg_table.attrs is untouched
    seg_info_only.to_parquet(seg_info_path)
    print(f"seg_info written to {seg_info_path}")

    segment_raster_path = cfg.paths.segment_raster_path
    segment_raster_path.parent.mkdir(parents=True, exist_ok=True)
    write_segment_raster(
        str(segment_raster_path), seg_table, fdr_info.rows, fdr_info.cols, fdr_info
    )
    print(f"segment raster written to {segment_raster_path}")


def run_model(cfg: PyfldplnConfig, seg_ids: Optional[list[int]] = None) -> None:
    """Run the per-segment flood-fill simulation, replacing the parfeval loop
    in fldpln_wrapper_v8.m that calls fldpln_model_v8 for each segment.

    `run.n_workers` (null = os.cpu_count()) controls segment-level
    parallelism; see model/runner.py's docstring for how it's implemented.
    """
    fil, fil_info = read_raster(cfg.paths.fil_path)
    fdr, _ = read_raster(cfg.paths.fdr_path)
    dem, _ = read_raster(cfg.paths.dem_path)
    fil = fil.astype(np.float64).ravel()
    fdr = fdr.ravel()
    dem = dem.astype(np.float64).ravel()
    bg = fil_info.nodata if fil_info.nodata is not None else 0.0

    seg_info = pd.read_parquet(cfg.paths.seg_info_path)

    output = cfg.output.resolve(cfg.paths.home_dir)
    segdr = output.segment_files_dir

    run_library(
        seg_info=seg_info,
        fil=fil,
        fdr=fdr,
        dem=dem,
        fldmn=cfg.flood.fldmn,
        fldmx=cfg.flood.fldmx,
        dh=cfg.flood.dh,
        mxht0=cfg.flood.max_height_cap,
        ssflg=cfg.flood.steady_state,
        bg=bg,
        n_rows=fil_info.rows,
        n_cols=fil_info.cols,
        segdr=segdr,
        seg_ids=seg_ids,
        n_workers=cfg.run.n_workers or os.cpu_count() or 1,
    )


def create_map(cfg: PyfldplnConfig, seg_ids: Optional[list[int]] = None) -> None:
    """Merge per-segment flood-fill results into one DTF map, replacing
    create_fldpln_map.m / custom_floodplain_map.m."""
    fdr_info = read_raster_info(cfg.paths.fdr_path)
    seg_info = pd.read_parquet(cfg.paths.seg_info_path)
    output = cfg.output.resolve(cfg.paths.home_dir)

    create_fldpln_map(
        seg_info=seg_info,
        fdr_info=fdr_info,
        fldmn=cfg.flood.fldmn,
        fldmx=cfg.flood.fldmx,
        dh=cfg.flood.dh,
        segdr=output.segment_files_dir,
        out_path=output.map_output,
        seg_ids=seg_ids,
    )
    print(f"DTF map written to {output.map_output}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="PyFLDPLN_src")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-segments", help="build the stream segment library")
    p_build.add_argument("config", help="path to YAML config file")

    p_run = sub.add_parser("run-model", help="run the per-segment flood-fill simulation")
    p_run.add_argument("config", help="path to YAML config file")
    p_run.add_argument(
        "--segments", help="comma-separated 1-based segment IDs to run (default: all)"
    )

    p_map = sub.add_parser("create-map", help="merge segment results into one DTF map")
    p_map.add_argument("config", help="path to YAML config file")
    p_map.add_argument(
        "--segments", help="comma-separated 1-based segment IDs to merge (default: all)"
    )

    args = parser.parse_args(argv)

    if args.command == "build-segments":
        cfg = load_config(args.config)
        build_segments(cfg)
    elif args.command == "run-model":
        cfg = load_config(args.config)
        seg_ids = (
            [int(s) for s in args.segments.split(",")] if args.segments else None
        )
        run_model(cfg, seg_ids=seg_ids)
    elif args.command == "create-map":
        cfg = load_config(args.config)
        seg_ids = (
            [int(s) for s in args.segments.split(",")] if args.segments else None
        )
        create_map(cfg, seg_ids=seg_ids)

    return 0


if __name__ == "__main__":
    sys.exit(main())
