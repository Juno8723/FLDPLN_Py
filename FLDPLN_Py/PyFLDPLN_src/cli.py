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
from .segmentation.network_core import (
    write_segment_preview_png,
    write_segment_raster,
    write_segment_shapefile,
)
from .segmentation.seedpoints import build_seedpoint_segments
from .segmentation.units import miles_to_pixels, sqmiles_to_pixels


def build_segments(cfg: PyfldplnConfig) -> None:
    """Build the stream segment library (Steps 0-4), replacing the
    create_segdb_facthr_maxlen[.m|_seedpts.m] call in fldpln_wrapper_v8.m.
    """
    fdr, fdr_info = read_raster(cfg.paths.fdr_path, crs=cfg.paths.crs)
    fac, fac_info = read_raster(cfg.paths.fac_path, crs=cfg.paths.crs)
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

    # Vector + preview outputs -- replace the original's manual GIS
    # round-trip (arcpy RasterToPolyline + Dissolve, done by hand to review
    # the auto-generated network before committing to flood-fill) with two
    # auto-generated review aids: an actual shapefile (for opening in
    # ArcGIS/QGIS with full pan/zoom/attribute-click) and a static PNG (for
    # an instant glance with no GIS at all).
    vector_path = segment_raster_path.parent.parent / "vector" / "segments.shp"
    write_segment_shapefile(str(vector_path), seg_table, fdr_info.cols, fdr_info)
    print(f"segment shapefile written to {vector_path}")

    preview_path = segment_raster_path.parent / "segments_preview.png"
    write_segment_preview_png(str(preview_path), seg_table, fdr_info.cols, fdr_info)
    print(f"segment preview image written to {preview_path}")

    _print_shortest_segments(seg_info_only, segment_raster_path)


def _print_shortest_segments(seg_table, segment_raster_path, n: int = 5) -> None:
    """Surface the n shortest segments as a quick review starting point --
    replaces the original's manual GIS round-trip (raster-to-polyline,
    dissolve, eyeball the result for a stray disconnected sliver) with a
    printed shortlist. This isn't anomaly *detection* (a length threshold
    tuned to flag genuine outliers doesn't hold up across differently-sized
    AOIs -- e.g. plain IQR fencing never fires on real Wildcat/Amite length
    distributions), just a convenience pointer to where to look first; the
    actual judgment call is still the user's, same as it always was.
    """
    n = min(n, len(seg_table))
    shortest = seg_table["length"].nsmallest(n)
    print(f"shortest {n} segment(s) (for a quick sanity check against {segment_raster_path}):")
    for row_id, length in shortest.items():
        print(f"  seg {row_id + 1}: length={length}px")
    print("  if any look unexpectedly isolated, use run-model --exclude-segments to skip them")


def _resolve_seg_ids(
    n_segments: int,
    seg_ids: Optional[list[int]],
    exclude_seg_ids: Optional[list[int]],
) -> Optional[list[int]]:
    """Apply an optional exclude list on top of an optional include list.

    Replaces the original's shapefile-based `seg_list` mechanism
    (fldpln_wrapper_v8.m reading GRID_CODE from a hand-edited, dissolved
    'str_segid...mod4lib.shp' to drop segments a human decided not to
    process -- e.g. a stray, disconnected tiny segment) with a plain CLI
    flag: rather than round-tripping through GIS (raster-to-polyline,
    dissolve, manually delete features, reload), the user just looks up the
    offending segment's ID (e.g. by clicking its pixel in str_segid.bil) and
    passes it here.
    """
    if seg_ids is None:
        seg_ids = list(range(1, n_segments + 1))
    if exclude_seg_ids:
        exclude_set = set(exclude_seg_ids)
        seg_ids = [s for s in seg_ids if s not in exclude_set]
    return seg_ids


def run_model(
    cfg: PyfldplnConfig,
    seg_ids: Optional[list[int]] = None,
    exclude_seg_ids: Optional[list[int]] = None,
) -> None:
    """Run the per-segment flood-fill simulation, replacing the parfeval loop
    in fldpln_wrapper_v8.m that calls fldpln_model_v8 for each segment.

    `run.n_workers` (null = os.cpu_count()) controls segment-level
    parallelism; see model/runner.py's docstring for how it's implemented.
    """
    fil, fil_info = read_raster(cfg.paths.fil_path, crs=cfg.paths.crs)
    fdr, _ = read_raster(cfg.paths.fdr_path, crs=cfg.paths.crs)
    dem, _ = read_raster(cfg.paths.dem_path, crs=cfg.paths.crs)
    fil = fil.astype(np.float64).ravel()
    fdr = fdr.ravel()
    dem = dem.astype(np.float64).ravel()
    bg = fil_info.nodata if fil_info.nodata is not None else 0.0

    seg_info = pd.read_parquet(cfg.paths.seg_info_path)
    seg_ids = _resolve_seg_ids(len(seg_info), seg_ids, exclude_seg_ids)

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


def create_map(
    cfg: PyfldplnConfig,
    seg_ids: Optional[list[int]] = None,
    exclude_seg_ids: Optional[list[int]] = None,
) -> None:
    """Merge per-segment flood-fill results into one DTF map, replacing
    create_fldpln_map.m / custom_floodplain_map.m."""
    fdr_info = read_raster_info(cfg.paths.fdr_path, crs=cfg.paths.crs)
    seg_info = pd.read_parquet(cfg.paths.seg_info_path)
    seg_ids = _resolve_seg_ids(len(seg_info), seg_ids, exclude_seg_ids)
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
    p_run.add_argument(
        "--exclude-segments",
        help="comma-separated 1-based segment IDs to skip (e.g. a stray tiny "
        "segment) -- applied on top of --segments/the default 'all'",
    )

    p_map = sub.add_parser("create-map", help="merge segment results into one DTF map")
    p_map.add_argument("config", help="path to YAML config file")
    p_map.add_argument(
        "--segments", help="comma-separated 1-based segment IDs to merge (default: all)"
    )
    p_map.add_argument(
        "--exclude-segments",
        help="comma-separated 1-based segment IDs to skip -- applied on top "
        "of --segments/the default 'all'",
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
        exclude_seg_ids = (
            [int(s) for s in args.exclude_segments.split(",")]
            if args.exclude_segments
            else None
        )
        run_model(cfg, seg_ids=seg_ids, exclude_seg_ids=exclude_seg_ids)
    elif args.command == "create-map":
        cfg = load_config(args.config)
        seg_ids = (
            [int(s) for s in args.segments.split(",")] if args.segments else None
        )
        exclude_seg_ids = (
            [int(s) for s in args.exclude_segments.split(",")]
            if args.exclude_segments
            else None
        )
        create_map(cfg, seg_ids=seg_ids, exclude_seg_ids=exclude_seg_ids)

    return 0


if __name__ == "__main__":
    sys.exit(main())
