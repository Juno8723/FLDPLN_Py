"""Entry point for turning a segment library (built once by
cli_segmentation.py) into a map-coordinate, tiled library: reorganize + tile,
run in that order as one command. Only needs re-running if the segment
library itself changes -- see cli_floodmap.py for the gauge-based event map,
which is the one stage meant to be re-run repeatedly (e.g. against fresh
gauge readings on a schedule) without touching this one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .cli_common import _resolve_seg_ids, parse_seg_id_list
from .config import PyfldplnConfig, load_config
from .io.raster import read_raster, read_raster_info
from .library_reorganization.fsp_table import add_downstream_distances, build_fsp_table
from .library_reorganization.library_files import build_segment_library_relations
from .library_reorganization.tiling import build_tiles, compute_relations_pixel_extents
from .library_reorganization.stream_order import assign_stream_order, build_stream_order_info


def organize_library(
    cfg: PyfldplnConfig,
    seg_ids: Optional[list[int]] = None,
    exclude_seg_ids: Optional[list[int]] = None,
) -> None:
    """Reorganize per-segment flood-fill output (written by
    cli_segmentation.py's run-model step) into a map-coordinate library:
    fsp_info.csv / segment_info.csv / stream_order_info.csv /
    CellSizeSpatialReference.json (matching
    fldpln_header.py's schemas exactly) plus one FSP-FPP relation Parquet
    per segment (schema-compatible with the existing tiled-library .snz
    format, see library_reorganization/library_files.py).

    Replaces fsp_info_table.m + library_files.m, and automates the stream-
    order assignment step that Tools/ReorganizeLibrary.ipynb otherwise
    requires doing by hand in ArcGIS (see library_reorganization/stream_order.py).
    Tiling (grouping relations into fixed-size geographic tiles for fast
    lookup) is a later, separate step -- not run here.
    """
    fil, fil_info = read_raster(cfg.paths.fil_path, crs=cfg.paths.crs)
    fdr, fdr_info = read_raster(cfg.paths.fdr_path, crs=cfg.paths.crs)
    fil = fil.astype(np.float64).ravel()
    fdr = fdr.ravel()
    n_rows, n_cols = fil_info.rows, fil_info.cols

    seg_info = pd.read_parquet(cfg.paths.seg_info_path)
    seg_ids = _resolve_seg_ids(len(seg_info), seg_ids, exclude_seg_ids)

    print("building FSP table ...")
    fsp_df = build_fsp_table(seg_info, fil, fdr, fdr_info.transform, n_rows, n_cols)

    print("computing downstream distances ...")
    seg_df = add_downstream_distances(seg_info, fsp_df)

    print("assigning stream order (automated level-path method) ...")
    order = assign_stream_order(seg_info)
    fsp_df["StrOrd"] = order[fsp_df["SegId"].to_numpy() - 1]
    seg_df["StrOrd"] = order
    str_ord_df = build_stream_order_info(seg_info, order, fsp_df)
    print(f"  {len(set(order.tolist()))} distinct stream order(s) assigned")

    lib_dir = Path(cfg.output.resolve(cfg.paths.home_dir).library_dir)
    lib_dir.mkdir(parents=True, exist_ok=True)

    fsp_out = fsp_df[["FspId", "FspX", "FspY", "SegId", "FilledElev", "DsDist", "StrOrd"]]
    fsp_out.to_csv(lib_dir / "fsp_info.csv", index=False)
    seg_df[["SegId", "CellCount", "DsSegId", "Length", "DsDist", "StrOrd"]].to_csv(
        lib_dir / "segment_info.csv", index=False
    )
    str_ord_df.to_csv(lib_dir / "stream_order_info.csv", index=False)

    with open(lib_dir / "CellSizeSpatialReference.json", "w") as jf:
        json.dump(
            {
                "CellSize": fil_info.pixel_size_x,
                "SpatialReference": fil_info.crs.to_wkt() if fil_info.crs else "",
            },
            jf,
        )
    print(f"library info files written to {lib_dir}")

    relations_dir = lib_dir / "relations"
    relations_dir.mkdir(parents=True, exist_ok=True)
    segdr = cfg.output.resolve(cfg.paths.home_dir).segment_files_dir
    for seg_id in seg_ids:
        rel = build_segment_library_relations(
            seg_id, fsp_df, cfg.flood.fldmx, cfg.flood.dh, segdr, n_cols
        )
        rel.to_parquet(relations_dir / f"seg{seg_id}.parquet", index=False)
        print(f"  segment {seg_id}: {len(rel)} relations -> {relations_dir / f'seg{seg_id}.parquet'}")


def tile_library(cfg: PyfldplnConfig) -> None:
    """Bucket the FSP-FPP relations already written by organize_library() into
    fixed-size geographic tiles (see library_reorganization/tiling.py). Reads
    library_dir/fsp_info.csv and library_dir/relations/seg*.parquet; writes
    library_dir/FLDPLN_tiled_*.snz + the two tile/fsp index CSVs.
    """
    lib_dir = Path(cfg.output.resolve(cfg.paths.home_dir).library_dir)
    fsp_df = pd.read_csv(lib_dir / "fsp_info.csv")

    relations_dir = lib_dir / "relations"
    relations_paths = {
        int(p.stem.removeprefix("seg")): p for p in relations_dir.glob("seg*.parquet")
    }
    if not relations_paths:
        raise FileNotFoundError(f"no relation Parquet files found in {relations_dir}")
    print(f"computing pixel extents for {len(relations_paths)} segment relation file(s) ...")
    seg_extents = compute_relations_pixel_extents(relations_paths)

    fdr_info = read_raster_info(cfg.paths.fdr_path, crs=cfg.paths.crs)
    build_tiles(
        seg_extents=seg_extents,
        fsp_df=fsp_df,
        tile_size_px=cfg.output.tile_size_px,
        pixel_size=fdr_info.pixel_size_x,
        transform=fdr_info.transform,
        out_dir=lib_dir,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pyfldpln-library-reorganization",
        description="Turn a segment library (built by pyfldpln-segmentation) "
        "into a map-coordinate, tiled library: reorganize + tile, run in that "
        "order. See pyfldpln-floodmap for the gauge-based event map that "
        "consumes this stage's output.",
    )
    parser.add_argument("config", help="path to YAML config file")
    parser.add_argument(
        "--segments",
        help="comma-separated 1-based segment IDs to include (default: all) "
        "-- applies to the library reorganization step, not tiling",
    )
    parser.add_argument(
        "--exclude-segments",
        help="comma-separated 1-based segment IDs to skip -- applied on top "
        "of --segments/the default 'all'",
    )

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    organize_library(
        cfg,
        seg_ids=parse_seg_id_list(args.segments),
        exclude_seg_ids=parse_seg_id_list(args.exclude_segments),
    )
    tile_library(cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
