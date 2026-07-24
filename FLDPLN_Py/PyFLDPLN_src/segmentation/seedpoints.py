"""User seed-point stream network segmentation, replacing
create_segdb_facthr_maxlen_seedpts.m (Steps 1-2: seed coordinate -> pixel
conversion, downstream tracing, and priority-based overlap trimming). Steps
3-4 (subdivision + raster write) are shared via network_core.py.

Seed points must be supplied main-stem-first, then tributaries in decreasing
priority order (matching the original docstring's requirement): when two
seed paths overlap, the lower-priority (later) path is truncated at the
first (most upstream) pixel it shares with a higher-priority path.

NOTE: the original MATLAB implementation only handles the case where such a
confluence pixel happens to coincide with an already-existing natural segment
break in the higher-priority path. If it doesn't, the original raises
`error('Need to add code for forcing a new segment break')` -- an
acknowledged, unimplemented gap. This port preserves that behavior exactly
(raising NotImplementedError) rather than inventing new tie-break logic the
original never defined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rasterio.transform import rowcol

from . import d8
from ..io.raster import RasterInfo
from .network_core import ReachPath, build_segment_table


def geo_to_pixel(x: float, y: float, info: RasterInfo) -> tuple[int, int]:
    """Convert a geographic (x, y) coordinate to the (row, col) pixel that
    contains it.

    NOTE: the original MATLAB code computed round((uly-y)/pxsz) using
    ESRI's ULXMAP/ULYMAP, which are the *centroid* of the upper-left pixel.
    rasterio's `transform` instead uses the pixel *corner* (standard GDAL
    geotransform convention). Mixing the two conventions naively (e.g. round()
    against a corner-based transform) introduces a systematic half-pixel
    offset. Using rasterio's own corner-based inverse transform with floor
    (via `rowcol`) avoids that convention mismatch entirely and is the
    geometrically correct way to find "which pixel contains this point".
    """
    row, col = rowcol(info.transform, x, y)
    return int(row), int(col)


def trace_seed_path(
    start_row: int,
    start_col: int,
    fdr: np.ndarray,
    fac: np.ndarray,
    n_rows: int,
    n_cols: int,
    end_row: int | None = None,
    end_col: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace downstream from a seed point via FDR.

    Stops when the path leaves the image extent, hits an invalid FDR value,
    or (if end_row/end_col given) reaches the specified endpoint -- inclusive
    of the endpoint pixel itself.
    """
    rows = [start_row]
    cols = [start_col]
    row, col = start_row, start_col
    if end_row is not None and row == end_row and col == end_col:
        return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)

    while True:
        fdr_code = int(fdr[row, col])
        step = d8.step_downstream(row, col, fdr_code)
        if step is None:
            break
        nrow, ncol = step
        if not (0 <= nrow < n_rows and 0 <= ncol < n_cols):
            break
        rows.append(nrow)
        cols.append(ncol)
        row, col = nrow, ncol
        if end_row is not None and row == end_row and col == end_col:
            break
    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)


def resolve_seed_paths(
    paths: list[tuple[np.ndarray, np.ndarray]], n_cols: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[int, int]]:
    """Trim lower-priority paths at their first (most upstream) intersection
    with any higher-priority path already processed.

    Returns (trimmed_paths, conpts), where conpts maps path index -> flat
    pixel id of the (final, most-upstream) confluence pixel where that path
    was cut off. Only the final trim point is kept per path (earlier,
    superseded trim points from the same path are not meaningful once a
    further-upstream trim has been applied).
    """
    npts = len(paths)
    trimmed: list[tuple[np.ndarray, np.ndarray]] = [(r.copy(), c.copy()) for r, c in paths]
    conpts: dict[int, int] = {}

    for j in range(1, npts):
        rj, cj = trimmed[j]
        for k in range(j):
            rk, ck = trimmed[k]
            k_pixels = set(zip(rk.tolist(), ck.tolist()))
            hit = None
            for idx in range(len(rj)):
                if (int(rj[idx]), int(cj[idx])) in k_pixels:
                    hit = idx
                    break
            if hit is not None:
                conpts[j] = int(rj[hit]) * n_cols + int(cj[hit])
                rj = rj[:hit]
                cj = cj[:hit]
        trimmed[j] = (rj, cj)

    return trimmed, conpts


def link_confluences(seg_table: pd.DataFrame, conpts: dict[int, int]) -> None:
    """Force a downstream link from each truncated path's last subsegment to
    the subsegment starting at its confluence pixel -- only when that pixel
    coincides with an existing natural break. Mutates seg_table in place.
    """
    start_lookup = {int(v): i for i, v in enumerate(seg_table["start_pixel"])}
    for path_idx, confluence_px in conpts.items():
        target = start_lookup.get(confluence_px)
        if target is None:
            raise NotImplementedError(
                f"Seed path {path_idx} joins a higher-priority path at pixel "
                f"{confluence_px}, which does not coincide with an existing "
                "natural segment break. create_segdb_facthr_maxlen_seedpts.m "
                "left this case unimplemented (it raises an error rather than "
                "forcing a new break there) -- this is a known upstream gap, "
                "not a bug in this port. Adjust seed point placement or "
                "segmentation thresholds to avoid it."
            )
        reach_id = path_idx + 1
        reach_rows = seg_table.index[seg_table["reach_id"] == reach_id]
        last_row = reach_rows[-1]
        seg_table.at[last_row, "downstream"] = target


def build_seedpoint_segments(
    points_xy: np.ndarray,
    fdr: np.ndarray,
    fac: np.ndarray,
    raster_info: RasterInfo,
    facthr: float,
    seglen: float,
) -> pd.DataFrame:
    """Full seedpoints-segmentation pipeline: Steps 1-4.

    points_xy: (n, 2) array of [x, y] seed coordinates, or (n, 4) of
    [x, y, x_end, y_end] if segment endpoints are also specified. Points must
    be ordered main-stem-first, in decreasing priority.
    """
    n_rows, n_cols = fdr.shape
    npts, n_coord_cols = points_xy.shape
    has_endpoints = n_coord_cols == 4
    if n_coord_cols not in (2, 4):
        raise ValueError("points_xy must have 2 (start only) or 4 (start+end) columns")

    paths: list[tuple[np.ndarray, np.ndarray]] = []
    for j in range(npts):
        start_row, start_col = geo_to_pixel(points_xy[j, 0], points_xy[j, 1], raster_info)
        if not (0 <= start_row < n_rows and 0 <= start_col < n_cols):
            raise ValueError(f"seed point {j} falls outside the raster extent")

        end_row = end_col = None
        if has_endpoints:
            end_row, end_col = geo_to_pixel(points_xy[j, 2], points_xy[j, 3], raster_info)
            if not (0 <= end_row < n_rows and 0 <= end_col < n_cols):
                raise ValueError(f"seed point {j} endpoint falls outside the raster extent")

        rows, cols = trace_seed_path(
            start_row, start_col, fdr, fac, n_rows, n_cols, end_row, end_col
        )
        paths.append((rows, cols))

    trimmed_paths, conpts = resolve_seed_paths(paths, n_cols)

    reaches = []
    for j, (rows, cols) in enumerate(trimmed_paths):
        pixels = rows * n_cols + cols
        fac_vals = fac[rows, cols].astype(np.float64)
        reaches.append(ReachPath(pixels=pixels, fac=fac_vals, reach_id=j + 1))

    seg_table = build_segment_table(reaches, facthr, seglen)
    link_confluences(seg_table, conpts)
    return seg_table
