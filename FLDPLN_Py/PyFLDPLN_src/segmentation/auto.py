"""Automatic stream-network segmentation, replacing create_segdb_facthr_maxlen.m
(Steps 0-2: stream pixel detection, headwater/confluence classification, and
downstream path tracing). Steps 3-4 (subdivision + raster write) are shared
via network_core.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import d8
from .network_core import ReachPath, build_segment_table, link_downstream_by_fdr_step


def find_stream_pixels(
    fac: np.ndarray, strthr: float, nodata: float | None
) -> np.ndarray:
    """Boolean mask of pixels belonging to the stream network (FAC >= strthr)."""
    mask = fac >= strthr
    if nodata is not None:
        mask &= fac != nodata
    return mask


def classify_headwater_confluence(
    stream_mask: np.ndarray, fdr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For every stream pixel, count inflowing stream-pixel neighbors.

    0 inflows -> headwater; >1 inflows -> confluence. Matches Step 1 of
    create_segdb_facthr_maxlen.m (boundary pixels naturally have fewer valid
    neighbors since out-of-bounds neighbor positions are excluded).
    """
    n_rows, n_cols = stream_mask.shape
    rows, cols = np.nonzero(stream_mask)

    nrow, ncol = d8.neighbor_rows_cols(rows, cols)  # (n, 8)
    valid = d8.in_bounds(nrow, ncol, n_rows, n_cols)

    is_stream_neighbor = np.zeros(nrow.shape, dtype=bool)
    neighbor_fdr = np.zeros(nrow.shape, dtype=np.int64)
    vr, vc = nrow[valid], ncol[valid]
    is_stream_neighbor[valid] = stream_mask[vr, vc]
    neighbor_fdr[valid] = fdr[vr, vc].astype(np.int64)

    pos_idx = np.broadcast_to(np.arange(8), nrow.shape)
    inflow_match = neighbor_fdr == d8.INFLOW_CODES[pos_idx]
    inflow_count = (is_stream_neighbor & inflow_match & valid).sum(axis=1)

    headwater_mask = np.zeros((n_rows, n_cols), dtype=bool)
    confluence_mask = np.zeros((n_rows, n_cols), dtype=bool)
    headwater_mask[rows[inflow_count == 0], cols[inflow_count == 0]] = True
    confluence_mask[rows[inflow_count > 1], cols[inflow_count > 1]] = True
    return headwater_mask, confluence_mask


def trace_reach(
    start_row: int,
    start_col: int,
    fdr: np.ndarray,
    confluence_mask: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace one natural segment downstream from (start_row, start_col).

    Stops (excluding the stopping pixel) when the path would: leave the image
    extent, reach a confluence pixel, or hit an invalid FDR value. The
    confluence pixel that terminates a reach is never included in it -- it
    becomes the start of its own, separately-traced reach. Matches Step 2 of
    create_segdb_facthr_maxlen.m.
    """
    rows = [start_row]
    cols = [start_col]
    row, col = start_row, start_col
    while True:
        fdr_code = int(fdr[row, col])
        step = d8.step_downstream(row, col, fdr_code)
        if step is None:
            break
        nrow, ncol = step
        if not (0 <= nrow < n_rows and 0 <= ncol < n_cols):
            break
        if confluence_mask[nrow, ncol]:
            break
        if d8.outflow_position(int(fdr[nrow, ncol])) is None:
            break
        rows.append(nrow)
        cols.append(ncol)
        row, col = nrow, ncol
    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)


def build_auto_segments(
    fac: np.ndarray,
    fdr: np.ndarray,
    fac_nodata: float | None,
    strthr: float,
    facthr: float,
    seglen: float,
) -> pd.DataFrame:
    """Full auto-segmentation pipeline: Steps 0-4."""
    n_rows, n_cols = fac.shape
    stream_mask = find_stream_pixels(fac, strthr, fac_nodata)
    headwater_mask, confluence_mask = classify_headwater_confluence(stream_mask, fdr)

    # Headwaters first (raster-scan order), then confluences (raster-scan
    # order) -- matches the original's `seg_info0 = [[strhd; strcf] ...]`
    # (Step 2), which processes the two groups separately rather than
    # merging them into one spatially-interleaved scan order. Getting this
    # order right matters even though it doesn't change *which* pixels end
    # up in which segment (that's fully determined by trace_reach/
    # subdivide_reach) -- it changes which segment gets which ID number,
    # since IDs are assigned in traversal order.
    hw_rows, hw_cols = np.nonzero(headwater_mask)
    cf_rows, cf_cols = np.nonzero(confluence_mask)
    start_rows = np.concatenate([hw_rows, cf_rows])
    start_cols = np.concatenate([hw_cols, cf_cols])

    reaches: list[ReachPath] = []
    for reach_id, (r0, c0) in enumerate(zip(start_rows, start_cols), start=1):
        rows, cols = trace_reach(int(r0), int(c0), fdr, confluence_mask, n_rows, n_cols)
        pixels = rows * n_cols + cols
        reach_fac = fac[rows, cols].astype(np.float64)
        reaches.append(ReachPath(pixels=pixels, fac=reach_fac, reach_id=reach_id))

    seg_table = build_segment_table(reaches, facthr, seglen)
    link_downstream_by_fdr_step(seg_table, fdr, n_rows, n_cols)
    return seg_table
