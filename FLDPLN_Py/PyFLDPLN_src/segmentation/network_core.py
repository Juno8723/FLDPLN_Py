"""Shared segmentation core: FAC/length-based subdivision + segment raster writing.

Ports Step 3 and Step 4 of create_segdb_facthr_maxlen.m /
create_segdb_facthr_maxlen_seedpts.m. Both MATLAB files implement these two
steps with near-identical code; only Step 1-2 (how the initial downstream
paths/"reaches" are traced) differs between the 'auto' and 'seedpoints'
methods. Those front-ends live in auto.py / seedpoints.py and both call into
this module.

Indexing convention: pixel IDs are 0-based, row-major flat indices
(v = row * n_cols + col), unlike the original MATLAB's 1-based
v = (row-1)*n_cols + col. seg_info's 'downstream' column uses -1 to mean "no
downstream segment" (MATLAB used 0, since MATLAB row indices are 1-based).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import d8
from ..io.raster import RasterInfo, write_raster

SEG_INFO_COLUMNS = [
    "start_pixel",
    "end_pixel",
    "fac_start",
    "fac_end",
    "length",
    "downstream",
    "reach_id",
]


@dataclass
class ReachPath:
    """A single traced downstream path (natural segment / 'reach')."""

    pixels: np.ndarray  # (n,) int64, 0-based flat pixel ids, in downstream order
    fac: np.ndarray  # (n,) FAC value at each pixel
    reach_id: int


def _split_by_breaks(n: int, break_positions: list[int]) -> list[tuple[int, int]]:
    """Split index range [0, n) at the given 0-based break positions.

    A break at position p means "break between local index p and p+1".
    Returns a list of inclusive (start, end) local index ranges.
    """
    ranges = []
    prev = 0
    for p in break_positions:
        ranges.append((prev, p))
        prev = p + 1
    ranges.append((prev, n - 1))
    return ranges


def _tie_break_positions(sdf: np.ndarray) -> list[int]:
    """Reproduce the original MATLAB tie-break rule for bisecting an over-length
    subsegment at its largest internal FAC jump(s).

    - If 1 or 2 positions tie for the max jump, split at all of them.
    - If 3+ positions tie (a flat area), pick a single "middle-ish" one:
      the exact median for an odd count, or the lower-middle element for an
      even count (matching MATLAB's `f1(lf1/2)`, 1-based).
    """
    max_val = sdf.max()
    tied = np.flatnonzero(sdf == max_val).tolist()
    lf1 = len(tied)
    if lf1 > 2:
        if lf1 % 2 == 1:
            return [tied[(lf1 - 1) // 2]]
        else:
            return [tied[lf1 // 2 - 1]]
    return tied


def subdivide_reach(
    pixels: np.ndarray, fac: np.ndarray, facthr: float, seglen: float
) -> list[tuple[int, int]]:
    """Subdivide one reach's pixel path into subsegments.

    Returns a list of inclusive (start_idx, end_idx) local index ranges into
    `pixels`/`fac`, in downstream order.
    """
    n = len(pixels)
    fac = np.asarray(fac, dtype=np.float64)
    df = fac[1:] - fac[:-1]
    breaks = np.flatnonzero(df >= facthr).tolist()
    segments = _split_by_breaks(n, breaks)

    changed = True
    while changed:
        changed = False
        for i, (start, end) in enumerate(segments):
            length = end - start + 1
            if length > seglen:
                local_fac = fac[start : end + 1]
                sdf = local_fac[1:] - local_fac[:-1]
                local_breaks = _tie_break_positions(sdf)
                local_ranges = _split_by_breaks(end - start + 1, local_breaks)
                new_segments = [(start + a, start + b) for a, b in local_ranges]
                segments = segments[:i] + new_segments + segments[i + 1 :]
                changed = True
                break
    return segments


def build_segment_table(reaches: list[ReachPath], facthr: float, seglen: float) -> pd.DataFrame:
    """Build the full seg_info table across all reaches (Step 3).

    Also returns, via the returned DataFrame's attrs, a per-reach list of
    (pixel_id -> segment row index) assignments usable for raster writing.
    """
    rows = []
    pixel_seg_ids: list[np.ndarray] = []  # per reach, aligned with reach.pixels
    for reach in reaches:
        local_segments = subdivide_reach(reach.pixels, reach.fac, facthr, seglen)
        seg_id_for_pixel = np.empty(len(reach.pixels), dtype=np.int64)
        base_row = len(rows)
        for local_i, (start, end) in enumerate(local_segments):
            row_id = base_row + local_i
            rows.append(
                {
                    "start_pixel": int(reach.pixels[start]),
                    "end_pixel": int(reach.pixels[end]),
                    "fac_start": float(reach.fac[start]),
                    "fac_end": float(reach.fac[end]),
                    "length": end - start + 1,
                    "downstream": -1,
                    "reach_id": reach.reach_id,
                }
            )
            seg_id_for_pixel[start : end + 1] = row_id
        pixel_seg_ids.append(seg_id_for_pixel)

    table = pd.DataFrame(rows, columns=SEG_INFO_COLUMNS)
    table.attrs["pixel_segment_ids"] = pixel_seg_ids
    table.attrs["reach_pixels"] = [r.pixels for r in reaches]
    return table


def link_downstream_by_fdr_step(
    seg_table: pd.DataFrame, fdr: np.ndarray, n_rows: int, n_cols: int
) -> None:
    """Resolve 'downstream' for subsegments whose end coincides with a reach's
    end (i.e. still -1 after build_segment_table): step one more pixel via FDR
    from the end pixel and check whether that lands on another segment's start
    pixel. Mutates seg_table in place. Matches Step 3's downstream-linkage
    loop in create_segdb_facthr_maxlen.m.
    """
    start_lookup = {int(v): i for i, v in enumerate(seg_table["start_pixel"])}
    unresolved = seg_table.index[seg_table["downstream"] == -1]
    for row_id in unresolved:
        end_px = int(seg_table.at[row_id, "end_pixel"])
        row, col = divmod(end_px, n_cols)
        fdr_code = int(fdr[row, col])
        step = d8.step_downstream(row, col, fdr_code)
        if step is None:
            continue
        nrow, ncol = step
        if not (0 <= nrow < n_rows and 0 <= ncol < n_cols):
            continue
        next_px = nrow * n_cols + ncol
        target = start_lookup.get(next_px)
        if target is not None:
            seg_table.at[row_id, "downstream"] = target


def _segment_dtype(n_segments: int) -> tuple[np.dtype, str]:
    if n_segments < 256:
        return np.dtype(np.uint8), "uint8"
    elif n_segments < 65536:
        return np.dtype(np.uint16), "uint16"
    else:
        return np.dtype(np.uint32), "uint32"


def write_segment_raster(
    path: str,
    seg_table: pd.DataFrame,
    n_rows: int,
    n_cols: int,
    reference: RasterInfo,
) -> None:
    """Write the per-pixel segment-ID raster (Step 4). Segment row IDs are
    written 1-based (0 reserved for background/no-segment), matching the
    original BIL convention where 0 = no stream pixel.
    """
    dtype, _ = _segment_dtype(len(seg_table) + 1)
    grid = np.zeros((n_rows, n_cols), dtype=dtype)

    pixel_segment_ids = seg_table.attrs["pixel_segment_ids"]
    reach_pixels = seg_table.attrs["reach_pixels"]
    for pixels, seg_ids in zip(reach_pixels, pixel_segment_ids):
        rows, cols = np.divmod(pixels, n_cols)
        grid[rows, cols] = (seg_ids + 1).astype(dtype)  # +1: reserve 0 for background

    write_raster(path, grid, like=reference, dtype=dtype.name, nodata=0)
