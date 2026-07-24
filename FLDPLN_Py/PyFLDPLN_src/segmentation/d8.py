"""D8 flow-direction constants shared by segmentation and the flood-fill model.

Preserves the exact ESRI D8 encoding and neighbor ordering used throughout the
original MATLAB code (see comments in fldpln_model_v8.m /
create_segdb_facthr_maxlen.m):

    3x3 neighbor position layout (center excluded):
        NW  N  NE
        W   x  E
        SW  S  SE

    ESRI flow-direction codes at each position:
        32  64  128
        16  x   1
        8   4   2

Pixel indexing here is 0-based, row-major (v = row * n_cols + col), unlike the
original MATLAB's 1-based column-vector linear index (v = (row-1)*n_cols+col).
The neighbor *offset* values are base-invariant so they carry over unchanged
from the MATLAB 'bdy0' array; only boundary comparisons shift by one.
"""

from __future__ import annotations

import numpy as np

# Neighbor order: NW, N, NE, W, E, SW, S, SE (matches original 'bdy0' ordering)
ROW_OFFSETS = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
COL_OFFSETS = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)

# FDR value at a pixel if it flows OUT toward that neighbor position
OUTFLOW_CODES = np.array([32, 64, 128, 16, 1, 8, 4, 2], dtype=np.uint8)

# FDR value a neighbor at that position must have to flow INTO the center pixel
INFLOW_CODES = np.array([2, 4, 8, 1, 16, 128, 64, 32], dtype=np.uint8)

# Lookup table: OUTFLOW_LUT[code] -> neighbor position index (0-7), or -1 if invalid.
# Sized 256 to cover the full uint8 range of possible (garbage) FDR values.
_OUTFLOW_LUT = np.full(256, -1, dtype=np.int64)
_OUTFLOW_LUT[OUTFLOW_CODES] = np.arange(8)


def flat_offsets(n_cols: int) -> np.ndarray:
    """8-neighbor flat-index offsets for a 0-based row-major array with n_cols columns."""
    return ROW_OFFSETS * n_cols + COL_OFFSETS


def outflow_position(fdr_code: int) -> int | None:
    """Neighbor position index (0-7) that this FDR code flows toward, or None if invalid."""
    if not (0 <= fdr_code < 256):
        return None
    pos = int(_OUTFLOW_LUT[fdr_code])
    return pos if pos >= 0 else None


def outflow_positions(fdr_codes: np.ndarray) -> np.ndarray:
    """Vectorized outflow_position: returns -1 (not None) for invalid codes."""
    codes = np.asarray(fdr_codes).astype(np.int64)
    valid = (codes >= 0) & (codes < 256)
    out = np.full(codes.shape, -1, dtype=np.int64)
    out[valid] = _OUTFLOW_LUT[codes[valid]]
    return out


def step_downstream(row: int, col: int, fdr_code: int) -> tuple[int, int] | None:
    """Follow one FDR step downstream from (row, col). None if fdr_code is not a valid outflow code."""
    pos = outflow_position(fdr_code)
    if pos is None:
        return None
    return row + int(ROW_OFFSETS[pos]), col + int(COL_OFFSETS[pos])


def neighbor_rows_cols(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n,) center pixels -> (n, 8) row/col arrays of all 8 neighbors."""
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    nrow = rows[:, None] + ROW_OFFSETS[None, :]
    ncol = cols[:, None] + COL_OFFSETS[None, :]
    return nrow, ncol


def in_bounds(row: np.ndarray, col: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    return (row >= 0) & (row < n_rows) & (col >= 0) & (col < n_cols)
