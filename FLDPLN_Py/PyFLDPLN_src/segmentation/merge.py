"""DTF map merging, porting custom_floodplain_map.m.

Merges per-segment flood-fill Parquet outputs (written by
PyFLDPLN_src/segmentation/runner.py) into a single raster. Where multiple
segments' floodplains cover the same pixel, the shallowest (minimum) DTF
wins -- matching the original's `min(ln0, max(fldmn, fldpln(:,2)))` merge
rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from ..io.raster import RasterInfo, write_raster
from .runner import output_basename


def _load_segment_pixel_dtf(segdr: Path, fldmx: float, dh: float, seg_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a segment's (pixel, DTF) arrays, from the final .parquet if fully
    processed, falling back to the checkpoint _tmp.parquet otherwise
    (matching the original's same final-or-checkpoint fallback)."""
    basename = output_basename(fldmx, dh, seg_id)
    final_path = segdr / f"{basename}.parquet"
    if final_path.exists():
        df = pd.read_parquet(final_path, columns=["pixel", "dtf"])
    else:
        tmp_path = segdr / f"{basename}_tmp.parquet"
        if not tmp_path.exists():
            raise FileNotFoundError(f"no output found for segment {seg_id} in {segdr}")
        df = pd.read_parquet(tmp_path, columns=["pixel", "dtf"])
    pixels = df["pixel"].to_numpy(dtype=np.int64)
    dtf = df["dtf"].to_numpy(dtype=np.float64)
    return pixels, dtf


def merge_segments(
    seg_ids: list[int],
    n_rows: int,
    n_cols: int,
    fldmn: float,
    fldmx: float,
    dh: float,
    segdr: Union[str, Path],
    seg_depth_cutoff: Optional[Union[float, dict[int, float]]] = None,
) -> np.ndarray:
    """Merge per-segment flood-fill Parquet outputs into a single DTF raster
    (0 = background / never flooded), keeping the shallowest DTF on overlap.

    `seg_depth_cutoff`: optional per-segment (dict) or uniform (float) max
    DTF to include -- matches the original's `seg_dpth` parameter, letting
    the caller cap individual segments to a shallower depth than `fldmx`
    without re-running the flood-fill.
    """
    segdr = Path(segdr)
    grid = np.zeros(n_rows * n_cols, dtype=np.float32)

    for seg_id in seg_ids:
        pixels, dtf = _load_segment_pixel_dtf(segdr, fldmx, dh, seg_id)

        if seg_depth_cutoff is not None:
            cutoff = (
                seg_depth_cutoff.get(seg_id, fldmx)
                if isinstance(seg_depth_cutoff, dict)
                else seg_depth_cutoff
            )
            keep = dtf <= cutoff
            pixels = pixels[keep]
            dtf = dtf[keep]

        dtf = np.maximum(fldmn, dtf)

        existing = grid[pixels]
        better = (existing == 0) | (dtf < existing)
        grid[pixels[better]] = dtf[better].astype(np.float32)

    return grid.reshape(n_rows, n_cols)


def write_dtf_map(out_path: str, grid: np.ndarray, reference: RasterInfo) -> None:
    write_raster(out_path, grid, like=reference, dtype="float32", nodata=0)
