"""DTF map creation driver, porting create_fldpln_map.m: merges every
segment's flood-fill output into one whole-AOI DTF raster.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..io.raster import RasterInfo
from .merge import merge_segments, write_dtf_map


def create_fldpln_map(
    seg_info: pd.DataFrame,
    fdr_info: RasterInfo,
    fldmn: float,
    fldmx: float,
    dh: float,
    segdr: Path,
    out_path: str,
    seg_ids: Optional[list[int]] = None,
) -> None:
    """Merge every (or a chosen subset of) segment's flood-fill result into a
    single DTF raster and write it out. seg_id is the 1-based row position in
    seg_info, matching the rest of this package's convention.
    """
    if seg_ids is None:
        seg_ids = list(range(1, len(seg_info) + 1))

    grid = merge_segments(
        seg_ids=seg_ids,
        n_rows=fdr_info.rows,
        n_cols=fdr_info.cols,
        fldmn=fldmn,
        fldmx=fldmx,
        dh=dh,
        segdr=segdr,
    )
    write_dtf_map(out_path, grid, fdr_info)
