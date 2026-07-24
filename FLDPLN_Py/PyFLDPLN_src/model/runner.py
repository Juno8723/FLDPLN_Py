"""Segment-level orchestration for the flood-fill core.

Replaces fldpln_wrapper_v8.m's per-segment I/O, checkpointing, and output
file naming. Output files are genuine MATLAB .mat files (via scipy.io) using
the same naming convention and variable layout as the original
(`h{fldmx}_dh{dh}_seg{id}.mat` / `_tmp.mat`, with a `fldpln` array of
[FSP, floodplain pixel, DTF, fill depth]), so they remain directly
inspectable/loadable the way the original outputs were.

NOTE: segment-level parallelism (matching the original's parfeval loop /
this project's `run.n_workers` config) is not yet implemented here --
segments are run sequentially, one at a time, in-process. Parallelizing
efficiently requires sharing the (potentially very large) fil/fdr/dem arrays
across worker processes without copying them per-task (e.g. via memmap or
shared_memory), which is a follow-up, not yet done.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from .flood_fill import run_segment


def _format_number_for_filename(x: float) -> str:
    """Matches the original's num2str(x) with '.' replaced by 'p'."""
    s = f"{x:g}"
    return s.replace(".", "p")


def output_basename(fldmx: float, dh: float, seg_id: int) -> str:
    return (
        f"h{_format_number_for_filename(fldmx)}"
        f"_dh{_format_number_for_filename(dh)}_seg{seg_id}"
    )


def run_segment_to_disk(
    seg_id: int,
    seg_row: pd.Series,
    fil: np.ndarray,
    fdr: np.ndarray,
    dem: np.ndarray,
    fldmn: float,
    fldmx: float,
    dh: float,
    mxht0: float,
    ssflg: bool,
    bg: float,
    n_rows: int,
    n_cols: int,
    segdr: Path,
) -> Path:
    """Run the flood-fill simulation for one segment and write its .mat
    output (skipping segments already fully processed, matching the
    original's `if(~exist([outf,'.mat'],'file'))` guard)."""
    segdr = Path(segdr)
    segdr.mkdir(parents=True, exist_ok=True)
    basename = output_basename(fldmx, dh, seg_id)
    final_path = segdr / f"{basename}.mat"
    tmp_path = segdr / f"{basename}_tmp.mat"

    if final_path.exists():
        return final_path

    def checkpoint(fldht: float, entries) -> None:
        arr = np.array([[e.fsp, e.pixel, e.dtf] for e in entries], dtype=np.float64)
        savemat(
            str(tmp_path),
            {"fldht": float(fldht), "fldpln_info": arr, "ct_tot": len(entries)},
        )
        print(f"  seg {seg_id}: fldht={fldht:g}, {len(entries)} floodplain pixels so far")

    entries = run_segment(
        seg_id=seg_id,
        start_pixel=int(seg_row["start_pixel"]),
        end_pixel=int(seg_row["end_pixel"]),
        length=int(seg_row["length"]),
        fil=fil,
        fdr=fdr,
        fldmn=fldmn,
        fldmx=fldmx,
        dh=dh,
        mxht0=mxht0,
        ssflg=ssflg,
        bg=bg,
        n_rows=n_rows,
        n_cols=n_cols,
        on_iteration=checkpoint,
    )

    pixels = np.array([e.pixel for e in entries], dtype=np.int64)
    fldpln = np.zeros((len(entries), 4), dtype=np.float64)
    if len(entries):
        fldpln[:, 0] = [e.fsp for e in entries]
        fldpln[:, 1] = pixels
        fldpln[:, 2] = [e.dtf for e in entries]
        fldpln[:, 3] = np.maximum(0.0, fil[pixels] - dem[pixels])

    savemat(
        str(final_path),
        {
            "fldpln": fldpln,
            "header": np.array(
                [
                    "reference stream pixel",
                    "floodplain pixel",
                    "flood height",
                    "fill depth",
                ],
                dtype=object,
            ),
        },
    )
    if tmp_path.exists():
        tmp_path.unlink()
    return final_path


def run_library(
    seg_info: pd.DataFrame,
    fil: np.ndarray,
    fdr: np.ndarray,
    dem: np.ndarray,
    fldmn: float,
    fldmx: float,
    dh: float,
    mxht0: float,
    ssflg: bool,
    bg: float,
    n_rows: int,
    n_cols: int,
    segdr: Path,
    seg_ids: list[int] | None = None,
) -> list[Path]:
    """Run the flood-fill simulation for every segment in seg_info (or just
    `seg_ids` if given), sequentially. seg_id is the 1-based row position in
    seg_info (matching the original's row-index-as-segment-ID convention).
    """
    if seg_ids is None:
        seg_ids = list(range(1, len(seg_info) + 1))

    outputs = []
    for seg_id in seg_ids:
        seg_row = seg_info.iloc[seg_id - 1]
        path = run_segment_to_disk(
            seg_id, seg_row, fil, fdr, dem, fldmn, fldmx, dh, mxht0, ssflg, bg,
            n_rows, n_cols, segdr,
        )
        outputs.append(path)
        print(f"segment {seg_id} of {len(seg_ids)} completed -> {path.name}")
    return outputs
