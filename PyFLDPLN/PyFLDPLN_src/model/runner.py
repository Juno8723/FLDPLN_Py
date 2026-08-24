"""Segment-level orchestration for the flood-fill core.

Replaces fldpln_wrapper_v8.m's per-segment I/O, checkpointing, and output
file naming. Output files are Parquet tables (`h{fldmx}_dh{dh}_seg{id}.parquet`
/ `_tmp.parquet`, columns `fsp`, `pixel`, `dtf`, `fill_depth`) rather than the
original's MATLAB .mat format -- now that the whole pipeline is validated
against the original (byte-identical segmentation and flood-fill vs. the
MATLAB reference), there's no more need to round-trip through MATLAB's
serialization. Parquet keeps the same fast, columnar read path merge.py
already uses for seg_info, and is directly viewable without Python (e.g. a
Parquet-aware editor extension, or a standalone viewer like ParquetViewer)
instead of requiring MATLAB or scipy.io to inspect a segment's output.

Segment-level parallelism (matching the original's parfeval loop / this
project's `run.n_workers` config) mirrors the original: each segment is an
independent unit of work (fil/fdr/dem are read-only throughout the flood-fill
core, and merge is a pixel-wise min over segments -- see flood_fill.py and
mapping/merge.py -- so segments never interfere with each other, and
processing order doesn't affect any individual segment's result).
`run_library(..., n_workers>1)` parallelizes across segments with a
ProcessPoolExecutor pinned to the 'fork' start method: fil/fdr/dem are set as
a module-level global right before the pool is created, so worker processes
inherit them via copy-on-write instead of being pickled/copied per task --
essential since these arrays can be tens to hundreds of millions of pixels on
real AOIs. This only works on platforms where 'fork' is available (Linux; not
Windows, not macOS's default), which matches this project's HPC target.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

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
    """Run the flood-fill simulation for one segment and write its output
    (skipping segments already fully processed, matching the original's
    `if(~exist([outf,'.mat'],'file'))` guard, adapted to Parquet's
    `.parquet` extension)."""
    segdr = Path(segdr)
    segdr.mkdir(parents=True, exist_ok=True)
    basename = output_basename(fldmx, dh, seg_id)
    final_path = segdr / f"{basename}.parquet"
    tmp_path = segdr / f"{basename}_tmp.parquet"

    if final_path.exists():
        return final_path

    def checkpoint(fldht: float, entries) -> None:
        df = pd.DataFrame(
            {
                "fsp": [e.fsp for e in entries],
                "pixel": [e.pixel for e in entries],
                "dtf": [e.dtf for e in entries],
                # Constant column carrying the current depth step -- cheap in
                # Parquet (dictionary/RLE-encoded) and lets a checkpoint be a
                # single self-contained file instead of needing a metadata
                # sidecar.
                "fldht": fldht,
            }
        )
        df.to_parquet(tmp_path, index=False)
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
    fsp = np.array([e.fsp for e in entries], dtype=np.int64)
    dtf = np.array([e.dtf for e in entries], dtype=np.float64)
    fill_depth = np.maximum(0.0, fil[pixels] - dem[pixels]) if len(entries) else np.empty(0)

    df = pd.DataFrame({"fsp": fsp, "pixel": pixels, "dtf": dtf, "fill_depth": fill_depth})
    df.to_parquet(final_path, index=False)
    if tmp_path.exists():
        tmp_path.unlink()
    return final_path


def _run_one_segment(seg_id: int) -> tuple[int, "Path | None", "str | None"]:
    """Worker-process entry point (n_workers > 1 branch only). Reads shared
    inputs from `_worker_ctx`, populated by run_library right before the pool
    is created. Exceptions are caught and returned rather than raised so one
    bad segment doesn't kill the whole batch."""
    ctx = _worker_ctx
    seg_row = ctx["seg_info"].iloc[seg_id - 1]
    try:
        path = run_segment_to_disk(
            seg_id, seg_row, ctx["fil"], ctx["fdr"], ctx["dem"],
            ctx["fldmn"], ctx["fldmx"], ctx["dh"], ctx["mxht0"], ctx["ssflg"],
            ctx["bg"], ctx["n_rows"], ctx["n_cols"], ctx["segdr"],
        )
        return seg_id, path, None
    except Exception as exc:  # noqa: BLE001 -- reported to the parent, not swallowed
        return seg_id, None, repr(exc)


# Populated in the parent process right before the worker pool is created
# (run_library, n_workers > 1 branch only) so forked workers inherit
# fil/fdr/dem/seg_info via copy-on-write rather than having them pickled per
# task. Never touched in the sequential (n_workers <= 1) path.
_worker_ctx: dict = {}


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
    n_workers: int = 1,
) -> list[Path]:
    """Run the flood-fill simulation for every segment in seg_info (or just
    `seg_ids` if given). seg_id is the 1-based row position in seg_info
    (matching the original's row-index-as-segment-ID convention).

    n_workers <= 1 runs sequentially in-process (default, and what the test
    suite exercises). n_workers > 1 parallelizes across segments -- see the
    module docstring for how fil/fdr/dem sharing works. Segments that were
    already completed on a prior run are skipped either way (final .parquet
    already exists, checked in run_segment_to_disk), so a parallel run can
    safely resume where a killed sequential (or parallel) run left off.
    """
    if seg_ids is None:
        seg_ids = list(range(1, len(seg_info) + 1))
    segdr = Path(segdr)

    if n_workers <= 1:
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

    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError(
            "run.n_workers > 1 requires the 'fork' multiprocessing start method "
            "(Linux only). Set run.n_workers: 1 on this platform."
        )

    n_workers = min(n_workers, len(seg_ids))
    _worker_ctx.clear()
    _worker_ctx.update(
        seg_info=seg_info, fil=fil, fdr=fdr, dem=dem,
        fldmn=fldmn, fldmx=fldmx, dh=dh, mxht0=mxht0, ssflg=ssflg, bg=bg,
        n_rows=n_rows, n_cols=n_cols, segdr=segdr,
    )
    outputs_by_id: dict[int, Path] = {}
    failures: list[tuple[int, str]] = []
    try:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = {executor.submit(_run_one_segment, sid): sid for sid in seg_ids}
            done = 0
            for future in as_completed(futures):
                seg_id, path, error = future.result()
                done += 1
                if error is not None:
                    failures.append((seg_id, error))
                    print(f"segment {seg_id} FAILED ({done}/{len(seg_ids)}): {error}")
                else:
                    outputs_by_id[seg_id] = path
                    print(f"segment {seg_id} completed ({done}/{len(seg_ids)}) -> {path.name}")
    finally:
        _worker_ctx.clear()

    if failures:
        print(f"{len(failures)} segment(s) failed: {sorted(s for s, _ in failures)}")
    return [outputs_by_id[sid] for sid in seg_ids if sid in outputs_by_id]
