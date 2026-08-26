"""Unit test for the Parquet output writer, reusing the 3x3 synthetic
scenario from test_flood_fill.py."""

import numpy as np
import pandas as pd

from PyFLDPLN_src.segmentation.runner import output_basename, run_segment_to_disk
from tests.test_flood_fill import BG, EAST, N_COLS, N_ROWS, SOUTH, build_grid, px


def test_output_basename_matches_original_convention():
    assert output_basename(fldmx=36, dh=3, seg_id=5) == "h36_dh3_seg5"
    assert output_basename(fldmx=2.5, dh=0.5, seg_id=12) == "h2p5_dh0p5_seg12"


def test_run_segment_to_disk_writes_parquet(tmp_path):
    fil, fdr = build_grid()
    dem = fil.copy()  # DEM == FIL here -> fill_depth should be 0 everywhere

    seg_row = pd.Series(
        {"start_pixel": px(0, 1), "end_pixel": px(2, 1), "length": 3}
    )

    out_path = run_segment_to_disk(
        seg_id=1,
        seg_row=seg_row,
        fil=fil,
        fdr=fdr,
        dem=dem,
        fldmn=0.1,
        fldmx=5,
        dh=5,
        mxht0=0,
        ssflg=False,
        bg=BG,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        segdr=tmp_path,
    )

    assert out_path.name == "h5_dh5_seg1.parquet"
    assert out_path.exists()
    assert not (tmp_path / "h5_dh5_seg1_tmp.parquet").exists()  # cleaned up on completion

    df = pd.read_parquet(out_path)
    assert list(df.columns) == ["fsp", "pixel", "dtf", "fill_depth"]
    assert len(df) == 6
    assert np.allclose(df["fill_depth"], 0.0)  # fill depth: DEM==FIL everywhere

    by_pixel = {int(row.pixel): (int(row.fsp), row.dtf) for row in df.itertuples()}
    assert by_pixel[px(1, 2)] == (px(0, 1), 2.0)

    # re-running should short-circuit (final .parquet already exists) rather than recompute
    out_path2 = run_segment_to_disk(
        seg_id=1, seg_row=seg_row, fil=fil, fdr=fdr, dem=dem,
        fldmn=0.1, fldmx=5, dh=5, mxht0=0, ssflg=False, bg=BG,
        n_rows=N_ROWS, n_cols=N_COLS, segdr=tmp_path,
    )
    assert out_path2 == out_path
