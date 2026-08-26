"""Unit test for the DTF map merge (min-rule across overlapping segments)."""

import pandas as pd

from PyFLDPLN_src.segmentation.merge import merge_segments


def _write_segment_parquet(path, entries):
    """entries: list of (fsp, pixel, dtf, fill_depth)."""
    df = pd.DataFrame(entries, columns=["fsp", "pixel", "dtf", "fill_depth"])
    df.to_parquet(path, index=False)


def test_merge_segments_keeps_shallowest_on_overlap(tmp_path):
    n_rows, n_cols = 3, 3
    # segment 1 covers pixels 0,1,4 with DTF 1.0, 2.0, 5.0
    _write_segment_parquet(
        tmp_path / "h5_dh1_seg1.parquet",
        [(0, 0, 1.0, 0.0), (0, 1, 2.0, 0.0), (0, 4, 5.0, 0.0)],
    )
    # segment 2 covers pixels 4,5 with DTF 3.0 (shallower than seg1's 5.0 at
    # pixel 4), 1.5
    _write_segment_parquet(
        tmp_path / "h5_dh1_seg2.parquet",
        [(1, 4, 3.0, 0.0), (1, 5, 1.5, 0.0)],
    )

    grid = merge_segments(
        seg_ids=[1, 2], n_rows=n_rows, n_cols=n_cols,
        fldmn=0.1, fldmx=5, dh=1, segdr=tmp_path,
    )
    flat = grid.flatten()

    assert flat[0] == 1.0
    assert flat[1] == 2.0
    assert flat[2] == 0.0  # never flooded
    assert flat[4] == 3.0  # min(5.0, 3.0) -> segment 2 wins
    assert flat[5] == 1.5


def test_merge_segments_floors_at_fldmn(tmp_path):
    _write_segment_parquet(tmp_path / "h5_dh1_seg1.parquet", [(0, 0, 0.0, 0.0)])
    grid = merge_segments(
        seg_ids=[1], n_rows=2, n_cols=2, fldmn=0.05, fldmx=5, dh=1, segdr=tmp_path,
    )
    assert grid.flatten()[0] == 0.05


def test_merge_segments_falls_back_to_tmp_checkpoint(tmp_path):
    df = pd.DataFrame({"fsp": [0], "pixel": [3], "dtf": [2.0], "fldht": [2.0]})
    df.to_parquet(tmp_path / "h5_dh1_seg1_tmp.parquet", index=False)
    grid = merge_segments(
        seg_ids=[1], n_rows=2, n_cols=2, fldmn=0.1, fldmx=5, dh=1, segdr=tmp_path,
    )
    assert grid.flatten()[3] == 2.0
