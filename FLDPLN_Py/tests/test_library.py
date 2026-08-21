"""Unit tests for the library-reorganization port (fsp_info_table.m /
library_files.m), using the same synthetic confluence grid as
test_segmentation.py."""

import math

import numpy as np
import pandas as pd
import pytest
from affine import Affine
from rasterio.transform import xy as transform_xy

from PyFLDPLN_src.library.fsp_table import add_downstream_distances, build_fsp_table
from PyFLDPLN_src.library.library_files import build_segment_library_relations
from PyFLDPLN_src.model.runner import output_basename
from PyFLDPLN_src.segmentation.auto import build_auto_segments
from PyFLDPLN_src.segmentation.units import METERS_PER_FOOT

N_ROWS, N_COLS = 5, 5


def _confluence_seg_info(synthetic_grid) -> pd.DataFrame:
    fdr, fac = synthetic_grid
    return build_auto_segments(fac, fdr, None, strthr=5, facthr=999, seglen=999)


def test_build_fsp_table_matches_seg_info_length_and_elevation(synthetic_grid):
    fdr, fac = synthetic_grid
    seg_info = _confluence_seg_info(synthetic_grid)
    fdr_flat = fdr.ravel()
    fil = np.arange(N_ROWS * N_COLS, dtype=np.float64)  # distinct value per pixel
    transform = Affine.identity()

    fsp_df = build_fsp_table(seg_info, fil, fdr_flat, transform, N_ROWS, N_COLS)

    assert len(fsp_df) == seg_info["length"].sum()
    for seg_id in range(1, len(seg_info) + 1):
        group = fsp_df[fsp_df["SegId"] == seg_id]
        assert len(group) == seg_info.loc[seg_id - 1, "length"]
        np.testing.assert_allclose(
            group["FilledElev"].to_numpy(),
            fil[group["_pixel"].to_numpy()] / METERS_PER_FOOT,
        )
        # FSP order within a segment must be upstream(start_pixel)->downstream.
        assert group["_pixel"].iloc[0] == seg_info.loc[seg_id - 1, "start_pixel"]

    # FspId is a dense 0-based row index over the whole concatenated table.
    assert fsp_df["FspId"].tolist() == list(range(len(fsp_df)))


def test_add_downstream_distances_zero_at_outlet_and_positive_upstream(synthetic_grid):
    seg_info = _confluence_seg_info(synthetic_grid)
    fdr_flat = synthetic_grid[0].ravel()
    fil = np.zeros(N_ROWS * N_COLS)
    transform = Affine.identity()
    fsp_df = build_fsp_table(seg_info, fil, fdr_flat, transform, N_ROWS, N_COLS)

    seg_df = add_downstream_distances(seg_info, fsp_df)

    outlet_seg_id = 3  # row 2 -> downstream == -1
    assert seg_df.loc[seg_df["SegId"] == outlet_seg_id, "DsDist"].iloc[0] == 0.0
    # The outlet's most-downstream FSP has DsDist 0 too (last FSP = distance 0
    # to itself, plus its segment's own DsDist of 0).
    outlet_fsps = fsp_df[fsp_df["SegId"] == outlet_seg_id]
    assert outlet_fsps["DsDist"].iloc[-1] == 0.0

    # Both upstream segments (1 and 2) must have a strictly positive DsDist
    # (real distance to the outlet, including the confluence gap).
    for seg_id in (1, 2):
        assert seg_df.loc[seg_df["SegId"] == seg_id, "DsDist"].iloc[0] > 0.0

    # Within a segment, DsDist strictly decreases from the first (most
    # upstream) FSP to the last (most downstream, which is 0 only for the
    # segment containing the true network outlet).
    for seg_id in range(1, 4):
        g = fsp_df[fsp_df["SegId"] == seg_id]["DsDist"].to_numpy()
        assert (g[:-1] >= g[1:]).all()


def test_add_downstream_distances_rejects_pixel_count_mismatch():
    seg_info = pd.DataFrame(
        [dict(start_pixel=0, end_pixel=0, fac_start=0, fac_end=0, length=5, downstream=-1, reach_id=0)]
    )
    fsp_df = pd.DataFrame({"SegId": [1, 1], "FspX": [0.0, 1.0], "FspY": [0.0, 0.0]})
    with pytest.raises(ValueError):
        add_downstream_distances(seg_info, fsp_df)


def _write_raw_segment(segdr, seg_id, fldmx, dh, entries):
    """entries: list of (fsp_pixel, fpp_pixel, dtf_m, fill_depth_m)."""
    df = pd.DataFrame(entries, columns=["fsp", "pixel", "dtf", "fill_depth"])
    basename = output_basename(fldmx, dh, seg_id)
    df.to_parquet(segdr / f"{basename}.parquet", index=False)


def test_build_segment_library_relations_groups_and_sorts(tmp_path):
    n_cols = 10
    fsp_df = pd.DataFrame(
        {
            "SegId": [1, 1],
            "FspId": [0, 1],
            "_pixel": [100, 200],
        }
    )
    # FSP 200's relations arrive out of DTF order and interleaved with FSP
    # 100's relations in the raw file -- the port must regroup by canonical
    # FspId order (100 first) and stable-sort each group by ascending Dtf.
    _write_raw_segment(
        tmp_path, seg_id=1, fldmx=5, dh=1,
        entries=[
            (200, 205, 2.0, 0.1),
            (100, 101, 3.0, 0.3),
            (200, 206, 1.0, 0.2),
            (100, 102, 1.0, 0.4),
        ],
    )

    rel = build_segment_library_relations(1, fsp_df, fldmx=5, dh=1, segdr=tmp_path, n_cols=n_cols)

    assert list(rel["FspId"]) == [0, 0, 1, 1]  # FSP 100(id=0) group first
    # within FSP 100's group, Dtf ascending: 1.0 (pixel 102) then 3.0 (pixel 101)
    fsp0 = rel[rel["FspId"] == 0]
    assert list(fsp0["Dtf"]) == sorted(fsp0["Dtf"])
    np.testing.assert_allclose(fsp0["Dtf"].to_numpy(), np.array([1.0, 3.0]) / METERS_PER_FOOT, rtol=1e-5)
    # FppRow/FppCol decoded correctly (pixel 102 -> row 10, col 2 for n_cols=10)
    row102 = rel[(rel["FspId"] == 0) & (rel["Dtf"] < 2 / METERS_PER_FOOT)].iloc[0]
    assert row102["FppRow"] == 102 // n_cols and row102["FppCol"] == 102 % n_cols


def test_build_segment_library_relations_rejects_fsp_mismatch(tmp_path):
    fsp_df = pd.DataFrame({"SegId": [1], "FspId": [0], "_pixel": [100]})
    _write_raw_segment(
        tmp_path, seg_id=1, fldmx=5, dh=1,
        entries=[(999, 101, 1.0, 0.1)],  # pixel 999 not one of segment 1's own FSPs
    )
    with pytest.raises(ValueError, match="FSP mismatch"):
        build_segment_library_relations(1, fsp_df, fldmx=5, dh=1, segdr=tmp_path, n_cols=10)
