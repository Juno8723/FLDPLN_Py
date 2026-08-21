"""Unit tests for the automated (level-path) stream-order assignment,
against the same synthetic confluence grid used by test_segmentation.py:
a mainstem tributary (seg 1, FAC 6->8) and a higher-FAC tributary (seg 2,
FAC 6->10) both flow into a shared downstream segment (seg 3, the network
outlet)."""

import numpy as np
import pandas as pd

from PyFLDPLN_src.segmentation.auto import build_auto_segments
from PyFLDPLN_src.segmentation.stream_order import (
    assign_stream_order,
    build_stream_order_info,
)


def _confluence_seg_info(synthetic_grid) -> pd.DataFrame:
    fdr, fac = synthetic_grid
    return build_auto_segments(fac, fdr, None, strthr=5, facthr=999, seglen=999)


def test_assign_stream_order_picks_higher_fac_branch_as_mainstem(synthetic_grid):
    seg_info = _confluence_seg_info(synthetic_grid)
    # row 0 = seg 1 (low-FAC branch, fac_end=8), row 1 = seg 2 (high-FAC
    # branch, fac_end=10), row 2 = seg 3 (outlet, both flow into it).
    assert seg_info.loc[0, "fac_end"] == 8.0
    assert seg_info.loc[1, "fac_end"] == 10.0
    assert seg_info.loc[2, "downstream"] == -1

    order = assign_stream_order(seg_info)

    # Outlet and its higher-FAC upstream branch (seg 2) must share the same
    # (lowest) order number -- the level-path "main stem" rule.
    assert order[2] == order[1]
    # The lower-FAC branch (seg 1) is demoted to a distinct, higher number.
    assert order[0] != order[2]
    assert order.min() == 1
    assert len(set(order.tolist())) == 2  # exactly two paths in this network


def test_assign_stream_order_globally_unique_across_multiple_outlets():
    # Two fully independent watersheds (no shared segments) -- rule 2
    # (global uniqueness) must hold even though neither has any relation to
    # the other's FAC scale.
    seg_info = pd.DataFrame(
        [
            dict(start_pixel=0, end_pixel=0, fac_start=0, fac_end=100, length=1, downstream=-1, reach_id=0),
            dict(start_pixel=0, end_pixel=0, fac_start=0, fac_end=80, length=1, downstream=0, reach_id=0),
            dict(start_pixel=0, end_pixel=0, fac_start=0, fac_end=20, length=1, downstream=0, reach_id=0),
            dict(start_pixel=0, end_pixel=0, fac_start=0, fac_end=50, length=1, downstream=-1, reach_id=0),
        ]
    )
    order = assign_stream_order(seg_info)
    roots = [0, 3]
    assert len(set(order[roots].tolist())) == len(roots)  # each outlet distinct
    assert len(set(order.tolist())) == 3  # mainstem(0,1) + tributary(2) + watershed-B(3)
    assert order[0] == order[1]  # seg 0's mainstem continuation
    assert order[2] not in (order[0], order[3])


def test_assign_stream_order_rejects_cycle():
    seg_info = pd.DataFrame(
        [
            dict(start_pixel=0, end_pixel=0, fac_start=0, fac_end=1, length=1, downstream=1, reach_id=0),
            dict(start_pixel=0, end_pixel=0, fac_start=0, fac_end=1, length=1, downstream=0, reach_id=0),
        ]
    )
    import pytest

    with pytest.raises(ValueError):
        assign_stream_order(seg_info)


def test_build_stream_order_info_junction_and_outlet_sentinel(synthetic_grid):
    seg_info = _confluence_seg_info(synthetic_grid)
    order = assign_stream_order(seg_info)
    info = build_stream_order_info(seg_info, order)

    outlet_order = order[2]
    trib_order = order[0]
    outlet_row = info[info["StrOrd"] == outlet_order].iloc[0]
    assert outlet_row["DsStrOrd"] == 0
    assert outlet_row["JunctionFspX"] == 0.0 and outlet_row["JunctionFspY"] == 0.0

    trib_row = info[info["StrOrd"] == trib_order].iloc[0]
    assert trib_row["DsStrOrd"] == outlet_order
