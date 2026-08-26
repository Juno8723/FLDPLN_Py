"""Unit tests for the gauge-based flood mapping port (mapping/gauge_map.py),
including a regression test for the topological stream-order processing fix
(a real library can have StrOrd=12 -> DsStrOrd=14 -- ascending numeric order
would process the tributary before its own downstream mainstem order has an
interpolated DOF)."""

import numpy as np
import pandas as pd
import pytest
from affine import Affine

from PyFLDPLN_src.floodmap.gauge_map import (
    create_gauge_flood_map,
    estimate_fsp_dof,
    rasterize_tile,
    snap_gauges_to_fsps,
)
from PyFLDPLN_src.io.raster import RasterInfo


def _network():
    """mainstem (SegId=2, StrOrd=5, DsStrOrd=0) with a tributary (SegId=1,
    StrOrd=3, DsStrOrd=5) joining it -- StrOrd numbers deliberately chosen
    so DsStrOrd > StrOrd for the tributary, mirroring the real counter-
    example that broke plain ascending-order processing."""
    fsp_df = pd.DataFrame(
        [
            # tributary, SegId 1, StrOrd 3, upstream(FspId0)->downstream(FspId2)
            dict(FspId=0, FspX=0.0, FspY=0.0, SegId=1, FilledElev=102.0, DsDist=25.0, StrOrd=3),
            dict(FspId=1, FspX=1.0, FspY=0.0, SegId=1, FilledElev=101.0, DsDist=15.0, StrOrd=3),
            dict(FspId=2, FspX=2.0, FspY=0.0, SegId=1, FilledElev=100.5, DsDist=5.0, StrOrd=3),
            # mainstem, SegId 2, StrOrd 5, upstream(FspId3, the junction)->downstream(FspId5, outlet)
            dict(FspId=3, FspX=10.0, FspY=0.0, SegId=2, FilledElev=100.0, DsDist=4.0, StrOrd=5),
            dict(FspId=4, FspX=11.0, FspY=0.0, SegId=2, FilledElev=99.0, DsDist=2.0, StrOrd=5),
            dict(FspId=5, FspX=12.0, FspY=0.0, SegId=2, FilledElev=98.0, DsDist=0.0, StrOrd=5),
        ]
    )
    str_ord_df = pd.DataFrame(
        [
            dict(StrOrd=5, DsStrOrd=0, JunctionFspX=0.0, JunctionFspY=0.0),
            dict(StrOrd=3, DsStrOrd=5, JunctionFspX=10.0, JunctionFspY=0.0),  # FspId3
        ]
    )
    return fsp_df, str_ord_df


def test_estimate_fsp_dof_topological_order_and_junction_propagation():
    fsp_df, str_ord_df = _network()
    gauge_fsp_df = pd.DataFrame(
        [
            dict(FspId=4, FspX=11.0, FspY=0.0, stage_elevation=101.0),  # mainstem gauge
            dict(FspId=1, FspX=1.0, FspY=0.0, stage_elevation=103.0),  # tributary gauge
        ]
    )

    dof_df = estimate_fsp_dof(gauge_fsp_df, fsp_df, str_ord_df)
    dof = dict(zip(dof_df["FspId"], dof_df["Dof"]))

    assert dof.keys() == {0, 1, 2, 3, 4, 5}
    np.testing.assert_allclose(dof[3], 1.0)
    np.testing.assert_allclose(dof[4], 2.0)
    np.testing.assert_allclose(dof[5], 0.0)
    np.testing.assert_allclose(dof[1], 2.0)
    np.testing.assert_allclose(dof[0], 1.0)
    # interpolated between junction (DsDist=4, Dof=1.0) and the tributary
    # gauge (DsDist=15, Dof=2.0): frac=(5-4)/(15-4)
    np.testing.assert_allclose(dof[2], 1.0 + (5.0 - 4.0) / (15.0 - 4.0) * (2.0 - 1.0))


def test_estimate_fsp_dof_clamps_low_or_missing_dof_to_minimum():
    fsp_df, str_ord_df = _network()
    # a gauge whose stage is BELOW the FSP's own filled elevation -> Dof would be negative
    gauge_fsp_df = pd.DataFrame([dict(FspId=4, FspX=11.0, FspY=0.0, stage_elevation=50.0)])
    dof_df = estimate_fsp_dof(gauge_fsp_df, fsp_df, str_ord_df)
    gauged_row = dof_df[dof_df["FspId"] == 4].iloc[0]
    assert gauged_row["Dof"] == pytest.approx(0.0328084)


def test_snap_gauges_to_fsps_filters_by_distance():
    fsp_df = pd.DataFrame({"FspId": [0, 1], "FspX": [0.0, 100.0], "FspY": [0.0, 0.0]})
    gauges = pd.DataFrame({"gid": ["near", "far"], "x": [1.0, 500.0], "y": [0.0, 0.0]})
    snapped = snap_gauges_to_fsps(gauges, fsp_df, snap_dist=10.0)
    assert list(snapped["gid"]) == ["near"]
    assert snapped.iloc[0]["FspId"] == 0
    assert snapped.iloc[0]["d2NearestFsp"] == pytest.approx(1.0)


def test_rasterize_tile_max_across_fsps_plus_filled_depth():
    tile_df = pd.DataFrame(
        {
            "FspId": [0, 1, 2],
            "FppCol": [5, 5, 8],   # pixel (5,5)-ish shared by FSP 0 and 1
            "FppRow": [5, 5, 8],
            "Dtf": [1.0, 3.0, 10.0],
            "FilledDepth": [0.2, 0.2, 0.5],
        }
    )
    dof_df = pd.DataFrame({"FspId": [0, 1, 2], "Dof": [4.0, 4.0, 9.0]})
    # pixel (5,5): FSP0 -> Dtf=4-1=3.0, FSP1 -> Dtf=4-3=1.0 -> max=3.0, +FilledDepth(first=0.2)=3.2
    # pixel (8,8): FSP2 -> Dtf=9-10=-1.0 -> dropped entirely (not > 0)
    arr, row0, col0 = rasterize_tile(tile_df, dof_df)
    assert row0 == 5 and col0 == 5
    assert arr.shape == (1, 1)
    assert arr[0, 0] == pytest.approx(3.2)


def test_rasterize_tile_returns_none_when_nothing_floods():
    tile_df = pd.DataFrame({"FspId": [0], "FppCol": [0], "FppRow": [0], "Dtf": [10.0], "FilledDepth": [0.0]})
    dof_df = pd.DataFrame({"FspId": [0], "Dof": [1.0]})  # Dof < Dtf everywhere -> nothing floods
    assert rasterize_tile(tile_df, dof_df) is None


def test_create_gauge_flood_map_end_to_end(tmp_path):
    lib_dir = tmp_path / "library"
    lib_dir.mkdir()
    fsp_df, str_ord_df = _network()
    fsp_df.to_csv(lib_dir / "fsp_info.csv", index=False)
    str_ord_df.to_csv(lib_dir / "stream_order_info.csv", index=False)

    # one tile holding relations for FSP 4 (mainstem gauge FSP) and FSP 1
    # (tributary gauge FSP), each with one floodplain pixel.
    tile_df = pd.DataFrame(
        {
            "FspId": [4, 1],
            "FppCol": [0, 0],
            "FppRow": [0, 1],
            "Dtf": [0.5, 0.5],
            "FilledDepth": [0.0, 0.0],
        }
    )
    tile_df.to_parquet(lib_dir / "FLDPLN_tiled_1.snz", index=False)
    pd.DataFrame(
        [
            {"TileId": 1, "FspId": 4, "MinDtf": 0.0, "MaxDtf": 0.5},
            {"TileId": 1, "FspId": 1, "MinDtf": 0.0, "MaxDtf": 0.5},
        ]
    ).to_csv(lib_dir / "FLDPLN_tiled_fsp_index.csv", index=False)
    pd.DataFrame(
        [
            {
                "TileId": 1, "TileMinX": 0, "TileMaxX": 20, "TileMinY": -5, "TileMaxY": 5,
                "FppMinX": 10.5, "FppMaxX": 11.5, "FppMinY": -1.5, "FppMaxY": 0.5,
                "FspMinX": 0, "FspMaxX": 12, "FspMinY": -1, "FspMaxY": 1,
                "MinDtf": 0.5, "MaxDtf": 0.5, "NumOfRels": 2, "NumOfFpps": 2,
            }
        ]
    ).to_csv(lib_dir / "FLDPLN_tiled_tile_index.csv", index=False)

    gauges = pd.DataFrame(
        {"x": [11.0, 1.0], "y": [0.0, 0.0], "stage_elevation": [101.0, 103.0]}
    )
    # north-up transform: pixel (0,0)'s upper-left edge is exactly (10.5, 0.5)
    # (matches FppMinX/FppMaxY above), pixel size 1.
    transform = Affine(1, 0, 10.5, 0, -1, 0.5)
    raster_info = RasterInfo(rows=5, cols=5, pixel_size_x=1, pixel_size_y=1,
                              transform=transform, crs=None, nodata=-9999.0, dtype="float32")

    out_path = tmp_path / "flood_map.tif"
    canvas = create_gauge_flood_map(lib_dir, gauges, snap_dist=5.0, raster_info=raster_info, out_path=str(out_path))

    # FSP4's pixel (global row0,col0): Dof(FspId4)=2.0, Dtf=0.5 -> flood depth 1.5
    assert canvas[0, 0] == pytest.approx(1.5)
    # FSP1's pixel (local row1 -> global row1,col0): Dof(FspId1)=2.0, Dtf=0.5 -> 1.5
    assert canvas[1, 0] == pytest.approx(1.5)
    assert out_path.exists()
