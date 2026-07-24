"""Unit tests for the seedpoints segmentation module, reusing the synthetic
5x5 grid from conftest.py (main stem down column 2, tributary joining at
(2,2)).
"""

from affine import Affine
import numpy as np
import pytest
import rasterio

from PyFLDPLN_src.io.raster import RasterInfo
from PyFLDPLN_src.segmentation.seedpoints import (
    build_seedpoint_segments,
    geo_to_pixel,
    resolve_seed_paths,
    trace_seed_path,
)

N_ROWS, N_COLS = 5, 5


@pytest.fixture
def raster_info():
    # north-up, 1-unit pixels, upper-left corner at (0, 5)
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 5.0)
    return RasterInfo(
        rows=N_ROWS,
        cols=N_COLS,
        pixel_size_x=1.0,
        pixel_size_y=1.0,
        transform=transform,
        crs=rasterio.crs.CRS.from_epsg(4326),
        nodata=None,
        dtype="uint8",
    )


def test_geo_to_pixel_round_trip(raster_info):
    # pixel (row=2, col=3) centroid -> x = 3.5, y = 5 - 2.5 = 2.5
    row, col = geo_to_pixel(3.5, 2.5, raster_info)
    assert (row, col) == (2, 3)


def test_trace_seed_path_no_endpoint(synthetic_grid):
    fdr, fac = synthetic_grid
    rows, cols = trace_seed_path(0, 2, fdr, fac, N_ROWS, N_COLS)
    # unlike auto's trace_reach, seedpoints tracing does NOT stop at confluences
    assert list(zip(rows.tolist(), cols.tolist())) == [
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
    ]


def test_trace_seed_path_with_endpoint(synthetic_grid):
    fdr, fac = synthetic_grid
    rows, cols = trace_seed_path(0, 2, fdr, fac, N_ROWS, N_COLS, end_row=2, end_col=2)
    assert list(zip(rows.tolist(), cols.tolist())) == [(0, 2), (1, 2), (2, 2)]


def test_resolve_seed_paths_trims_lower_priority_at_confluence(synthetic_grid):
    fdr, fac = synthetic_grid
    main = trace_seed_path(0, 2, fdr, fac, N_ROWS, N_COLS)
    trib = trace_seed_path(2, 0, fdr, fac, N_ROWS, N_COLS)

    trimmed, conpts = resolve_seed_paths([main, trib], N_COLS)

    main_rows, main_cols = trimmed[0]
    assert list(zip(main_rows.tolist(), main_cols.tolist())) == [
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
    ]  # higher priority path is untouched

    trib_rows, trib_cols = trimmed[1]
    assert list(zip(trib_rows.tolist(), trib_cols.tolist())) == [(2, 0), (2, 1)]

    assert conpts == {1: 2 * N_COLS + 2}  # confluence at (2,2)


def test_build_seedpoint_segments_raises_when_confluence_not_a_natural_break(
    synthetic_grid, raster_info
):
    fdr, fac = synthetic_grid
    # facthr/seglen large -> main stem becomes a single segment, so (2,2) is
    # NOT the start of any subsegment -> the original's unimplemented gap.
    points_xy = np.array(
        [
            [2.5, 4.5],  # main stem seed -> pixel (0,2)
            [0.5, 2.5],  # tributary seed -> pixel (2,0)
        ]
    )
    with pytest.raises(NotImplementedError):
        build_seedpoint_segments(points_xy, fdr, fac, raster_info, facthr=1000, seglen=1000)


def test_build_seedpoint_segments_links_when_confluence_is_a_natural_break(
    synthetic_grid, raster_info
):
    fdr, fac = synthetic_grid
    # facthr=10 forces a natural break exactly between (1,2) and (2,2),
    # since fac jumps from 8 to 20 there (>= 10) -- so the confluence pixel
    # (2,2) becomes the start of its own subsegment, and linking succeeds.
    points_xy = np.array([[2.5, 4.5], [0.5, 2.5]])
    seg_table = build_seedpoint_segments(
        points_xy, fdr, fac, raster_info, facthr=10, seglen=1000
    )

    def px(row, col):
        return row * N_COLS + col

    tributary_row = seg_table.index[seg_table["start_pixel"] == px(2, 0)][0]
    confluence_row = seg_table.index[seg_table["start_pixel"] == px(2, 2)][0]
    assert seg_table.at[tributary_row, "downstream"] == confluence_row
