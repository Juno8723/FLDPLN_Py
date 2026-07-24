"""Unit tests for the segmentation module against a small, hand-designed
synthetic grid (see module docstring in each test for the layout).

Grid (5x5, 0-based row/col). Stream network:
  main stem:  (0,2) -> (1,2) -> (2,2)[confluence] -> (3,2) -> (4,2)[exits image]
  tributary:  (2,0) -> (2,1) -> joins (2,2)

FDR codes (ESRI D8): South=4, East=1. All non-stream pixels are filler
(FAC below strthr, arbitrary valid FDR) and should not affect the result.
"""

from affine import Affine
import numpy as np
import rasterio

from PyFLDPLN_src.io.raster import RasterInfo, read_raster
from PyFLDPLN_src.segmentation import d8
from PyFLDPLN_src.segmentation.auto import (
    build_auto_segments,
    classify_headwater_confluence,
    find_stream_pixels,
    trace_reach,
)
from PyFLDPLN_src.segmentation.network_core import (
    ReachPath,
    build_segment_table,
    subdivide_reach,
    write_segment_raster,
)

N_ROWS, N_COLS = 5, 5
SOUTH, EAST = 4, 1


def test_d8_step_downstream_matches_encoding():
    # South from (1,1) -> (2,1); East from (1,1) -> (1,2)
    assert d8.step_downstream(1, 1, SOUTH) == (2, 1)
    assert d8.step_downstream(1, 1, EAST) == (1, 2)
    assert d8.step_downstream(1, 1, 0) is None  # invalid code


def test_find_stream_pixels(synthetic_grid):
    fdr, fac = synthetic_grid
    mask = find_stream_pixels(fac, strthr=5, nodata=None)
    expected = {(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (2, 0), (2, 1)}
    found = set(map(tuple, np.argwhere(mask)))
    assert found == expected


def test_classify_headwater_confluence(synthetic_grid):
    fdr, fac = synthetic_grid
    stream_mask = find_stream_pixels(fac, strthr=5, nodata=None)
    headwater, confluence = classify_headwater_confluence(stream_mask, fdr)

    assert set(map(tuple, np.argwhere(headwater))) == {(0, 2), (2, 0)}
    assert set(map(tuple, np.argwhere(confluence))) == {(2, 2)}


def test_trace_reach_stops_before_confluence(synthetic_grid):
    fdr, fac = synthetic_grid
    stream_mask = find_stream_pixels(fac, strthr=5, nodata=None)
    _, confluence = classify_headwater_confluence(stream_mask, fdr)

    rows, cols = trace_reach(0, 2, fdr, confluence, N_ROWS, N_COLS)
    assert list(zip(rows.tolist(), cols.tolist())) == [(0, 2), (1, 2)]

    rows, cols = trace_reach(2, 0, fdr, confluence, N_ROWS, N_COLS)
    assert list(zip(rows.tolist(), cols.tolist())) == [(2, 0), (2, 1)]


def test_trace_reach_from_confluence_stops_at_image_edge(synthetic_grid):
    fdr, fac = synthetic_grid
    stream_mask = find_stream_pixels(fac, strthr=5, nodata=None)
    _, confluence = classify_headwater_confluence(stream_mask, fdr)

    rows, cols = trace_reach(2, 2, fdr, confluence, N_ROWS, N_COLS)
    assert list(zip(rows.tolist(), cols.tolist())) == [(2, 2), (3, 2), (4, 2)]


def test_subdivide_reach_no_breaks_when_thresholds_are_large():
    pixels = np.array([2, 7, 12])
    fac = np.array([6.0, 8.0, 20.0])
    segments = subdivide_reach(pixels, fac, facthr=1000, seglen=1000)
    assert segments == [(0, 2)]


def test_subdivide_reach_splits_on_fac_jump():
    pixels = np.array([0, 1, 2, 3])
    fac = np.array([1.0, 2.0, 50.0, 51.0])  # big jump between idx 1 and 2
    segments = subdivide_reach(pixels, fac, facthr=10, seglen=1000)
    assert segments == [(0, 1), (2, 3)]


def test_subdivide_reach_splits_on_max_length():
    pixels = np.arange(6)
    fac = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])  # max jump between idx 2,3
    segments = subdivide_reach(pixels, fac, facthr=1000, seglen=3)
    # forced to split since length 6 > seglen 3; splits at largest internal jump
    assert segments == [(0, 2), (3, 5)]
    for start, end in segments:
        assert end - start + 1 <= 3


def test_build_auto_segments_end_to_end(synthetic_grid):
    fdr, fac = synthetic_grid
    seg_table = build_auto_segments(
        fac, fdr, fac_nodata=None, strthr=5, facthr=1000, seglen=1000
    )

    assert len(seg_table) == 3

    def px(row, col):
        return row * N_COLS + col

    by_start = {int(r.start_pixel): r for r in seg_table.itertuples()}

    main_upper = by_start[px(0, 2)]
    assert main_upper.end_pixel == px(1, 2)
    assert main_upper.length == 2

    tributary = by_start[px(2, 0)]
    assert tributary.end_pixel == px(2, 1)
    assert tributary.length == 2

    main_lower = by_start[px(2, 2)]
    assert main_lower.end_pixel == px(4, 2)
    assert main_lower.length == 3
    assert main_lower.downstream == -1  # exits image, no downstream segment

    # both upstream segments should link downstream into the confluence segment
    assert seg_table.loc[seg_table["end_pixel"] == px(1, 2), "downstream"].iloc[0] == (
        seg_table.index[seg_table["start_pixel"] == px(2, 2)][0]
    )
    assert seg_table.loc[seg_table["end_pixel"] == px(2, 1), "downstream"].iloc[0] == (
        seg_table.index[seg_table["start_pixel"] == px(2, 2)][0]
    )


def test_write_segment_raster_round_trip(synthetic_grid, tmp_path):
    fdr, fac = synthetic_grid
    seg_table = build_auto_segments(
        fac, fdr, fac_nodata=None, strthr=5, facthr=1000, seglen=1000
    )

    ref = RasterInfo(
        rows=N_ROWS,
        cols=N_COLS,
        pixel_size_x=1.0,
        pixel_size_y=1.0,
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
        crs=rasterio.crs.CRS.from_epsg(4326),
        nodata=0,
        dtype="uint8",
    )
    out_path = tmp_path / "str_segid.bil"
    write_segment_raster(str(out_path), seg_table, N_ROWS, N_COLS, ref)

    written, info = read_raster(out_path)
    assert written.shape == (N_ROWS, N_COLS)

    def px(row, col):
        return row * N_COLS + col

    # every stream pixel should carry a nonzero segment id (1-based)
    for (row, col) in [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (2, 0), (2, 1)]:
        assert written[row, col] != 0

    # non-stream pixels stay background (0)
    assert written[0, 0] == 0

    # pixels within the same segment share the same id
    assert written[0, 2] == written[1, 2]
    assert written[2, 0] == written[2, 1]
    assert written[2, 2] == written[3, 2] == written[4, 2]
    assert written[0, 2] != written[2, 2]
