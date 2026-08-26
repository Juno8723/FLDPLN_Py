"""Unit tests for the tiling port (TileLibrary), using a small synthetic
2-segment relation set and an identity affine transform for easy
hand-computed expected map coordinates."""

import numpy as np
import pandas as pd
from affine import Affine

from PyFLDPLN_src.library_reorganization.tiling import (
    build_tiles,
    calculate_tile_grid,
    compute_relations_pixel_extents,
)


def _write_relations(path, entries):
    """entries: list of (FspId, FppCol, FppRow, Dtf, FilledDepth)."""
    df = pd.DataFrame(entries, columns=["FspId", "FppCol", "FppRow", "Dtf", "FilledDepth"])
    df.to_parquet(path, index=False)


def test_calculate_tile_grid_padding_matches_original_convention():
    # min_row=0,max_row=4 (inclusive), tile_size=3 -> tiles [0,3) and [3,6)
    # (padded past the true max_row=4, same as CalculateTileBoundary).
    grid_rows = {(r0, r1) for (r0, r1, c0, c1) in calculate_tile_grid(0, 4, 0, 0, 3)}
    assert grid_rows == {(0, 3), (3, 6)}


def test_build_tiles_partitions_relations_and_localizes_coords(tmp_path):
    # FSPs 0 and 1, both segment 1; a fake segment 2 whose relations fall in
    # a different tile entirely.
    seg1_path = tmp_path / "seg1.parquet"
    seg2_path = tmp_path / "seg2.parquet"
    _write_relations(
        seg1_path,
        [
            (0, 1, 1, 1.0, 0.1),   # tile A (rows/cols 0-2)
            (0, 2, 0, 3.0, 0.2),   # tile A
            (1, 0, 2, 2.0, 0.3),   # tile A
        ],
    )
    _write_relations(
        seg2_path,
        [
            (1, 5, 5, 4.0, 0.4),   # tile B (rows/cols 3-5, since tile size 3)
        ],
    )
    seg_extents = compute_relations_pixel_extents({1: seg1_path, 2: seg2_path})

    fsp_df = pd.DataFrame({"FspId": [0, 1], "FspX": [10.0, 20.0], "FspY": [-10.0, -20.0]})
    # North-up convention (pixel size 1, origin at (0,0)) -- e must be
    # negative like a real GDAL geotransform, unlike Affine.identity()'s
    # e=+1, or the tile-boundary min/max-Y bookkeeping below would invert.
    transform = Affine(1, 0, 0, 0, -1, 0)
    out_dir = tmp_path / "tiles"

    build_tiles(seg_extents, fsp_df, tile_size_px=3, pixel_size=1.0, transform=transform, out_dir=out_dir)

    tile_index = pd.read_csv(out_dir / "FLDPLN_tiled_tile_index.csv")
    fsp_index = pd.read_csv(out_dir / "FLDPLN_tiled_fsp_index.csv")

    assert len(tile_index) == 2  # tile A (3 relations) + tile B (1 relation)
    assert set(tile_index["NumOfRels"]) == {3, 1}

    tile_a_id = int(tile_index.loc[tile_index["NumOfRels"] == 3, "TileId"].iloc[0])
    tile_a = pd.read_parquet(out_dir / f"FLDPLN_tiled_{tile_a_id}.snz")
    assert len(tile_a) == 3
    # local origin = (min row=0, min col=0) among tile A's own relations
    # -> FppCol/FppRow are unchanged from global in this case (already 0-based
    # from that local min).
    assert set(zip(tile_a["FppCol"], tile_a["FppRow"])) == {(1, 1), (2, 0), (0, 2)}

    # FSP 1 appears in both tiles -> two fsp_index rows, one per TileId.
    assert len(fsp_index[fsp_index["FspId"] == 1]) == 2
    # FSP 0 only in tile A, MinDtf/MaxDtf across its 2 relations there.
    fsp0_row = fsp_index[(fsp_index["FspId"] == 0)].iloc[0]
    assert fsp0_row["MinDtf"] == 1.0 and fsp0_row["MaxDtf"] == 3.0

    # Tile boundary in map coords: tile A covers pixel rows/cols [0,3), and
    # with this north-up transform (x=col, y=-row), that's X in [0,3],
    # Y in [-3,0].
    tile_a_row = tile_index[tile_index["TileId"] == tile_a_id].iloc[0]
    assert (tile_a_row["TileMinX"], tile_a_row["TileMaxX"]) == (0.0, 3.0)
    assert (tile_a_row["TileMinY"], tile_a_row["TileMaxY"]) == (-3.0, 0.0)
