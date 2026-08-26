"""Direct (structure-preserving) port of the existing (not-ours) Tools/
fldpln_reorg.py's `TileLibrary`/`CalculateLibraryExtent`/
`CalculateTileBoundary`: buckets per-segment FSP-FPP relations into
fixed-size geographic tiles for fast lookup, matching the original's output
schema exactly (fspIdxColumnNames / tileIdxColumnNames /
`FLDPLN_tiled_<id>.snz`, all from fldpln_header.py).

Deliberately dropped, per user instruction ("동일한 구조와 결과, 근데 이전
연구자의 서버 연결/멀티 라이브러리 기능은 필요 없음" -- same structure/
results, but the previous researcher's server-download and multi-library
bookkeeping aren't needed by anyone): the multi-libName download/zip
machinery, the .mat/Excel round-trip, and the fragile float-equality
(FspX,FspY) join used to recover FspId (fldpln_reorg.py:165) -- our
per-segment relations already carry the real FspId directly (see
library_reorganization/library_files.py), so that join is structurally unnecessary here,
not just skipped for convenience.

Implemented directly in pixel-index space rather than the original's map-
coordinate arithmetic (tile membership test, local row/col within a tile,
segment/tile-extent overlap) -- mathematically equivalent for an axis-
aligned raster (tile membership commutes with the raster's affine
transform), but avoids the float-rounding (`.round()`) and float-equality-
join fragility the original's approach was documented to require
`float_precision='round_trip'` workarounds for. Geographic extents in the
final tile_index.csv are still produced in real map coordinates (via the
raster's affine transform), matching the original's output exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compute_relations_pixel_extents(relations_paths: dict[int, Path]) -> pd.DataFrame:
    """Per-segment (seg_id -> relations Parquet path) min/max FppRow/FppCol,
    read cheaply (2 columns only) -- used to skip segments that can't
    possibly overlap a given tile, replicating CalculateLibraryExtent's
    per-segment extent table (fldpln_reorg.py:288-330), in pixel space
    instead of map coordinates.
    """
    rows = []
    for seg_id, path in relations_paths.items():
        df = pd.read_parquet(path, columns=["FppRow", "FppCol"])
        rows.append(
            {
                "seg_id": seg_id,
                "path": path,
                "min_row": int(df["FppRow"].min()),
                "max_row": int(df["FppRow"].max()),
                "min_col": int(df["FppCol"].min()),
                "max_col": int(df["FppCol"].max()),
            }
        )
    return pd.DataFrame(rows)


def calculate_tile_grid(
    min_row: int, max_row: int, min_col: int, max_col: int, tile_size_px: int
) -> list[tuple[int, int, int, int]]:
    """Uniform grid of half-open pixel-space tile bounds (row0, row1, col0,
    col1), replicating CalculateTileBoundary's padding=True convention
    (fldpln_reorg.py:263-283): every tile is exactly tile_size_px, even the
    last one in each dimension, which may extend past the true data extent.
    """

    def bounds_1d(lo: int, hi_inclusive: int, size: int) -> list[tuple[int, int]]:
        starts = list(range(lo, hi_inclusive + 1, size))
        return [(s, s + size) for s in starts]

    row_ranges = bounds_1d(min_row, max_row, tile_size_px)
    col_ranges = bounds_1d(min_col, max_col, tile_size_px)
    return [(r0, r1, c0, c1) for (r0, r1) in row_ranges for (c0, c1) in col_ranges]


def build_tiles(
    seg_extents: pd.DataFrame,
    fsp_df: pd.DataFrame,
    tile_size_px: int,
    pixel_size: float,
    transform,
    out_dir: Path,
) -> None:
    """Bucket every segment's relations into fixed-size tiles and write
    `FLDPLN_tiled_<id>.snz` + `FLDPLN_tiled_fsp_index.csv` +
    `FLDPLN_tiled_tile_index.csv` under out_dir, matching
    fldpln_header.py's schemas exactly. Empty tiles are skipped (tile IDs
    are dense over non-empty tiles only, same as the original).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    half_cell = pixel_size / 2.0

    min_row = int(seg_extents["min_row"].min())
    max_row = int(seg_extents["max_row"].max())
    min_col = int(seg_extents["min_col"].min())
    max_col = int(seg_extents["max_col"].max())
    grid = calculate_tile_grid(min_row, max_row, min_col, max_col, tile_size_px)
    print(f"library pixel extent: rows[{min_row},{max_row}] cols[{min_col},{max_col}], "
          f"{len(grid)} possible tile(s) of {tile_size_px}px")

    fsp_xy = fsp_df.set_index("FspId")[["FspX", "FspY"]]

    fsp_idx_frames: list[pd.DataFrame] = []
    tile_idx_rows: list[dict] = []
    tile_id = 1

    for (r0, r1, c0, c1) in grid:
        overlap = seg_extents[
            ~(
                (seg_extents["min_row"] >= r1)
                | (seg_extents["max_row"] < r0)
                | (seg_extents["min_col"] >= c1)
                | (seg_extents["max_col"] < c0)
            )
        ]
        if overlap.empty:
            continue

        parts = []
        for _, seg in overlap.iterrows():
            df = pd.read_parquet(seg["path"])
            mask = (
                (df["FppRow"] >= r0)
                & (df["FppRow"] < r1)
                & (df["FppCol"] >= c0)
                & (df["FppCol"] < c1)
            )
            if mask.any():
                parts.append(df[mask])
        if not parts:
            continue
        tdf = pd.concat(parts, ignore_index=True)

        local_row0 = int(tdf["FppRow"].min())
        local_col0 = int(tdf["FppCol"].min())
        out = pd.DataFrame(
            {
                "FspId": tdf["FspId"].to_numpy(dtype=np.int32),
                "FppCol": (tdf["FppCol"].to_numpy(dtype=np.int64) - local_col0).astype(np.int32),
                "FppRow": (tdf["FppRow"].to_numpy(dtype=np.int64) - local_row0).astype(np.int32),
                "Dtf": tdf["Dtf"].to_numpy(dtype=np.float32),
                "FilledDepth": tdf["FilledDepth"].to_numpy(dtype=np.float32),
            }
        )
        out.to_parquet(out_dir / f"FLDPLN_tiled_{tile_id}.snz", index=False, compression="snappy")

        # Tile boundary in real map coordinates (pixel-edge/"external
        # border" convention, matching the original's units -- transform *
        # (col,row) gives a pixel's upper-left edge, not its centroid).
        tminX, tmaxY = transform * (c0, r0)
        tmaxX, tminY = transform * (c1, r1)

        # FPP data extent: centroid min/max of the actual pixels present,
        # expanded by half a cell to "external border" (matches
        # fldpln_reorg.py:220-221).
        fpp_rows_global = tdf["FppRow"].to_numpy()
        fpp_cols_global = tdf["FppCol"].to_numpy()
        cx0, cy0 = transform * (fpp_cols_global.min() + 0.5, fpp_rows_global.min() + 0.5)
        cx1, cy1 = transform * (fpp_cols_global.max() + 0.5, fpp_rows_global.max() + 0.5)
        fppMinX, fppMaxX = min(cx0, cx1) - half_cell, max(cx0, cx1) + half_cell
        fppMinY, fppMaxY = min(cy0, cy1) - half_cell, max(cy0, cy1) + half_cell

        fsp_pts = fsp_xy.loc[tdf["FspId"].unique()]
        fspMinX = fsp_pts["FspX"].min() - half_cell
        fspMaxX = fsp_pts["FspX"].max() + half_cell
        fspMinY = fsp_pts["FspY"].min() - half_cell
        fspMaxY = fsp_pts["FspY"].max() + half_cell

        agg = out.groupby("FspId", as_index=False).agg(MinDtf=("Dtf", "min"), MaxDtf=("Dtf", "max"))
        agg.insert(0, "TileId", tile_id)
        fsp_idx_frames.append(agg[["TileId", "FspId", "MinDtf", "MaxDtf"]])

        tile_idx_rows.append(
            {
                "TileId": tile_id,
                "TileMinX": tminX, "TileMaxX": tmaxX, "TileMinY": tminY, "TileMaxY": tmaxY,
                "FppMinX": fppMinX, "FppMaxX": fppMaxX, "FppMinY": fppMinY, "FppMaxY": fppMaxY,
                "FspMinX": fspMinX, "FspMaxX": fspMaxX, "FspMinY": fspMinY, "FspMaxY": fspMaxY,
                "MinDtf": float(agg["MinDtf"].min()), "MaxDtf": float(agg["MaxDtf"].max()),
                "NumOfRels": len(out),
                "NumOfFpps": len(out[["FppCol", "FppRow"]].drop_duplicates()),
            }
        )
        print(f"  tile {tile_id}: {len(out)} relations, {len(agg)} FSPs")
        tile_id += 1

    fsp_index = pd.concat(fsp_idx_frames, ignore_index=True) if fsp_idx_frames else pd.DataFrame(
        columns=["TileId", "FspId", "MinDtf", "MaxDtf"]
    )
    tile_index = pd.DataFrame(tile_idx_rows)
    fsp_index.to_csv(out_dir / "FLDPLN_tiled_fsp_index.csv", index=False)
    tile_index.to_csv(out_dir / "FLDPLN_tiled_tile_index.csv", index=False)
    print(f"{tile_id - 1} non-empty tile(s) written to {out_dir}")
