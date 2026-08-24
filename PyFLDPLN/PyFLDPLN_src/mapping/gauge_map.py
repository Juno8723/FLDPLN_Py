"""Direct (structure-preserving) port of the existing (not-ours) gauge-based
event flood mapping stage: `Tools/fldpln_gauge.py` + `Tools/fldpln.py`'s
`SnapGauges2Fsps` / `EstimateFspDofFromGaugeNew` / `TileFspFppRelations2Array`
/ `MapFloodDepthWithTiles`, targeting the same output structure (a single
flood-depth GeoTIFF) as the reference `FloodMap_h15dh1_Amite.tif`.

Deliberately dropped, per user instruction (same structure/results, but the
previous researcher's server-connection features aren't needed by anyone):
all of `fldpln_gauge.py`'s live USGS/AHPS/Postgres gauge-fetching machinery
(none of it was ever exercised by the actual Aug-2016 Amite run anyway --
that run read static gauge stages from an Excel file, see
`FIM_Amite_Aug2016.ipynb` cell 10), Azure Blob Storage variants, and the
multi-library-name bookkeeping. The gauge input here is just a plain
DataFrame (x, y, stage_elevation, ...) -- read it from wherever (CSV, Excel,
whatever) before calling into this module.

Kept identical: the DOF-interpolation algorithm (level-pool upstream
boundary, junction/zero downstream boundary, per-stream-order linear
interpolation on downstream distance) and the rasterization rule
(Dtf = Dof(FSP) - Dtf(FSP,FPP), max across FSPs per pixel, + FilledDepth).
One correctness fix carried over from the legacy-file bugfix pass (see
[[project_library_reorg_and_gauge_mapping]]): stream orders are processed in
true topological (downstream-first) order, not plain ascending StrOrd
(which a real library can violate, e.g. StrOrd=12 -> DsStrOrd=14).

Also simplified relative to the original: our relation files already carry
each FSP's real FspId (see library/library_files.py), so rasterization joins
on FspId directly (matches what the original ALSO already did in
TileFspFppRelations2Array -- only the DOF-interpolation stage's gauge/
junction lookups used the more fragile FspX/FspY float-equality join, which
is preserved here since it's only ever used in-memory within one run, never
round-tripped through a lossy format).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..io.raster import RasterInfo, write_raster

MIN_GAUGE_DOF_FT = 0.0328084  # 1 cm, matches the original's default
NODATA = -9999.0


def snap_gauges_to_fsps(
    gauges: pd.DataFrame,
    fsp_df: pd.DataFrame,
    snap_dist: float,
    x_field: str = "x",
    y_field: str = "y",
) -> pd.DataFrame:
    """Snap each gauge to its nearest FSP (Euclidean, via a KD-tree -- same
    nearest-point semantics as the original's brute-force
    `NearestPoint`/`SnapGauges2Fsps`, just not O(gauges x FSPs)), keeping
    only gauges within `snap_dist` of their nearest FSP.

    `snap_dist` is in the FSP table's own native coordinate units (e.g.
    meters for a UTM library, degrees for a geographic one) -- unlike the
    original's hardcoded `snapDist=350`, which silently meant nothing for
    the real Amite reference library (geographic-degree coordinates, so 350
    "degrees" filtered nothing at all). Pass a real, unit-aware threshold.
    """
    fsp_xy = fsp_df[["FspX", "FspY"]].to_numpy()
    tree = cKDTree(fsp_xy)
    gauge_xy = gauges[[x_field, y_field]].to_numpy()
    dist, idx = tree.query(gauge_xy)

    out = gauges.reset_index(drop=True).copy()
    out["FspId"] = fsp_df["FspId"].to_numpy()[idx]
    out["FspX"] = fsp_xy[idx, 0]
    out["FspY"] = fsp_xy[idx, 1]
    out["d2NearestFsp"] = dist
    return out[out["d2NearestFsp"] <= snap_dist].reset_index(drop=True)


def _topological_stream_order(str_ords: list, str_ord_df: pd.DataFrame) -> list:
    """Process each gauge-bearing stream order only after its own DsStrOrd
    (per stream_order_info.csv) has already been processed, instead of
    assuming ascending StrOrd == downstream-first (see module docstring)."""
    str_ord_set = set(str_ords)
    ds_of = dict(zip(str_ord_df["StrOrd"], str_ord_df["DsStrOrd"]))

    ordered: list = []
    processed: set = set()
    remaining = list(str_ords)
    while remaining:
        progressed = False
        for so in list(remaining):
            ds = ds_of.get(so, 0)
            if ds == 0 or ds not in str_ord_set or ds in processed:
                ordered.append(so)
                processed.add(so)
                remaining.remove(so)
                progressed = True
        if not progressed:
            ordered.extend(sorted(remaining))  # unresolved cycle -- shouldn't happen
            break
    return ordered


def estimate_fsp_dof(
    gauge_fsp_df: pd.DataFrame,
    fsp_df: pd.DataFrame,
    str_ord_df: pd.DataFrame,
    min_gauge_dof: float = MIN_GAUGE_DOF_FT,
) -> pd.DataFrame:
    """Port of EstimateFspDofFromGaugeNew: interpolate FSP depth-of-flow
    (stage) from gauge readings, per stream order, with level-pool upstream
    and junction/zero downstream virtual boundary conditions. Returns
    [FspId, Dof] for every FSP reachable from a gauge.

    gauge_fsp_df: output of snap_gauges_to_fsps, needs a 'stage_elevation'
    column (feet, same vertical datum as fsp_df's FilledElev).
    fsp_df: fsp_info.csv contents (FspId, FspX, FspY, SegId, FilledElev,
    DsDist, StrOrd).
    str_ord_df: stream_order_info.csv contents (StrOrd, DsStrOrd,
    JunctionFspX, JunctionFspY).
    """
    g = gauge_fsp_df.merge(
        fsp_df[["FspId", "StrOrd", "DsDist", "SegId", "FilledElev"]], on="FspId", how="inner"
    )
    g["Dof"] = g["stage_elevation"] - g["FilledElev"]
    g.loc[(g["Dof"] <= min_gauge_dof) | g["Dof"].isna(), "Dof"] = min_gauge_dof

    str_ords = _topological_stream_order(sorted(g["StrOrd"].drop_duplicates().tolist()), str_ord_df)
    if not str_ords:
        return pd.DataFrame(columns=["FspId", "Dof"])

    fsp_dof_frames: list[pd.DataFrame] = []
    # Accumulator carrying FspX/FspY alongside Dof, needed for the
    # downstream junction lookup (matches the original's `fspDof` table).
    accum = pd.DataFrame(columns=["FspId", "FspX", "FspY", "DsDist", "Dof"])

    for order in str_ords:
        gauge_ord = g[g["StrOrd"] == order].sort_values("DsDist")

        # --- upstream virtual boundary (level-pool from the most-upstream real gauge) ---
        up = gauge_ord.iloc[-1]
        up_seg_fsps = fsp_df[fsp_df["SegId"] == up["SegId"]].copy()
        up_seg_fsps["Dof"] = up["FilledElev"] + up["Dof"] - up_seg_fsps["FilledElev"]
        candidate = up_seg_fsps[up_seg_fsps["Dof"] >= 0].sort_values("DsDist").iloc[-1]
        us_end = None
        if not (candidate["FspX"] == up["FspX"] and candidate["FspY"] == up["FspY"]):
            us_end = pd.DataFrame(
                [{"FspX": candidate["FspX"], "FspY": candidate["FspY"],
                  "DsDist": candidate["DsDist"], "Dof": candidate["Dof"]}]
            )

        # --- downstream virtual boundary (junction Dof if already known, else Dof=0) ---
        so_row = str_ord_df[str_ord_df["StrOrd"] == order].iloc[0]
        ds_str_ord = int(so_row["DsStrOrd"])
        junc_df = pd.DataFrame()
        if ds_str_ord != 0:
            fspx, fspy = so_row["JunctionFspX"], so_row["JunctionFspY"]
            junc_df = accum[(accum["FspX"] == fspx) & (accum["FspY"] == fspy)]

        if ds_str_ord == 0 or junc_df.empty:
            ds_seg_id, gx, gy = gauge_ord.iloc[0][["SegId", "FspX", "FspY"]]
            last_fsp = fsp_df[fsp_df["SegId"] == ds_seg_id].iloc[-1]
            ds_end = None
            if not (last_fsp["FspX"] == gx and last_fsp["FspY"] == gy):
                ds_end = pd.DataFrame(
                    [{"FspX": last_fsp["FspX"], "FspY": last_fsp["FspY"],
                      "DsDist": last_fsp["DsDist"], "Dof": 0.0}]
                )
        else:
            ds_end = junc_df[["FspX", "FspY", "DsDist", "Dof"]]

        parts = [gauge_ord[["FspX", "FspY", "DsDist", "Dof"]]]
        if us_end is not None:
            parts.append(us_end)
        if ds_end is not None:
            parts.append(ds_end)
        gauge_ord_full = pd.concat(parts, ignore_index=True).sort_values("DsDist")

        min_dist, max_dist = gauge_ord_full["DsDist"].min(), gauge_ord_full["DsDist"].max()
        fsp_ord = fsp_df[fsp_df["StrOrd"] == order][["FspId", "FspX", "FspY", "DsDist"]]
        fsp_ord = fsp_ord[(fsp_ord["DsDist"] >= min_dist) & (fsp_ord["DsDist"] <= max_dist)].copy()
        fsp_ord["Dof"] = np.interp(
            fsp_ord["DsDist"], gauge_ord_full["DsDist"], gauge_ord_full["Dof"].astype(np.float64)
        )

        fsp_dof_frames.append(fsp_ord[["FspId", "Dof"]])
        accum = pd.concat([accum, fsp_ord[["FspId", "FspX", "FspY", "DsDist", "Dof"]]], ignore_index=True)

    return pd.concat(fsp_dof_frames, ignore_index=True) if fsp_dof_frames else pd.DataFrame(
        columns=["FspId", "Dof"]
    )


def rasterize_tile(tile_df: pd.DataFrame, dof_df: pd.DataFrame) -> "tuple[np.ndarray, int, int] | None":
    """Port of TileFspFppRelations2Array's DataFrame-fspDof branch:
    Dtf = Dof(FSP) - Dtf(FSP,FPP), keep positive, max across FSPs per pixel,
    + FilledDepth. Returns (dense_array, row_offset, col_offset) -- offsets
    are relative to the tile's own local origin (i.e. add the tile's global
    pixel origin, see create_gauge_flood_map) -- or None if nothing floods.
    """
    if tile_df.empty:
        return None
    max_dof = dof_df["Dof"].max()
    t = tile_df[tile_df["Dtf"] < max_dof]
    if t.empty:
        return None
    t = t.merge(
        dof_df[["FspId", "Dof"]].astype({"FspId": np.int32, "Dof": np.float32}), on="FspId", how="inner"
    )
    t = t.assign(Dtf=t["Dof"] - t["Dtf"])
    t = t[t["Dtf"] > 0]
    if t.empty:
        return None
    t = t.groupby(["FppCol", "FppRow"], as_index=False).agg(Dtf=("Dtf", "max"), FilledDepth=("FilledDepth", "first"))
    t["Dtf"] = t["Dtf"] + t["FilledDepth"]

    min_col, max_col = int(t["FppCol"].min()), int(t["FppCol"].max())
    min_row, max_row = int(t["FppRow"].min()), int(t["FppRow"].max())
    arr = np.full((max_row - min_row + 1, max_col - min_col + 1), NODATA, dtype=np.float32)
    arr[(t["FppRow"] - min_row).to_numpy(), (t["FppCol"] - min_col).to_numpy()] = t["Dtf"].to_numpy(
        dtype=np.float32
    )
    return arr, min_row, min_col


def create_gauge_flood_map(
    lib_dir: Path,
    gauges: pd.DataFrame,
    snap_dist: float,
    raster_info: RasterInfo,
    out_path: str,
    x_field: str = "x",
    y_field: str = "y",
) -> np.ndarray:
    """Full stage-3 driver: snap gauges -> interpolate FSP DOF -> rasterize
    every tile whose FSPs could flood -> mosaic into one full-AOI GeoTIFF.
    Reads fsp_info.csv / stream_order_info.csv / FLDPLN_tiled_fsp_index.csv /
    FLDPLN_tiled_tile_index.csv / FLDPLN_tiled_<id>.snz from lib_dir (i.e.
    the output of build-library + tile-library).
    """
    lib_dir = Path(lib_dir)
    fsp_df = pd.read_csv(lib_dir / "fsp_info.csv", float_precision="round_trip")
    str_ord_df = pd.read_csv(lib_dir / "stream_order_info.csv", float_precision="round_trip")
    tile_index = pd.read_csv(lib_dir / "FLDPLN_tiled_tile_index.csv")
    fsp_index = pd.read_csv(lib_dir / "FLDPLN_tiled_fsp_index.csv")

    gauge_fsp = snap_gauges_to_fsps(gauges, fsp_df, snap_dist, x_field, y_field)
    print(f"{len(gauge_fsp)}/{len(gauges)} gauge(s) snapped within {snap_dist} (native CRS units)")
    if gauge_fsp.empty:
        raise ValueError("no gauges snapped to any FSP -- check snap_dist/coordinate system")

    dof_df = estimate_fsp_dof(gauge_fsp, fsp_df, str_ord_df)
    print(f"{len(dof_df)} FSP(s) received an interpolated DOF")
    if dof_df.empty:
        raise ValueError("no FSPs received an interpolated DOF")

    reachable = fsp_index.merge(dof_df, on="FspId", how="inner")
    tile_ids = sorted(reachable.loc[reachable["Dof"] > reachable["MinDtf"], "TileId"].unique().tolist())
    print(f"{len(tile_ids)}/{tile_index['TileId'].nunique()} tile(s) selected for mapping")

    canvas = np.full((raster_info.rows, raster_info.cols), NODATA, dtype=np.float32)
    inv_transform = ~raster_info.transform
    for tid in tile_ids:
        trow = tile_index[tile_index["TileId"] == tid].iloc[0]
        tile_df = pd.read_parquet(lib_dir / f"FLDPLN_tiled_{tid}.snz")
        result = rasterize_tile(tile_df, dof_df)
        if result is None:
            continue
        arr, local_row0, local_col0 = result
        global_col0f, global_row0f = inv_transform * (trow["FppMinX"], trow["FppMaxY"])
        global_row0 = int(round(global_row0f)) + local_row0
        global_col0 = int(round(global_col0f)) + local_col0
        h, w = arr.shape
        dest = canvas[global_row0 : global_row0 + h, global_col0 : global_col0 + w]
        mask = arr != NODATA
        dest[mask] = arr[mask]
        print(f"  tile {tid}: {mask.sum()} flooded pixel(s) placed at rows[{global_row0}:{global_row0+h}] cols[{global_col0}:{global_col0+w}]")

    write_raster(out_path, canvas, like=raster_info, dtype="float32", nodata=NODATA, driver="GTiff")
    print(f"gauge-based flood map written to {out_path}")
    return canvas
