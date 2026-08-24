"""Direct Python port of fsp_info_table.m's core computation, plus the
FSP/segment downstream-distance calculation that the existing (pre-built,
not-ours) Tools/fldpln_reorg.py's CreateFspSegmentInfoSpatialReferenceFiles
performs as a second pass over Excel files (fldpln_reorg.py:337-524).

Adapted to consume our already-in-memory pipeline outputs (seg_info, fil,
fdr) directly instead of round-tripping through raw BIL seeks + an Excel
intermediate file -- this project's segments are built directly from
FDR/FAC with no lossy raster->polyline->dissolve round-trip (see
segmentation/network_core.py's build_segment_geometries docstring), so
there is no "some segments silently went missing" cleanup step to replicate
(fsp_info_table.m:119's seg_list/ismember filtering, and
CreateFspSegmentInfoSpatialReferenceFiles's "remove segment if not in FSP
table" loop, are both structurally unnecessary here).

Each segment's FSP list (its own stream centerline pixels) is re-derived via
trace_downstream, exactly the same call already used inside
model/flood_fill.py's run_segment -- so a segment's FSPs here are
guaranteed identical to the stream points its own flood-fill run used.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from rasterio.transform import xy as transform_xy

from ..model.flood_fill import trace_downstream
from ..segmentation.units import METERS_PER_FOOT


def build_fsp_table(
    seg_info: pd.DataFrame,
    fil: np.ndarray,
    fdr: np.ndarray,
    transform,
    n_rows: int,
    n_cols: int,
) -> pd.DataFrame:
    """One row per FSP (flood source pixel = stream centerline pixel),
    ordered upstream-to-downstream within each segment, segments
    concatenated in SegId order (SegId = 1-based row position in seg_info,
    matching the rest of this package's convention).

    Columns: FspId (0-based row position, matching the existing toolkit's
    `fspDf['FspId'] = fspDf.index` convention, fldpln_reorg.py:499), SegId,
    FspX, FspY, FilledElev (feet), and an internal `_pixel` column (the
    FSP's own raw flat pixel id) kept for joining against a segment's raw
    model output by pixel id rather than by fragile float-coordinate
    equality (the existing toolkit joins on (FspX,FspY) throughout, e.g.
    fldpln.py:372 -- flagged as fragile in the stage-2 audit; `_pixel`
    avoids that here and should be dropped before writing a final
    fsp_info.csv to match the reference schema).

    FilledElev's meters->feet conversion replicates fsp_info_table.m:178
    (`tabl(:,5) = tabl(:,5)/0.3048`) applied at this exact pipeline point.
    """
    frames = []
    for seg_id in range(1, len(seg_info) + 1):
        seg_row = seg_info.iloc[seg_id - 1]
        pixels = trace_downstream(
            int(seg_row["start_pixel"]), int(seg_row["length"]), fdr, n_rows, n_cols
        )
        rows, cols = np.divmod(pixels, n_cols)
        xs, ys = transform_xy(transform, rows, cols)
        frames.append(
            pd.DataFrame(
                {
                    "SegId": seg_id,
                    "FspX": np.asarray(xs, dtype=np.float64),
                    "FspY": np.asarray(ys, dtype=np.float64),
                    "FilledElev": fil[pixels].astype(np.float64) / METERS_PER_FOOT,
                    "_pixel": pixels,
                }
            )
        )
    fsp_df = pd.concat(frames, ignore_index=True)
    fsp_df.insert(0, "FspId", fsp_df.index.to_numpy(dtype=np.int64))
    return fsp_df


def add_downstream_distances(seg_info: pd.DataFrame, fsp_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'DsDist' column onto fsp_df (in place) and returns a
    segment-level DataFrame [SegId, CellCount, DsSegId, Length, DsDist].

    Replicates CreateFspSegmentInfoSpatialReferenceFiles's Length/DsDist
    algorithm exactly (fldpln_reorg.py:383-462):
      - Length / a FSP's within-segment DsDist: walk a segment's FSPs
        downstream->upstream, summing consecutive Euclidean centroid
        distances (0 at the most-downstream FSP).
      - A segment's DsDist (to the network outlet): walk downstream
        segment-by-segment, adding each downstream segment's Length plus
        the real Euclidean gap between the two segments' adjacent FSPs
        (segments are pixel-center chains, so there's a genuine ~1-to-
        sqrt(2)-cell gap at every confluence -- using real FSP coordinates
        for the gap reproduces this for free, no separate correction term
        needed).
      - A FSP's final DsDist = its within-segment DsDist + its segment's
        DsDist.
    """
    n = len(seg_info)
    downstream = seg_info["downstream"].to_numpy()
    length_px = seg_info["length"].to_numpy()

    seg_ids = fsp_df["SegId"].to_numpy()
    xs_all = fsp_df["FspX"].to_numpy()
    ys_all = fsp_df["FspY"].to_numpy()

    seg_length_m = np.zeros(n)
    within_dsdist = np.zeros(len(fsp_df))
    first_fsp = np.empty(n, dtype=np.int64)
    last_fsp = np.empty(n, dtype=np.int64)

    for seg_id in range(1, n + 1):
        idx = np.flatnonzero(seg_ids == seg_id)
        if len(idx) != length_px[seg_id - 1]:
            raise ValueError(
                f"segment {seg_id}: seg_info length={length_px[seg_id - 1]} "
                f"but found {len(idx)} FSPs -- fsp_df/seg_info mismatch"
            )
        first_fsp[seg_id - 1] = idx[0]
        last_fsp[seg_id - 1] = idx[-1]
        xs, ys = xs_all[idx], ys_all[idx]
        seg_dist = 0.0
        within = np.zeros(len(idx))
        for k in range(len(idx) - 2, -1, -1):
            seg_dist += math.hypot(xs[k] - xs[k + 1], ys[k] - ys[k + 1])
            within[k] = seg_dist
        within_dsdist[idx] = within
        seg_length_m[seg_id - 1] = seg_dist

    seg_ds_dist = np.zeros(n)
    for seg_id in range(1, n + 1):
        i = seg_id - 1
        cur, ds = i, downstream[i]
        dist = 0.0
        while ds != -1:
            dist += seg_length_m[ds]
            dist += math.hypot(
                xs_all[last_fsp[cur]] - xs_all[first_fsp[ds]],
                ys_all[last_fsp[cur]] - ys_all[first_fsp[ds]],
            )
            cur, ds = ds, downstream[ds]
        seg_ds_dist[i] = dist

    fsp_df["DsDist"] = within_dsdist + seg_ds_dist[seg_ids - 1]

    seg_df = pd.DataFrame(
        {
            "SegId": np.arange(1, n + 1),
            "CellCount": length_px,
            "DsSegId": np.where(downstream == -1, 0, downstream + 1),
            "Length": seg_length_m,
            "DsDist": seg_ds_dist,
        }
    )
    return seg_df
