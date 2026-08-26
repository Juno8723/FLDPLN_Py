"""Direct Python port of library_files.m: converts a segment's raw
flood-fill model output (fsp pixel id, floodplain pixel id, dtf,
fill_depth -- exactly what segmentation/runner.py's run_segment_to_disk already
writes) into the final map-coordinate "library" relation table used by
tiling/mapping.

Key semantic operation being replicated (library_files.m:89-96): re-group
the segment's whole FSP-FPP relation table so FSPs appear in the canonical
order given by the segment's own FSP table (not whatever order the raw
model output happened to emit them in), and within each FSP's group, sort
floodplain pixels by ascending DTF. `sortrows` in MATLAB is a stable sort
(ties keep their original relative order) -- replicated here with
`kind="stable"`.

Unlike the original (whose raw `fldpln` column 4 is cumulative "DTF + fill
depth", later un-done by the existing Tools/fldpln_reorg.py's
`tdf['FilledDepth'] -= tdf['Dtf']`, fldpln_reorg.py:142), our own
segmentation/runner.py already computes the final, non-cumulative fill_depth
directly (`fil[pixels] - dem[pixels]`) -- so no extra subtraction step is
needed here; fill_depth is converted to feet and used as-is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..segmentation.runner import output_basename
from ..segmentation.units import METERS_PER_FOOT


def build_segment_library_relations(
    seg_id: int,
    fsp_df: pd.DataFrame,
    fldmx: float,
    dh: float,
    segdr: Path,
    n_cols: int,
) -> pd.DataFrame:
    """Read this segment's raw model-output Parquet and reorganize it into
    the final relation table: [FspId, FppRow, FppCol, Dtf, FilledDepth]
    (feet), FSPs in fsp_df's canonical order, each FSP's floodplain pixels
    stable-sorted by ascending Dtf.

    Raises ValueError if any relation's fsp pixel isn't one of this
    segment's own FSPs -- replicates library_files.m's integrity assertion
    (lines 110-118: `all(ismember(fldpln(:,1:2), pth_map0, 'rows'))`, "FSP
    mismatch for segment ...") rather than silently writing bad data.
    """
    segdr = Path(segdr)
    basename = output_basename(fldmx, dh, seg_id)
    final_path = segdr / f"{basename}.parquet"
    tmp_path = segdr / f"{basename}_tmp.parquet"
    path = final_path if final_path.exists() else tmp_path
    if not path.exists():
        raise FileNotFoundError(f"no output found for segment {seg_id} in {segdr}")
    raw = pd.read_parquet(path)

    seg_fsps = fsp_df[fsp_df["SegId"] == seg_id]
    pixel_to_fsp_id = dict(zip(seg_fsps["_pixel"].to_numpy(), seg_fsps["FspId"].to_numpy()))

    unknown = set(raw["fsp"].unique()) - pixel_to_fsp_id.keys()
    if unknown:
        raise ValueError(
            f"FSP mismatch for segment {seg_id}: {len(unknown)} relation(s) reference "
            f"a pixel not in this segment's own FSP set (fsp_table/model output out of sync)"
        )

    fpp_rows, fpp_cols = np.divmod(raw["pixel"].to_numpy(), n_cols)
    # Column order/dtypes match fldpln_header.py's relColumnNames and the
    # confirmed on-disk .snz schema (FspId/FppCol/FppRow: int32, Dtf/
    # FilledDepth: float32) for direct compatibility with the tiling stage.
    frame = pd.DataFrame(
        {
            "FspId": raw["fsp"].map(pixel_to_fsp_id).to_numpy(dtype=np.int32),
            "FppCol": fpp_cols.astype(np.int32),
            "FppRow": fpp_rows.astype(np.int32),
            "Dtf": (raw["dtf"].to_numpy(dtype=np.float64) / METERS_PER_FOOT).astype(np.float32),
            "FilledDepth": (raw["fill_depth"].to_numpy(dtype=np.float64) / METERS_PER_FOOT).astype(
                np.float32
            ),
        }
    )

    # Canonical FSP order (this segment's fsp_df rows are already
    # upstream->downstream / FspId-ascending by construction, see
    # library_reorganization/fsp_table.py:build_fsp_table), each group stable-sorted by Dtf.
    frame["_order"] = frame["FspId"].map(
        {fid: i for i, fid in enumerate(seg_fsps["FspId"].to_numpy())}
    )
    frame = frame.sort_values(["_order", "Dtf"], kind="stable").drop(columns="_order")
    return frame.reset_index(drop=True)
