"""Automated stream-order assignment (level-path method).

Replaces the manual ArcGIS step documented in
FLDPLN_MatlabRaw/Tools/ReorganizeLibrary.ipynb ("Assign segment stream
order"), whose own markdown cell notes: "The level path approach used in NWC
may be used here to automate stream order assignment!" (NWC = NOAA National
Water Center; a level path walks each stream network from its outlet
upstream, and at every confluence the branch with the larger drainage area
continues the same path id while every other branch starts a new one).

Two rules the manual process had to satisfy (same markdown cell), which this
implementation satisfies by construction:
    1. Low order number = main stem; high order = tributary (opposite of
       Strahler order).
    2. Order numbers are globally unique per contiguous stream network, even
       across separate sub-watersheds/outlets within the same library.

Uses only seg_info's existing 'downstream' linkage and 'fac_end' column
(flow accumulation at the segment's downstream-adjacent end, i.e. the
drainage-area signal at the confluence) -- no additional raster reads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_stream_order(seg_info: pd.DataFrame) -> np.ndarray:
    """Return one stream-order int per seg_info row (row position == seg_id
    - 1, matching the rest of this codebase's 1-based-row-position
    convention). Orders start at 1 and are globally unique per contiguous
    network; ties in fac_end at a confluence are broken by segment row
    position (deterministic, but arbitrary -- there is no MATLAB reference
    to match here since stream order was previously assigned by hand).
    """
    n = len(seg_info)
    downstream = seg_info["downstream"].to_numpy()
    fac_end = seg_info["fac_end"].to_numpy()

    upstream_of: dict[int, list[int]] = {}
    for i, d in enumerate(downstream):
        if d != -1:
            upstream_of.setdefault(int(d), []).append(i)

    order = np.full(n, -1, dtype=np.int64)
    next_order = 1

    roots = [i for i in range(n) if downstream[i] == -1]
    roots.sort(key=lambda r: (-fac_end[r], r))

    for root in roots:
        order[root] = next_order
        next_order += 1
        stack = [root]
        while stack:
            seg = stack.pop()
            ups = upstream_of.get(seg, [])
            if not ups:
                continue
            ups_sorted = sorted(ups, key=lambda u: (-fac_end[u], u))
            main, tributaries = ups_sorted[0], ups_sorted[1:]
            order[main] = order[seg]
            for trib in tributaries:
                order[trib] = next_order
                next_order += 1
            stack.extend(tributaries)
            stack.append(main)

    if (order == -1).any():
        missing = np.flatnonzero(order == -1)
        raise ValueError(
            f"stream order left unassigned for {len(missing)} segment(s) "
            f"(row positions {missing[:10].tolist()}...) -- seg_info's "
            "downstream linkage likely contains a cycle"
        )
    if len(set(order[roots].tolist())) != len(roots):
        raise ValueError("root segments did not receive distinct orders")
    return order


def build_stream_order_info(
    seg_info: pd.DataFrame, order: np.ndarray, fsp_df: "pd.DataFrame | None" = None
) -> pd.DataFrame:
    """Build the StrOrd/DsStrOrd/JunctionFspX/JunctionFspY connectivity
    table, matching fldpln_header.py's strOrdColumnNames exactly, needed by
    the downstream gauge-interpolation stage.

    Junction FSP coordinates replicate GetStreamOrdersForFspsSegments's
    convention (fldpln_reorg.py:649-665): for a stream order with a real
    downstream order, the junction point is the FIRST (most upstream, by
    fsp_df's own upstream->downstream row order -- see
    library/fsp_table.py:build_fsp_table) FSP of the downstream *segment*.
    Outlet orders (no downstream order) get the sentinel (DsStrOrd=0,
    JunctionFspX=0.0, JunctionFspY=0.0), matching fldpln_reorg.py:663-665.

    Pass fsp_df to populate JunctionFspX/Y; omit it (leaves 0.0 sentinels
    everywhere) if only the StrOrd/DsStrOrd connectivity is needed.
    """
    n = len(seg_info)
    downstream = seg_info["downstream"].to_numpy()

    rows = []
    for str_ord in sorted(set(order.tolist())):
        members = np.flatnonzero(order == str_ord)
        # The order's most-downstream segment is the one whose own
        # 'downstream' pointer either leaves the order (crosses to a
        # different StrOrd) or is -1 (network outlet).
        ds_seg = None
        for m in members:
            d = downstream[m]
            if d == -1 or order[d] != str_ord:
                ds_seg = m
                break
        if ds_seg is None:
            raise ValueError(f"stream order {str_ord} has no exit segment (cycle?)")
        d = downstream[ds_seg]
        if d == -1:
            rows.append(
                {"StrOrd": int(str_ord), "DsStrOrd": 0, "JunctionFspX": 0.0, "JunctionFspY": 0.0}
            )
            continue
        ds_order = int(order[d])
        junc_x, junc_y = 0.0, 0.0
        if fsp_df is not None:
            first_row = fsp_df[fsp_df["SegId"] == d + 1].iloc[0]
            junc_x, junc_y = float(first_row["FspX"]), float(first_row["FspY"])
        rows.append(
            {"StrOrd": int(str_ord), "DsStrOrd": ds_order, "JunctionFspX": junc_x, "JunctionFspY": junc_y}
        )

    return pd.DataFrame(rows, columns=["StrOrd", "DsStrOrd", "JunctionFspX", "JunctionFspY"])
