"""Per-segment flood-fill simulation, porting the core algorithm of
fldpln_model_v8.m (Kastens 2008 FLDPLN model).

The original file has three branches (flg==1 'ram', flg==2 'ram0', flg==3
'hd') that were confirmed via line-by-line diff (during the MATLAB analysis
phase of this project) to implement the *exact same algorithm* -- they only
differ in whether fil/fdr/the working flood-zone indicator are held as
in-RAM arrays or read via repeated disk seeks. This module ports the
algorithm once, operating on plain in-RAM numpy arrays; a disk-backed
(memmap) variant can reuse the same functions later without changing the
governing logic, since numpy arrays and memmaps share the same indexing
interface.

Pixel ids are flat, 0-based, row-major (v = row*n_cols + col), matching the
rest of this package (see PyFLDPLN_src/segmentation/d8.py's module docstring for
the indexing-convention note).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit, types
from numba.typed import Dict

from ..segmentation import d8

# Matches MATLAB's realmax('single'): effectively "no cap" for mxht0.
REALMAX_SINGLE = float(np.finfo(np.float32).max)

# Numba-friendly (int64) copies of d8's neighbor-offset tables, and plain
# Python-int tuples of the same for the handful of call sites that stay in
# pure Python -- profiling on real segments showed the *generator*
# form (d8-adjacent `_in_bounds_neighbors`, used millions of times per
# segment) was itself the single largest cost, ahead of any one algorithm
# step, purely from CPython generator/yield overhead -- so every hot loop
# below inlines its own fixed 8-neighbor scan instead of delegating to it.
_ROW_OFF = d8.ROW_OFFSETS.astype(np.int64)
_COL_OFF = d8.COL_OFFSETS.astype(np.int64)
_INFLOW = d8.INFLOW_CODES.astype(np.int64)
_ROW_OFF_PY = tuple(int(x) for x in d8.ROW_OFFSETS)
_COL_OFF_PY = tuple(int(x) for x in d8.COL_OFFSETS)


# ---------------------------------------------------------------------------
# Path tracing helpers (re-derive a segment's own pixels / its downstream
# exclusion zone from FDR, exactly as fldpln_model_v8.m does rather than
# storing full per-pixel paths in seg_info).
# ---------------------------------------------------------------------------


def trace_downstream(
    start_px: int, length: int, fdr: np.ndarray, n_rows: int, n_cols: int
) -> np.ndarray:
    """Step `length` pixels downstream via FDR starting at start_px (inclusive).
    Stops early (returning fewer than `length` pixels) if the path leaves the
    image or hits an invalid FDR value.
    """
    pixels = np.empty(length, dtype=np.int64)
    pixels[0] = start_px
    px = start_px
    for i in range(1, length):
        row, col = divmod(px, n_cols)
        step = d8.step_downstream(row, col, int(fdr[px]))
        if step is None:
            return pixels[:i]
        nr, nc = step
        if not (0 <= nr < n_rows and 0 <= nc < n_cols):
            return pixels[:i]
        px = nr * n_cols + nc
        pixels[i] = px
    return pixels


def trace_exclusion_zone(
    end_px: int, fdr: np.ndarray, n_rows: int, n_cols: int
) -> np.ndarray:
    """Trace downstream from a segment's end pixel to the image edge / an
    invalid FDR value, collecting pixels (including end_px itself) to
    exclude from spillover -- this prevents floodwater from this segment
    spilling back into the main channel it drains into.
    """
    pixels = [end_px]
    px = end_px
    while True:
        row, col = divmod(px, n_cols)
        step = d8.step_downstream(row, col, int(fdr[px]))
        if step is None:
            break
        nr, nc = step
        if not (0 <= nr < n_rows and 0 <= nc < n_cols):
            break
        px = nr * n_cols + nc
        pixels.append(px)
    return np.array(pixels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Backfill: grow a flood reach from a seed pixel via strict D8 inflow
# connections, capped at a maximum elevation.
# ---------------------------------------------------------------------------


@njit(cache=True)
def _backfill_reach_nb(start_px, fil0, mxht, fil, fdr, flddat, bg, n_rows, n_cols,
                        row_off, col_off, inflow, blocked, check_flooded):
    """JIT kernel for backfill_reach (see wrapper below for the docstring/
    semantics). `blocked` is always a (possibly empty) int64->boolean typed
    Dict -- the "no extra_blocked" call sites pass a shared empty one so this
    signature never needs an Optional type.

    `check_flooded` toggles whether `flddat[npx] != 0.0` excludes a neighbor.
    This must be True for the main per-iteration backfill (matches the
    original's 3-column dat=[FDR,FIL,FLDMAP0] filter at
    fldpln_model_v8.m:383-397) but False for backfill along a spillover path
    (the original's 2-column dat=[FDR,FIL] filter at lines 766-784/818-836
    has no flddat check at all -- it deliberately lets this reach re-discover
    already-flooded pixels so the caller's "reclaim if shallower" step
    (lines 844-868) can pull them onto a new, shallower path). Excluding
    already-flooded pixels unconditionally here -- as a prior version of this
    port did -- silently defeats that reclaim mechanism, since it never gets
    the chance to see the pixel again.

    Traversal is an explicit FIFO queue (breadth-first), matching the
    original's `fldtmp` array: scanned front-to-back via an incrementing
    index while newly-qualifying neighbors are appended at the end.
    """
    visited = Dict.empty(key_type=types.int64, value_type=types.boolean)

    cap = 64
    out_px = np.empty(cap, dtype=np.int64)
    out_depth = np.empty(cap, dtype=np.float64)
    cnt = 0

    qcap = 64
    queue = np.empty(qcap, dtype=np.int64)
    queue[0] = start_px
    qsize = 1
    head = 0

    while head < qsize:
        px = queue[head]
        head += 1
        row = px // n_cols
        col = px % n_cols
        for k in range(8):
            nr = row + row_off[k]
            nc = col + col_off[k]
            if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                continue
            npx = nr * n_cols + nc
            if npx in visited or npx in blocked:
                continue
            if check_flooded and flddat[npx] != 0.0:
                continue
            if fdr[npx] != inflow[k]:
                continue
            nelev = fil[npx]
            if nelev == bg or nelev > mxht:
                continue
            visited[npx] = True

            depth = nelev - fil0
            if depth < 0.0:
                depth = 0.0
            if cnt >= cap:
                cap *= 2
                new_px = np.empty(cap, dtype=np.int64)
                new_depth = np.empty(cap, dtype=np.float64)
                new_px[:cnt] = out_px[:cnt]
                new_depth[:cnt] = out_depth[:cnt]
                out_px, out_depth = new_px, new_depth
            out_px[cnt] = npx
            out_depth[cnt] = depth
            cnt += 1

            if qsize >= qcap:
                qcap *= 2
                new_queue = np.empty(qcap, dtype=np.int64)
                new_queue[:qsize] = queue[:qsize]
                queue = new_queue
            queue[qsize] = npx
            qsize += 1

    return out_px[:cnt], out_depth[:cnt]


# Shared empty "no extra_blocked" scratch dict, reused across the (far more
# common) boundary-backfill call site so it isn't reallocated per call.
_NO_BLOCKED = Dict.empty(key_type=types.int64, value_type=types.boolean)


def backfill_reach(
    start_px: int,
    fil0: float,
    mxht: float,
    fil: np.ndarray,
    fdr: np.ndarray,
    flddat: np.ndarray,
    bg: float,
    n_rows: int,
    n_cols: int,
    extra_blocked: set | None = None,
    check_flooded: bool = True,
) -> list[tuple[int, float]]:
    """BFS/flood-fill outward from start_px, following only strict D8 inflow
    connections (neighbor's FDR must point into the current pixel), capped at
    elevation <= mxht. A pixel is off-limits if listed in `extra_blocked` (a
    small local set -- used by the "spilled-into path" backfill usage to also
    exclude pixels already part of the new path itself) or, when
    `check_flooded` is True, already in the global flood zone (`flddat[px]`
    truthy).

    `check_flooded=False` is for the spillover-path backfill call site: the
    original allows that reach to re-discover already-flooded pixels so a
    later "reclaim if shallower" step can pull them onto a new, shorter path
    (see `_backfill_reach_nb`'s docstring); the main per-iteration backfill
    call site keeps the default `check_flooded=True`.

    NOTE: `flddat` is read directly (never copied) so this stays cheap to
    call once per boundary pixel per iteration, even on large rasters.

    Returns [(pixel, depth_above_fil0), ...] for newly-reached pixels
    (start_px itself is not included).
    """
    if extra_blocked:
        blocked = Dict.empty(key_type=types.int64, value_type=types.boolean)
        for px in extra_blocked:
            blocked[np.int64(px)] = True
    else:
        blocked = _NO_BLOCKED

    out_px, out_depth = _backfill_reach_nb(
        np.int64(start_px), float(fil0), float(mxht), fil, fdr, flddat, bg,
        n_rows, n_cols, _ROW_OFF, _COL_OFF, _INFLOW, blocked, check_flooded,
    )
    return list(zip(out_px.tolist(), out_depth.tolist()))


def _in_bounds_neighbors(row: int, col: int, n_rows: int, n_cols: int):
    for pos in range(8):
        nr = row + int(d8.ROW_OFFSETS[pos])
        nc = col + int(d8.COL_OFFSETS[pos])
        if 0 <= nr < n_rows and 0 <= nc < n_cols:
            yield pos, nr * n_cols + nc


@njit(cache=True)
def _is_boundary_nb(px, flooded, n_rows, n_cols, row_off, col_off):
    row = px // n_cols
    col = px % n_cols
    for k in range(8):
        nr = row + row_off[k]
        nc = col + col_off[k]
        if 0 <= nr < n_rows and 0 <= nc < n_cols:
            if flooded[nr * n_cols + nc] == 0.0:
                return True
    return False


def is_boundary_pixel(px: int, flooded: np.ndarray, n_rows: int, n_cols: int) -> bool:
    """A flood-zone pixel is a 'boundary' pixel if at least one of its
    in-bounds neighbors is not yet in the flood zone."""
    return bool(_is_boundary_nb(np.int64(px), flooded, n_rows, n_cols, _ROW_OFF, _COL_OFF))


# ---------------------------------------------------------------------------
# Segment/model state
# ---------------------------------------------------------------------------


@dataclass
class FloodEntry:
    fsp: int  # flood source pixel (which stream pixel this water originated from)
    pixel: int
    dtf: float  # depth to flood


@dataclass
class SegmentFloodState:
    """Mutable state carried across iterations for one segment (checkpointable)."""

    entries: list[FloodEntry] = field(default_factory=list)
    fldht: float = 0.0
    # pixel -> index into `entries`, kept in sync via _add_entry(). Lets the
    # spillover "reclaim a shallower depth" step look up an existing entry in
    # O(1) instead of a linear scan (matters once a segment's flood zone
    # grows into the hundreds of thousands of pixels).
    pixel_index: dict = field(default_factory=dict)
    # Insertion-ordered set (dict-as-ordered-set) of pixels currently
    # classified as boundary. Maintained incrementally by _refresh_boundary
    # instead of being recomputed from scratch every iteration -- see that
    # function's docstring for why this is safe.
    boundary_order: dict = field(default_factory=dict)


def _add_entry(state: SegmentFloodState, entry: "FloodEntry") -> None:
    state.pixel_index[entry.pixel] = len(state.entries)
    state.entries.append(entry)


def _init_state(
    strpts: np.ndarray, fil: np.ndarray, bg: float, fldmn: float, n_rows: int, n_cols: int
) -> tuple[SegmentFloodState, np.ndarray]:
    state = SegmentFloodState()
    flddat = np.zeros(n_rows * n_cols, dtype=np.float64)
    for px in strpts:
        px = int(px)
        if fil[px] == bg:
            continue
        _add_entry(state, FloodEntry(fsp=px, pixel=px, dtf=0.0))
        flddat[px] = fldmn
    # Initial boundary = every seed pixel, unconditionally (matches the
    # original: it never filters the seed set by is_boundary_pixel).
    for e in state.entries:
        state.boundary_order[e.pixel] = None
    return state, flddat


def _refresh_boundary(
    state: SegmentFloodState,
    flooded_mask: np.ndarray,
    start_idx: int,
    n_rows: int,
    n_cols: int,
) -> None:
    """Incrementally update `state.boundary_order` after state.entries[start_idx:]
    have just been newly flooded, instead of rescanning every entry ever
    added (which is what a full `is_boundary_pixel` sweep over `state.entries`
    would cost -- O(total flood-zone size) on every single depth-step/
    spillover-pass, the dominant cost on large segments).

    This is exact, not an approximation: flooding is monotonic (a pixel is
    never un-flooded), so a pixel's boundary status can only change at the
    moment one of its neighbors transitions from unflooded to flooded --
    i.e. exactly when that neighbor is newly added here. So it's enough to
    re-evaluate is_boundary_pixel for (a) every newly-flooded pixel itself,
    and (b) every *already*-flooded neighbor of a newly-flooded pixel; every
    other pixel's boundary status is provably unchanged. Newly-flooded
    pixels are evaluated in their state.entries insertion order (matching
    the original's insertion-order-preserving list comprehension), and
    dict-key reassignment doesn't reorder an already-present key, so the
    resulting order exactly matches what a full rescan would have produced.
    """
    new_pixels = [e.pixel for e in state.entries[start_idx:]]
    seen = set(new_pixels)
    candidates = list(new_pixels)
    for px in new_pixels:
        row, col = divmod(px, n_cols)
        for dr, dc in zip(_ROW_OFF_PY, _COL_OFF_PY):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                continue
            npx = nr * n_cols + nc
            if flooded_mask[npx] and npx not in seen:
                seen.add(npx)
                candidates.append(npx)
    for px in candidates:
        if is_boundary_pixel(px, flooded_mask, n_rows, n_cols):
            state.boundary_order[px] = None
        else:
            state.boundary_order.pop(px, None)


def _boundary_entries(state: SegmentFloodState) -> list[FloodEntry]:
    return [state.entries[state.pixel_index[px]] for px in state.boundary_order]


@njit(cache=True)
def _spillover_claims_nb(
    bdy_pixels, bdy_dtf, bdy_fsp, fil, flooded_mask, excluded, bg, fldht, mxht0,
    n_rows, n_cols, row_off, col_off,
):
    """JIT kernel for the O(boundary * 8) claims-competition scan -- profiling
    on real segments showed this was the single largest cost (~2/3 of
    total run_segment time on a 700k-pixel segment), more than backfill_reach,
    because the spillover while-loop can re-run this scan many times per
    depth-step when `ssflg` (steady state) is set.

    Returns (target_px, target_fsp, target_dtf) in first-claimed order (i.e.
    the same order Python dict insertion would produce for `claims`), so the
    final multi-key sort in the wrapper below is stable exactly the way the
    original's `list.sort` was.
    """
    claim_fsp = Dict.empty(key_type=types.int64, value_type=types.int64)
    claim_dtf = Dict.empty(key_type=types.int64, value_type=types.float64)
    claim_elev = Dict.empty(key_type=types.int64, value_type=types.float64)

    ocap = 64
    order = np.empty(ocap, dtype=np.int64)
    ocnt = 0

    n = bdy_pixels.shape[0]
    for i in range(n):
        fj0 = bdy_pixels[i]
        entry_dtf = bdy_dtf[i]
        entry_fsp = bdy_fsp[i]
        bdy_fil = fil[fj0]
        thr = bdy_fil + fldht - entry_dtf
        if mxht0 < thr:
            thr = mxht0
        row = fj0 // n_cols
        col = fj0 % n_cols
        for k in range(8):
            nr = row + row_off[k]
            nc = col + col_off[k]
            if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                continue
            npx = nr * n_cols + nc
            if excluded[npx] or flooded_mask[npx] != 0.0:
                continue
            nelev = fil[npx]
            if nelev == bg or nelev > thr:
                continue
            rd = nelev - bdy_fil
            if rd < 0.0:
                rd = 0.0
            resulting_dtf = entry_dtf + rd

            if npx in claim_dtf:
                prev_dtf = claim_dtf[npx]
                prev_elev = claim_elev[npx]
                if resulting_dtf < prev_dtf or (resulting_dtf == prev_dtf and bdy_fil > prev_elev):
                    claim_fsp[npx] = entry_fsp
                    claim_dtf[npx] = resulting_dtf
                    claim_elev[npx] = bdy_fil
            else:
                claim_fsp[npx] = entry_fsp
                claim_dtf[npx] = resulting_dtf
                claim_elev[npx] = bdy_fil
                if ocnt >= ocap:
                    ocap *= 2
                    new_order = np.empty(ocap, dtype=np.int64)
                    new_order[:ocnt] = order[:ocnt]
                    order = new_order
                order[ocnt] = npx
                ocnt += 1

    out_px = order[:ocnt].copy()
    out_fsp = np.empty(ocnt, dtype=np.int64)
    out_dtf = np.empty(ocnt, dtype=np.float64)
    for i in range(ocnt):
        p = out_px[i]
        out_fsp[i] = claim_fsp[p]
        out_dtf[i] = claim_dtf[p]
    return out_px, out_fsp, out_dtf


def _find_spillover_targets(
    boundary: list[FloodEntry],
    fil: np.ndarray,
    flooded_mask: np.ndarray,
    excluded: np.ndarray,
    bg: float,
    fldht: float,
    mxht0: float,
    strpts_order: dict[int, int],
    n_rows: int,
    n_cols: int,
) -> list[tuple[int, float, int]]:
    """For each boundary pixel, find neighboring pixels reachable by
    "spilling over" (any of its 8 neighbors within elevation reach, not
    restricted by D8 inflow direction). When multiple boundary pixels
    compete for the same target, keep whichever gives the target the
    shallowest resulting DTF; break remaining ties by the competing boundary
    pixel's own (higher) elevation. Returns a list of
    (fsp, resulting_dtf_at_target, target_pixel), sorted to match the
    original's processing-order rule: FSP position (descending, i.e. most
    downstream source first) -> resulting DTF (ascending) -> boundary pixel's
    own elevation (ascending).
    """
    if not boundary:
        return []

    bdy_pixels = np.fromiter((e.pixel for e in boundary), dtype=np.int64, count=len(boundary))
    bdy_dtf = np.fromiter((e.dtf for e in boundary), dtype=np.float64, count=len(boundary))
    bdy_fsp = np.fromiter((e.fsp for e in boundary), dtype=np.int64, count=len(boundary))

    out_px, out_fsp, out_dtf = _spillover_claims_nb(
        bdy_pixels, bdy_dtf, bdy_fsp, fil, flooded_mask, excluded, bg, fldht, mxht0,
        n_rows, n_cols, _ROW_OFF, _COL_OFF,
    )
    if len(out_px) == 0:
        return []

    # Matches the original's sortrows([-1 3 6]): FSP position descending (most
    # downstream source processed first), then resulting DTF ascending. The
    # third original sort key (competing boundary pixel's own elevation) was
    # already used inside the kernel to resolve which boundary pixel wins a
    # contested target, so it doesn't need to reappear here. np.lexsort is
    # stable, matching list.sort's tie-breaking (fall back to first-claimed
    # order) exactly.
    ranks = np.fromiter(
        (strpts_order.get(int(fsp), -1) for fsp in out_fsp), dtype=np.int64, count=len(out_fsp)
    )
    order_idx = np.lexsort((out_dtf, -ranks))
    return [
        (int(out_fsp[i]), float(out_dtf[i]), int(out_px[i])) for i in order_idx
    ]


def _trace_spillover_path(
    target_px: int,
    fdr: np.ndarray,
    fil: np.ndarray,
    flddat: np.ndarray,
    excluded: np.ndarray,
    bg: float,
    fldzn: float,
    n_rows: int,
    n_cols: int,
) -> list[tuple[int, float]]:
    """From a spillover target, continue tracing downstream (matching
    fldpln_model_v8.m's forward-flood trace), collecting [(pixel, elevation)]
    until: self-intersection, an excluded (downstream-exclusion-zone) pixel is
    reached (included, then stop), nodata is reached (stop, not included),
    or an existing flood-zone pixel at <= our current DTF is reached (stop,
    not included) -- meaning the new path has rejoined water that is already
    at least as shallow.
    """
    path: list[tuple[int, float]] = [(target_px, fil[target_px])]
    seen = {target_px}
    px = target_px
    while True:
        row, col = divmod(px, n_cols)
        step = d8.step_downstream(row, col, int(fdr[px]))
        if step is None:
            break
        nr, nc = step
        if not (0 <= nr < n_rows and 0 <= nc < n_cols):
            break
        npx = nr * n_cols + nc
        if npx in seen:
            break
        nelev = fil[npx]
        if excluded[npx]:
            path.append((npx, nelev))
            break
        if nelev == bg:
            break
        existing_dtf = flddat[npx]
        if existing_dtf and fldzn >= existing_dtf:
            break
        path.append((npx, nelev))
        seen.add(npx)
        px = npx
    return path


# ---------------------------------------------------------------------------
# Main per-segment driver
# ---------------------------------------------------------------------------


def run_segment(
    seg_id: int,
    start_pixel: int,
    end_pixel: int,
    length: int,
    fil: np.ndarray,
    fdr: np.ndarray,
    fldmn: float,
    fldmx: float,
    dh: float,
    mxht0: float,
    ssflg: bool,
    bg: float,
    n_rows: int,
    n_cols: int,
    on_iteration: "callable | None" = None,
    trace_pixels: "set[int] | None" = None,
    trace_log: "list | None" = None,
    capture_boundary_at: "set[tuple[int, int]] | None" = None,
    boundary_snapshots: "dict | None" = None,
) -> list[FloodEntry]:
    """Run the full flood-fill simulation for one segment, from DTF=0 up to
    fldmx in steps of dh. Returns the final list of FloodEntry (FSP, pixel,
    DTF), with DTF floored at fldmn (matching the original's final
    max(fldmn, dtf) pass).

    `on_iteration(fldht, entries)` is called after each completed depth step,
    for checkpointing.

    DEBUG-ONLY: if `trace_pixels` is given, every new-entry/reclaim event
    touching one of those pixels, AND every iteration's boundary-membership
    snapshot for those pixels, is appended to `trace_log`. Zero-cost/no-op
    when trace_pixels is None -- temporary instrumentation for the
    2026-08-13 FSP tie-break investigation, not meant to stay long-term.
    """
    strpts = trace_downstream(start_pixel, length, fdr, n_rows, n_cols)
    exclusion_zone = trace_exclusion_zone(end_pixel, fdr, n_rows, n_cols)

    excluded = np.zeros(n_rows * n_cols, dtype=bool)
    excluded[exclusion_zone] = True

    mxht0_eff = REALMAX_SINGLE if not mxht0 else 0.99999 * mxht0

    state, flddat = _init_state(strpts, fil, bg, fldmn, n_rows, n_cols)
    strpts_order = {int(px): i for i, px in enumerate(strpts)}

    itnum = int(np.ceil(fldmx / dh))
    fldht = state.fldht
    bdy = _boundary_entries(state)

    g_iter = 0
    for _ in range(itnum):
        g_iter += 1
        fldht = fldht + min(dh, fldmx - fldht)

        # ---- Backfill: grow the flood zone from every current boundary pixel ----
        # NOTE: `flddat` (a plain float array, truthy iff flooded) is passed
        # directly everywhere below -- never copied -- so this stays cheap
        # per boundary pixel even on large rasters.
        if trace_pixels is not None:
            bdy_set = {e.pixel for e in bdy}
            for tp in trace_pixels:
                trace_log.append(dict(iteration=g_iter, pass_=0, event="boundary_check(pre-backfill)",
                                       pixel=tp, fsp=None, dtf=None, old_dtf=None,
                                       in_boundary=(tp in bdy_set)))
        start_idx = len(state.entries)
        for entry in bdy:
            fil0 = fil[entry.pixel]
            v0 = fil0 + fldht - entry.dtf
            mxht = min(mxht0_eff, v0)
            reach = backfill_reach(entry.pixel, fil0, mxht, fil, fdr, flddat, bg, n_rows, n_cols)
            for px, rel_depth in reach:
                dtf = entry.dtf + rel_depth
                _add_entry(state, FloodEntry(fsp=entry.fsp, pixel=px, dtf=dtf))
                flddat[px] = max(fldmn, dtf)
                if trace_pixels is not None and px in trace_pixels:
                    trace_log.append(dict(iteration=g_iter, pass_=0, event="new(backfill)",
                                           pixel=px, fsp=entry.fsp, dtf=dtf, old_dtf=None))

        _refresh_boundary(state, flddat, start_idx, n_rows, n_cols)
        bdy = _boundary_entries(state)

        # ---- Spillover (repeated to exhaustion if ssflg / 'steady state') ----
        spill_flg = True
        pass_num = 0
        while spill_flg:
            pass_num += 1
            if trace_pixels is not None:
                bdy_set = {e.pixel for e in bdy}
                for tp in trace_pixels:
                    trace_log.append(dict(iteration=g_iter, pass_=pass_num, event="boundary_check(pre-spillover)",
                                           pixel=tp, fsp=None, dtf=None, old_dtf=None,
                                           in_boundary=(tp in bdy_set)))
            if capture_boundary_at is not None and (g_iter, pass_num) in capture_boundary_at:
                boundary_snapshots[(g_iter, pass_num)] = dict(
                    bdy=[(e.pixel, e.fsp, e.dtf) for e in bdy],
                    flddat=flddat.copy(),
                    excluded=excluded.copy(),
                    strpts_order=dict(strpts_order),
                    fldht=fldht,
                    mxht0_eff=mxht0_eff,
                )

            n_before = len(state.entries)
            targets = _find_spillover_targets(
                bdy, fil, flddat, excluded, bg, fldht, mxht0_eff,
                strpts_order, n_rows, n_cols,
            )
            if trace_pixels is not None:
                for fsp_t, fldzn_t, target_px_t in targets:
                    if target_px_t in trace_pixels:
                        trace_log.append(dict(iteration=g_iter, pass_=pass_num, event="target_candidate",
                                               pixel=target_px_t, fsp=fsp_t, dtf=fldzn_t, old_dtf=None))
            for fsp, fldzn, target_px in targets:
                # NOTE: the original does not skip a target just because it's
                # already flooded (e.g. by an earlier target's path/backfill
                # this same pass) -- it still traces/backfills from it and
                # lets the reclaim-if-shallower step below decide whether the
                # new path wins. Skipping here would silently drop legitimate
                # reclaim opportunities.
                path = _trace_spillover_path(
                    target_px, fdr, fil, flddat, excluded, bg, fldzn, n_rows, n_cols
                )
                path_pixels = {px for px, _ in path}
                d0 = min(mxht0_eff, fil[target_px] + fldht - fldzn)
                fldstg0 = d0 - fil[target_px]

                new_entries: dict[int, float] = {px: 0.0 for px, _ in path}
                for px, elev in path:
                    v0 = elev + fldstg0
                    reach = backfill_reach(
                        px, elev, v0, fil, fdr, flddat, bg, n_rows, n_cols,
                        extra_blocked=path_pixels, check_flooded=False,
                    )
                    for rpx, rel_depth in reach:
                        if rpx not in new_entries or rel_depth < new_entries[rpx]:
                            new_entries[rpx] = rel_depth

                for px, rel_depth in new_entries.items():
                    dtf = rel_depth + fldzn
                    if flddat[px]:
                        e = state.entries[state.pixel_index[px]]
                        if e.dtf > dtf:
                            # reclaim: a shallower path reaches this pixel.
                            # Compares against the entry's own raw DTF
                            # (matching MATLAB's fldpln_info(:,3), which stays
                            # unfloored until the very end of the segment) --
                            # NOT flddat[px], which is floored to fldmn on
                            # every write. Using the floored value here made
                            # any two candidates whose raw DTF are both below
                            # fldmn compare as "tied at fldmn" and always
                            # reclaim regardless of which was truly deeper,
                            # turning ties (and near-ties) into "whichever
                            # candidate is processed last wins" instead of
                            # "whichever is truly shallowest wins".
                            old_dtf = flddat[px]
                            old_fsp = e.fsp
                            e.fsp = fsp
                            e.dtf = dtf
                            flddat[px] = max(fldmn, dtf)
                            if trace_pixels is not None and px in trace_pixels:
                                trace_log.append(dict(iteration=g_iter, pass_=pass_num, event="reclaim",
                                                       pixel=px, fsp=fsp, dtf=dtf, old_dtf=old_dtf,
                                                       old_fsp=old_fsp))
                    else:
                        _add_entry(state, FloodEntry(fsp=fsp, pixel=px, dtf=dtf))
                        flddat[px] = max(fldmn, dtf)
                        if trace_pixels is not None and px in trace_pixels:
                            trace_log.append(dict(iteration=g_iter, pass_=pass_num, event="new(spillover)",
                                                   pixel=px, fsp=fsp, dtf=dtf, old_dtf=None))

            _refresh_boundary(state, flddat, n_before, n_rows, n_cols)
            bdy = _boundary_entries(state)
            grew = len(state.entries) > n_before
            if ssflg and grew and any(e.dtf < fldht for e in bdy):
                continue
            spill_flg = False

        state.fldht = fldht
        if on_iteration is not None:
            on_iteration(fldht, state.entries)

    for e in state.entries:
        e.dtf = max(fldmn, e.dtf)
    return state.entries
