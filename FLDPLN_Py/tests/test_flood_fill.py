"""Unit test for the flood-fill core, against a small hand-derived 3x3
scenario (see derivation notes below -- every expected value here was
computed by hand by tracing fldpln_model_v8.m's exact governing logic).

Grid (3x3, 0-based row/col), flat pixel id = row*3+col:
  (0,0)=0  (0,1)=1  (0,2)=2
  (1,0)=3  (1,1)=4  (1,2)=5
  (2,0)=6  (2,1)=7  (2,2)=8

Segment (the "stream"): (0,1) -> (1,1) -> (2,1), all flowing South.
Elevations (FIL): channel (0,1)=10, (1,1)=8, (2,1)=6 (decreasing downstream).
Left bank (1,0)=9 flows East into (1,1) -> reachable by BACKFILL.
Right bank (1,2)=12 flows South (away from the channel, not into it) -> NOT
reachable by backfill (no D8 inflow connection); only reachable by SPILLOVER
once floodwater rises enough. (2,2)=20 continues (1,2)'s flow path and gets
swept in as part of the same spillover trace before exiting the image.
Everything else is elevation 99 (unreachable within this test's flood range).

fldmn=0.1, dh=5, fldmx=5 (single iteration), no height cap, ssflg=False
(one spillover pass is enough here since (1,2) and (2,2) are captured by a
single forward trace from one spillover target).
"""

import numpy as np

from PyFLDPLN_src.model.flood_fill import run_segment

N_ROWS, N_COLS = 3, 3
SOUTH, EAST = 4, 1
BG = -9999.0


def px(row, col):
    return row * N_COLS + col


def build_grid():
    fil = np.full((N_ROWS, N_COLS), 99.0, dtype=np.float64)
    fdr = np.full((N_ROWS, N_COLS), EAST, dtype=np.uint8)  # inert filler

    fil[0, 1] = 10.0
    fil[1, 1] = 8.0
    fil[2, 1] = 6.0
    fdr[0, 1] = SOUTH
    fdr[1, 1] = SOUTH
    fdr[2, 1] = SOUTH

    fil[1, 0] = 9.0
    fdr[1, 0] = EAST  # flows into (1,1) -> backfill-reachable

    fil[1, 2] = 12.0
    fdr[1, 2] = SOUTH  # flows away from channel -> spillover-only

    fil[2, 2] = 20.0
    fdr[2, 2] = SOUTH  # continues (1,2)'s forward trace, then exits image

    return fil.flatten(), fdr.flatten()


def test_run_segment_backfill_and_spillover():
    fil, fdr = build_grid()

    entries = run_segment(
        seg_id=1,
        start_pixel=px(0, 1),
        end_pixel=px(2, 1),
        length=3,
        fil=fil,
        fdr=fdr,
        fldmn=0.1,
        fldmx=5,
        dh=5,
        mxht0=0,
        ssflg=False,
        bg=BG,
        n_rows=N_ROWS,
        n_cols=N_COLS,
    )

    result = {e.pixel: (e.fsp, round(e.dtf, 6)) for e in entries}

    expected = {
        px(0, 1): (px(0, 1), 0.1),
        px(1, 1): (px(1, 1), 0.1),
        px(2, 1): (px(2, 1), 0.1),
        px(1, 0): (px(1, 1), 1.0),   # backfilled from (1,1), depth = 0 + (9-8)
        px(1, 2): (px(0, 1), 2.0),   # spillover from (0,1): 0 + (12-10)
        px(2, 2): (px(0, 1), 2.0),   # same forward-traced spillover path
    }

    assert result == expected
    assert len(result) == 6


def test_run_segment_second_iteration_is_stable_with_no_growth():
    """With fldmx=10, dh=5 (2 iterations), every remaining unflooded neighbor
    in this grid has elevation 99 -- far beyond reach even at fldht=10 -- so
    the second iteration should find nothing new and leave iteration 1's
    result unchanged (checks multi-iteration stability/idempotency)."""
    fil, fdr = build_grid()

    entries = run_segment(
        seg_id=1,
        start_pixel=px(0, 1),
        end_pixel=px(2, 1),
        length=3,
        fil=fil,
        fdr=fdr,
        fldmn=0.1,
        fldmx=10,
        dh=5,
        mxht0=0,
        ssflg=False,
        bg=BG,
        n_rows=N_ROWS,
        n_cols=N_COLS,
    )

    result = {e.pixel: (e.fsp, round(e.dtf, 6)) for e in entries}
    expected = {
        px(0, 1): (px(0, 1), 0.1),
        px(1, 1): (px(1, 1), 0.1),
        px(2, 1): (px(2, 1), 0.1),
        px(1, 0): (px(1, 1), 1.0),
        px(1, 2): (px(0, 1), 2.0),
        px(2, 2): (px(0, 1), 2.0),
    }
    assert result == expected
