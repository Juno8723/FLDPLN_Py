"""Shared pytest fixtures.

Synthetic 5x5 D8 grid used across segmentation tests (see test_segmentation.py
module docstring for the full layout description):

  main stem:  (0,2) -> (1,2) -> (2,2)[confluence] -> (3,2) -> (4,2)[exits image]
  tributary:  (2,0) -> (2,1) -> joins (2,2)
"""

import numpy as np
import pytest

N_ROWS, N_COLS = 5, 5
SOUTH, EAST = 4, 1


@pytest.fixture
def synthetic_grid():
    fdr = np.full((N_ROWS, N_COLS), SOUTH, dtype=np.uint8)
    fac = np.full((N_ROWS, N_COLS), 1.0, dtype=np.float64)

    fdr[0, 2] = SOUTH
    fdr[1, 2] = SOUTH
    fdr[2, 2] = SOUTH
    fdr[3, 2] = SOUTH
    fdr[4, 2] = SOUTH
    fac[0, 2] = 6
    fac[1, 2] = 8
    fac[2, 2] = 20
    fac[3, 2] = 22
    fac[4, 2] = 24

    fdr[2, 0] = EAST
    fdr[2, 1] = EAST
    fac[2, 0] = 6
    fac[2, 1] = 10

    return fdr, fac
