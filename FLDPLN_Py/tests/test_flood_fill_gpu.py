"""Validates the GPU (cupy) spillover-claims path against the CPU numba
kernel it's meant to replace under `run.use_gpu` -- see
`_spillover_claims_reduce`'s docstring in flood_fill.py for why the two are
expected to be exactly, not approximately, equal.

Skipped entirely (not failed) if cupy/a GPU aren't available, since `gpu` is
an optional extra -- `pip install -e ".[gpu]"` to run these.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from PyFLDPLN_src.segmentation.flood_fill import (  # noqa: E402
    _COL_OFF,
    _ROW_OFF,
    _spillover_claims_candidates,
    _spillover_claims_nb,
    _spillover_claims_reduce,
    run_segment,
)

BG = -9999.0


def _random_scenario(rng, n_rows, n_cols, n_boundary, n_fsp, tie_prone):
    """A raster + boundary set deliberately prone to ties when `tie_prone`
    (many pixels share the same elevation, and multiple boundary entries
    share the same (dtf, elevation) so the reduction's tie-break -- not just
    its primary sort -- gets exercised) -- otherwise fully random."""
    n = n_rows * n_cols
    if tie_prone:
        fil = rng.choice([5.0, 10.0, 15.0, np.nan], size=n, p=[0.3, 0.3, 0.3, 0.1])
        fil = np.where(np.isnan(fil), BG, fil)
    else:
        fil = rng.uniform(0, 100, size=n)
        fil[rng.random(n) < 0.1] = BG

    flooded_mask = (rng.random(n) < 0.3).astype(np.float64)
    excluded = rng.random(n) < 0.05

    bdy_pixels = rng.choice(n, size=min(n_boundary, n), replace=False).astype(np.int64)
    if tie_prone:
        bdy_dtf = rng.choice([0.0, 1.0, 2.0], size=len(bdy_pixels)).astype(np.float64)
    else:
        bdy_dtf = rng.uniform(0, 5, size=len(bdy_pixels))
    bdy_fsp = rng.integers(0, n_fsp, size=len(bdy_pixels)).astype(np.int64)
    # a boundary pixel's own dtf must be consistent with a flooded state;
    # not flooded/excluded so it's a legitimate boundary entry itself.
    flooded_mask[bdy_pixels] = 0.0
    excluded[bdy_pixels] = False

    fldht = float(rng.uniform(1, 10))
    mxht0 = float(rng.uniform(5, 50))
    return bdy_pixels, bdy_dtf, bdy_fsp, fil, flooded_mask, excluded, fldht, mxht0


@pytest.mark.parametrize("tie_prone", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_gpu_reduce_matches_numba_kernel_exactly(seed, tie_prone):
    rng = np.random.default_rng(seed)
    n_rows, n_cols = 40, 40
    bdy_pixels, bdy_dtf, bdy_fsp, fil, flooded_mask, excluded, fldht, mxht0 = _random_scenario(
        rng, n_rows, n_cols, n_boundary=60, n_fsp=4, tie_prone=tie_prone
    )

    cpu_px, cpu_fsp, cpu_dtf = _spillover_claims_nb(
        bdy_pixels, bdy_dtf, bdy_fsp, fil, flooded_mask, excluded, BG, fldht, mxht0,
        n_rows, n_cols, _ROW_OFF, _COL_OFF,
    )

    npx_v, fsp_v, order_v, dtf_v, elev_v = _spillover_claims_candidates(
        bdy_pixels, bdy_dtf, bdy_fsp, fil, flooded_mask, excluded, BG, fldht, mxht0,
        n_rows, n_cols, _ROW_OFF, _COL_OFF,
    )
    gpu_px, gpu_fsp, gpu_dtf = _spillover_claims_reduce(
        cp.asarray(npx_v), cp.asarray(fsp_v), cp.asarray(order_v),
        cp.asarray(dtf_v), cp.asarray(elev_v), xp=cp,
    )
    gpu_px, gpu_fsp, gpu_dtf = cp.asnumpy(gpu_px), cp.asnumpy(gpu_fsp), cp.asnumpy(gpu_dtf)

    np.testing.assert_array_equal(gpu_px, cpu_px)
    np.testing.assert_array_equal(gpu_fsp, cpu_fsp)
    np.testing.assert_array_equal(gpu_dtf, cpu_dtf)  # exact -- same float ops, no reduction-order variance


def test_gpu_reduce_matches_numpy_reduce_cross_check():
    """`_spillover_claims_reduce` is xp-generic -- running it with xp=numpy
    is an independent (pure-vectorized, non-numba) reimplementation of the
    same rule, useful as a second witness beyond the numba kernel itself."""
    rng = np.random.default_rng(0)
    n_rows, n_cols = 30, 30
    bdy_pixels, bdy_dtf, bdy_fsp, fil, flooded_mask, excluded, fldht, mxht0 = _random_scenario(
        rng, n_rows, n_cols, n_boundary=40, n_fsp=3, tie_prone=True
    )
    npx_v, fsp_v, order_v, dtf_v, elev_v = _spillover_claims_candidates(
        bdy_pixels, bdy_dtf, bdy_fsp, fil, flooded_mask, excluded, BG, fldht, mxht0,
        n_rows, n_cols, _ROW_OFF, _COL_OFF,
    )
    np_px, np_fsp, np_dtf = _spillover_claims_reduce(npx_v, fsp_v, order_v, dtf_v, elev_v, xp=np)
    gpu_px, gpu_fsp, gpu_dtf = _spillover_claims_reduce(
        cp.asarray(npx_v), cp.asarray(fsp_v), cp.asarray(order_v),
        cp.asarray(dtf_v), cp.asarray(elev_v), xp=cp,
    )
    np.testing.assert_array_equal(cp.asnumpy(gpu_px), np_px)
    np.testing.assert_array_equal(cp.asnumpy(gpu_fsp), np_fsp)
    np.testing.assert_array_equal(cp.asnumpy(gpu_dtf), np_dtf)


def test_gpu_reduce_empty_boundary():
    empty_i = np.empty(0, dtype=np.int64)
    empty_f = np.empty(0, dtype=np.float64)
    px_out, fsp_out, dtf_out = _spillover_claims_reduce(
        cp.asarray(empty_i), cp.asarray(empty_i), cp.asarray(empty_i),
        cp.asarray(empty_f), cp.asarray(empty_f), xp=cp,
    )
    assert len(px_out) == len(fsp_out) == len(dtf_out) == 0


def test_run_segment_gpu_matches_cpu_end_to_end():
    """Same hand-derived 3x3 scenario as test_flood_fill.py's CPU test, run
    through the full `run_segment` with use_gpu=True vs False."""
    n_rows, n_cols = 3, 3
    SOUTH, EAST = 4, 1

    def build_grid():
        fil = np.full((n_rows, n_cols), 99.0, dtype=np.float64)
        fdr = np.full((n_rows, n_cols), EAST, dtype=np.uint8)
        fil[0, 1] = 10.0
        fil[1, 1] = 8.0
        fil[2, 1] = 6.0
        fdr[0, 1] = SOUTH
        fdr[1, 1] = SOUTH
        fdr[2, 1] = SOUTH
        fil[1, 0] = 9.0
        fdr[1, 0] = EAST
        fil[1, 2] = 12.0
        fdr[1, 2] = SOUTH
        fil[2, 2] = 20.0
        fdr[2, 2] = SOUTH
        return fil.ravel(), fdr.ravel()

    fil, fdr = build_grid()
    kwargs = dict(
        seg_id=1, start_pixel=1, end_pixel=7, length=3, fil=fil, fdr=fdr,
        fldmn=0.1, dh=5, fldmx=5, mxht0=0.0, ssflg=False, bg=BG,
        n_rows=n_rows, n_cols=n_cols,
    )
    cpu_entries = run_segment(**kwargs, use_gpu=False)
    gpu_entries = run_segment(**kwargs, use_gpu=True)

    cpu_set = {(e.fsp, e.pixel, e.dtf) for e in cpu_entries}
    gpu_set = {(e.fsp, e.pixel, e.dtf) for e in gpu_entries}
    assert cpu_set == gpu_set
    assert [(e.fsp, e.pixel) for e in cpu_entries] == [(e.fsp, e.pixel) for e in gpu_entries]
