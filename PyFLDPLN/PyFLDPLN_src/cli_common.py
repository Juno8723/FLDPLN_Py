"""Helpers shared between cli_segmentation.py and cli_library_reorganization.py."""

from __future__ import annotations

from typing import Optional


def _resolve_seg_ids(
    n_segments: int,
    seg_ids: Optional[list[int]],
    exclude_seg_ids: Optional[list[int]],
) -> Optional[list[int]]:
    """Apply an optional exclude list on top of an optional include list.

    Replaces the original's shapefile-based `seg_list` mechanism
    (fldpln_wrapper_v8.m reading GRID_CODE from a hand-edited, dissolved
    'str_segid...mod4lib.shp' to drop segments a human decided not to
    process -- e.g. a stray, disconnected tiny segment) with a plain CLI
    flag: rather than round-tripping through GIS (raster-to-polyline,
    dissolve, manually delete features, reload), the user just looks up the
    offending segment's ID (e.g. by clicking its pixel in str_segid.bil) and
    passes it here.
    """
    if seg_ids is None:
        seg_ids = list(range(1, n_segments + 1))
    if exclude_seg_ids:
        exclude_set = set(exclude_seg_ids)
        seg_ids = [s for s in seg_ids if s not in exclude_set]
    return seg_ids


def parse_seg_id_list(s: Optional[str]) -> Optional[list[int]]:
    """Parse a `--segments`/`--exclude-segments` comma-separated CLI value."""
    return [int(s_) for s_ in s.split(",")] if s else None
