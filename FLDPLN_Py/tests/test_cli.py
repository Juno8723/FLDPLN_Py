"""Unit tests for the --segments/--exclude-segments resolution logic."""

from PyFLDPLN_src.cli import _resolve_seg_ids


def test_resolve_seg_ids_defaults_to_all():
    assert _resolve_seg_ids(5, None, None) == [1, 2, 3, 4, 5]


def test_resolve_seg_ids_applies_exclude_to_default_all():
    assert _resolve_seg_ids(5, None, [2, 4]) == [1, 3, 5]


def test_resolve_seg_ids_applies_exclude_to_explicit_include_list():
    assert _resolve_seg_ids(5, [1, 2, 3], [2]) == [1, 3]


def test_resolve_seg_ids_no_exclude_returns_include_list_unchanged():
    assert _resolve_seg_ids(5, [3, 1], None) == [3, 1]
