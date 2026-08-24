"""Shapefile field reading, replacing readshapefield.m.

Field name lookups are case-insensitive, mirroring the original wrapper's
try/except fallback between 'GRID_CODE' and 'grid_code' column names.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np


def _resolve_column(columns: list[str], field: str) -> str:
    for col in columns:
        if col.lower() == field.lower():
            return col
    raise KeyError(f"field {field!r} not found among columns {columns}")


def read_shapefile_fields(path: str | Path, fields: list[str]) -> np.ndarray:
    """Read numeric fields (or 'X'/'Y' point coordinates) from a shapefile.

    Returns an (n_records, len(fields)) float array, mirroring readshapefield.m.
    """
    gdf = gpd.read_file(path)
    columns = list(gdf.columns)
    out = np.zeros((len(gdf), len(fields)), dtype=float)
    for i, fld in enumerate(fields):
        if fld in ("X", "Y"):
            centroids = gdf.geometry
            out[:, i] = centroids.x.to_numpy() if fld == "X" else centroids.y.to_numpy()
        else:
            col = _resolve_column(columns, fld)
            out[:, i] = gdf[col].to_numpy(dtype=float)
    return out
