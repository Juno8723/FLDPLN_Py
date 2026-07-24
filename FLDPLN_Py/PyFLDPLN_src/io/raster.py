"""BIL raster I/O, replacing readbilheader.m / scanbil.m.

Reading and writing standard ESRI BIL (.bil + .hdr) rasters is delegated to
rasterio (GDAL's EHdr driver) instead of hand-parsing header text, as MATLAB
did. Arrays are returned row-major (shape = (rows, cols)), matching the BIL
band-interleaved-by-line pixel order the original code relied on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from affine import Affine


@dataclass
class RasterInfo:
    rows: int
    cols: int
    pixel_size_x: float
    pixel_size_y: float
    transform: Affine
    crs: object
    nodata: Optional[float]
    dtype: str


def read_raster(path: str | Path) -> tuple[np.ndarray, RasterInfo]:
    """Read a single-band raster (e.g. dem/fil/fdr/fac .bil) into a 2-D array."""
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        info = RasterInfo(
            rows=ds.height,
            cols=ds.width,
            pixel_size_x=ds.transform.a,
            pixel_size_y=-ds.transform.e,
            transform=ds.transform,
            crs=ds.crs,
            nodata=ds.nodata,
            dtype=arr.dtype.name,
        )
    return arr, info


def read_raster_info(path: str | Path) -> RasterInfo:
    """Read only the header/spatial info, without loading pixel data."""
    with rasterio.open(path) as ds:
        return RasterInfo(
            rows=ds.height,
            cols=ds.width,
            pixel_size_x=ds.transform.a,
            pixel_size_y=-ds.transform.e,
            transform=ds.transform,
            crs=ds.crs,
            nodata=ds.nodata,
            dtype=ds.dtypes[0],
        )


def open_memmap(path: str | Path, mode: str = "r") -> tuple[np.memmap, RasterInfo]:
    """Open a raw ``.bil`` file as a disk-backed 1-D memmap (row-major, flat).

    Lets the flood-fill core avoid loading the full DEM/FIL/FDR array into
    RAM when needed (equivalent role to the original MATLAB 'hd'/'ram0'
    modes), without exposing that choice as a user-facing config option.
    """
    info = read_raster_info(path)
    dtype = np.dtype(info.dtype)
    arr = np.memmap(path, dtype=dtype, mode=mode, shape=(info.rows * info.cols,))
    return arr, info


def write_raster(
    path: str | Path,
    array: np.ndarray,
    like: RasterInfo,
    dtype: Optional[str] = None,
    nodata: Optional[float] = None,
) -> None:
    """Write a 2-D array to a new ESRI BIL raster, copying spatial info from ``like``."""
    out_dtype = dtype or array.dtype.name
    profile = {
        "driver": "EHdr",
        "height": like.rows,
        "width": like.cols,
        "count": 1,
        "dtype": out_dtype,
        "crs": like.crs,
        "transform": like.transform,
        "nodata": nodata if nodata is not None else like.nodata,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(array.astype(out_dtype), 1)
