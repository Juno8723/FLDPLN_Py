# PyFLDPLN

Python port of the FLDPLN floodplain-mapping model (Kastens 2008). Given a
filled DEM, D8 flow direction, and D8 flow accumulation for your area of
interest, this runs the full pipeline end to end: per-segment flood-fill
simulation → map-coordinate library reorganization + geographic tiling →
gauge-driven event flood map. No ArcGIS or MATLAB required; runs on CPU by
default, with an optional GPU-accelerated backend (see "Known limitations").

This folder is a self-contained release snapshot — everything needed to run
the pipeline, nothing else (no source data, no development history).

## 1. Prerequisites

- **Python 3.11+**
- Your own **DEM, FIL (filled DEM), FDR (D8 flow direction), FAC (D8 flow
  accumulation)** rasters for the AOI, all at the same extent, resolution,
  and CRS. Any format `rasterio`/GDAL can open (`.bil`, GeoTIFF, ...).
  - FDR must use **ESRI D8 codes**: `32 64 128 / 16 x 1 / 8 4 2` (NW/N/NE/W/E/SW/S/SE).
    This is what ArcGIS's own Flow Direction, WhiteboxTools' `D8Pointer
    (--esri_pntr)`, and pysheds' `dirmap` (set to the ESRI convention) all
    produce — pick whichever depression-fill/flow-direction/flow-accumulation
    tool you already have; this package only needs the resulting rasters.

## 2. Set up the environment

Using conda/mamba (recommended):

```bash
conda create -n pyfldpln python=3.11
conda activate pyfldpln
```

...or a plain venv:

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

## 3. Install

From this folder:

```bash
pip install -e .
```

This installs the `PyFLDPLN_src` console command and all required
dependencies: `numpy`, `pandas`, `pyarrow`, `rasterio`, `geopandas`,
`pyyaml`, `numba`, `matplotlib`, `shapely`, `scipy`.

> **GPU**: `pip install -e ".[gpu]"` adds `cupy` for the optional GPU-accelerated
> flood-fill backend (`run.use_gpu: true`). It's been validated bit-identical to
> the CPU path (80/80 segments, 487M+ output rows, zero mismatches) but is
> currently *slower* than CPU at production scale — leave `use_gpu: false`
> unless you're experimenting; the CPU (numba-JIT) path is the recommended one.

## 4. Configure your AOI

Copy the template and edit it:

```bash
cp configs/config.example.yaml configs/my_aoi.yaml
```

At minimum, set `paths.home_dir` and `paths.dem/fil/fdr/fac` to your own
files. Every other path in the config is resolved relative to `home_dir`.
See the comments in `config.example.yaml` (or `PyFLDPLN_src/config.py` for
the full schema) for what every field does.

## 5. Run the pipeline

Five commands, in this order. Each reads the previous stage's output via the
same config file.

```bash
# 1. Delineate stream segments from FDR/FAC
PyFLDPLN_src build-segments configs/my_aoi.yaml

# 2. Per-segment flood-fill simulation (the slow step -- scales with
#    run.n_workers and total pixel count across all segments)
PyFLDPLN_src run-model configs/my_aoi.yaml

# 3. Reorganize per-segment output into a map-coordinate library
#    (FSP table, stream order, FSP-FPP relations)
PyFLDPLN_src build-library configs/my_aoi.yaml

# 4. Bucket the library into geographic tiles for fast lookup
PyFLDPLN_src tile-library configs/my_aoi.yaml

# 5. Build an event-specific flood map from gauge readings
PyFLDPLN_src gauge-map configs/my_aoi.yaml gauges.csv --snap-dist 350
```

`gauges.csv` needs `x`, `y` (in the library's CRS) and `stage_elevation`
columns (see `--x-field`/`--y-field` if your columns are named differently).
`--snap-dist` is the max distance (in the library's CRS units, e.g. meters)
a gauge may sit from the nearest flood-source point to be used.

Step 5's output defaults to `<library_dir>/gauge_flood_map.tif`; override
with `--out`.

Two optional extras, not required for the gauge map:

```bash
# merge per-segment flood-fill output into one whole-basin depth-to-flood raster
PyFLDPLN_src create-map configs/my_aoi.yaml

# run any stage on a subset of segments (both build-library and run-model support this)
PyFLDPLN_src run-model configs/my_aoi.yaml --segments 1,2,3
PyFLDPLN_src run-model configs/my_aoi.yaml --exclude-segments 19
```

## 6. Output layout

Relative to `home_dir`, using the paths in `config.example.yaml`:

```
outputs/
  mat/seg_info.parquet          # per-segment metadata (from build-segments)
  bil/str_segid.bil             # segment ID raster
  segment_files/                # raw per-segment flood-fill output (from run-model)
  library/
    fsp_info.csv                # flood-source points: coordinates, elevation, stream order
    segment_info.csv
    stream_order_info.csv
    CellSizeSpatialReference.json
    relations/seg{N}.parquet    # per-segment FSP-FPP relations (from build-library)
    FLDPLN_tiled_*.snz          # tiled relations (from tile-library)
    FLDPLN_tiled_tile_index.csv
    FLDPLN_tiled_fsp_index.csv
    gauge_flood_map.tif         # final event flood map (from gauge-map)
```

## Known limitations

- **Flood-source attribution ties**: when two upstream points reach the same
  pixel at the exact same depth, which one gets *credited* as that pixel's
  source depends on internal processing order and can vary slightly run to
  run (parallel worker scheduling isn't deterministic). This never affects
  computed depth values — only a small number of source-attribution labels,
  typically well under 1% of points. See `PyFLDPLN_src/model/flood_fill.py`
  and `segmentation/stream_order.py` for where this shows up.
- **GPU acceleration** (`run.use_gpu` / the `cupy` backend) is correctness-validated
  (bit-identical to CPU) but currently *slower* than CPU at production scale —
  the CPU (numba-JIT) path remains the recommended default. GPU work is
  ongoing; a future release may change this.
