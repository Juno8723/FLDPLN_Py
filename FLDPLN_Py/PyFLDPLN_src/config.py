"""YAML configuration loading for the PyFLDPLN_src pipeline.

Replaces the hardcoded user-edited constants at the top of the original
MATLAB scripts (fldpln_wrapper_v8.m, create_fldpln_map.m) with a single
YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

_VALID_SEGMENTATION_METHODS = {"auto", "seedpoints"}


@dataclass
class RunConfig:
    """Compute resources to use. Unlike the original MATLAB 'hd'/'ram0'/'ram'
    modes (which were just an internal memory-access strategy), that
    distinction isn't user-facing here -- the flood-fill core picks an
    in-RAM-vs-memmap array strategy on its own. What the user does need to
    control is how many CPU workers to use for segment-level parallelism, and
    whether to use a GPU -- both relevant when scaling up on an HPC cluster.
    """

    n_workers: Optional[int] = None  # None = os.cpu_count()
    use_gpu: bool = False  # enable GPU-accelerated flood-fill backend, if available

    def __post_init__(self) -> None:
        if self.n_workers is not None and self.n_workers < 1:
            raise ValueError("run.n_workers must be a positive integer or null")


@dataclass
class PathsConfig:
    home_dir: Path
    dem: str = "bil/dem.bil"
    fil: str = "bil/fil.bil"
    fdr: str = "bil/fdr.bil"
    fac: str = "bil/fac.bil"
    segment_raster: str = "bil/str_segid.bil"
    seg_info: str = "mat/seg_info.parquet"
    seed_points_shapefile: Optional[str] = None
    # Optional CRS override (e.g. "EPSG:26915") for all raster/vector I/O in
    # this run. Different AOIs commonly sit in different projections (e.g.
    # UTM zone per region) -- null (default) trusts whatever CRS is already
    # embedded in the input .bil files' .prj sidecars; set this explicitly
    # when that's missing, wrong, or you just want it stated up front.
    crs: Optional[str] = None

    def _resolve(self, rel: str) -> Path:
        return self.home_dir / rel

    @property
    def dem_path(self) -> Path:
        return self._resolve(self.dem)

    @property
    def fil_path(self) -> Path:
        return self._resolve(self.fil)

    @property
    def fdr_path(self) -> Path:
        return self._resolve(self.fdr)

    @property
    def fac_path(self) -> Path:
        return self._resolve(self.fac)

    @property
    def segment_raster_path(self) -> Path:
        return self._resolve(self.segment_raster)

    @property
    def seg_info_path(self) -> Path:
        return self._resolve(self.seg_info)

    @property
    def seed_points_shapefile_path(self) -> Optional[Path]:
        if self.seed_points_shapefile is None:
            return None
        return self._resolve(self.seed_points_shapefile)


@dataclass
class SegmentationConfig:
    method: str = "auto"  # auto | seedpoints
    stream_threshold_sqmi: float = 70.0  # strthr
    break_threshold_sqmi: float = 25.0  # facthr
    max_segment_length_mi: float = 5.0  # seglen

    def __post_init__(self) -> None:
        if self.method not in _VALID_SEGMENTATION_METHODS:
            raise ValueError(
                f"segmentation.method must be one of {sorted(_VALID_SEGMENTATION_METHODS)}, "
                f"got {self.method!r}"
            )


@dataclass
class FloodConfig:
    fldmn: float = 0.03  # US standard minimum flood-stage depth
    dh: float = 3.0
    fldmx: float = 36.0
    max_height_cap: float = 0.0  # 0 = no cap (mxht)
    steady_state: bool = True  # ssflg


@dataclass
class OutputConfig:
    segment_files_dir: str = "segment_files/"
    map_output: str = "bil/fldpln_map.bil"

    def resolve(self, home_dir: Path) -> "OutputConfig":
        return OutputConfig(
            segment_files_dir=str(home_dir / self.segment_files_dir),
            map_output=str(home_dir / self.map_output),
        )


@dataclass
class PyfldplnConfig:
    run: RunConfig = field(default_factory=RunConfig)
    paths: PathsConfig = field(default_factory=lambda: PathsConfig(home_dir=Path(".")))
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    flood: FloodConfig = field(default_factory=FloodConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        if self.segmentation.method == "seedpoints" and self.paths.seed_points_shapefile is None:
            raise ValueError(
                "paths.seed_points_shapefile is required when segmentation.method == 'seedpoints'"
            )


def load_config(path: str | Path) -> PyfldplnConfig:
    """Load and validate a PyFLDPLN_src YAML config file."""
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    run = RunConfig(**raw.get("run", {}))

    paths_raw = dict(raw.get("paths", {}))
    if "home_dir" not in paths_raw:
        raise ValueError("paths.home_dir is required in config")
    paths_raw["home_dir"] = Path(paths_raw["home_dir"])
    paths = PathsConfig(**paths_raw)

    segmentation = SegmentationConfig(**raw.get("segmentation", {}))
    flood = FloodConfig(**raw.get("flood", {}))
    output = OutputConfig(**raw.get("output", {}))

    return PyfldplnConfig(
        run=run, paths=paths, segmentation=segmentation, flood=flood, output=output
    )
