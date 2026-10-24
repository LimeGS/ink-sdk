"""Runtime contracts shared by the resolver, renderers and adapters."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

IssueLevel = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    level: IssueLevel
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.level == "error"


@dataclass(frozen=True)
class RenderSpec:
    """How to turn a tifxyz mesh into a surface volume.

    ``scale=1`` is the CT's native level-0 sampling. The model scale is reached
    later by ``prepare``; it is never baked into the renderer.
    """

    scale: float = 1.0
    num_slices: int | None = None
    slice_step: int = 1
    orientation: str = "auto"  # auto | inward | outward
    umbilicus: Path | None = None
    roi: tuple[int, int, int, int] | None = None  # y0,y1,x0,x1 at output map scale
    renderer: Path | None = None


@dataclass(frozen=True)
class Segment:
    id: str
    root: Path
    surface_volume: Path | None = None
    format: str | None = None  # zarr | npy | tiff-stack
    surface_mesh: Path | None = None
    ct: Path | None = None
    pixel_um: tuple[float, float, float] | None = None  # z,y,x
    depth: int | None = None
    dtype: str | None = None
    direction: str | None = None  # forward | reverse | unknown
    render: RenderSpec | None = None


@dataclass(frozen=True)
class RunContext:
    device: str = "auto"
    workdir: Path | None = None
    cache_dir: Path | None = None
    renderer: Path | None = None
    render_cache: Path | None = None
    render_threads: int = 1
    models_dir: Path | None = None
    villa: Path | None = None
    python: str | None = None
    overwrite: bool = False
    dry_run: bool = False
    deterministic: bool = False
    progress: Callable[[str], None] | None = None
    max_render_gb: float = 64.0
    on_degenerate: str = "fail"  # fail | warn
    resample_from_um: float | None = None
    output_3d: bool = False
    amp: str | None = None
    roi: tuple[int, int, int, int] | None = None
    normal: str = "auto"  # auto | inward | outward
    umbilicus: Path | None = None
    primary: str | None = None
    orient_report: dict[str, Any] | None = None  # from `ink-sdk orient`; used only when primary is None

    def say(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)


@dataclass(frozen=True)
class RenderedVolume:
    path: Path
    shape_zyx: tuple[int, int, int]
    pixel_um: tuple[float, float, float]
    orientation: dict[str, Any]
    receipt: dict[str, Any]
    cache_hit: bool = False


@dataclass(frozen=True)
class ScalePlan:
    source_um: float
    target_um: float
    mode: str  # native | pool | resample
    factors_zyx: tuple[int, int, int] = (1, 1, 1)
    resample_from_um: float | None = None
    note: str = ""

    @property
    def output_pixel_um(self) -> float:
        if self.mode == "pool":
            return float(self.source_um * self.factors_zyx[0])
        if self.mode == "resample":
            return float(self.target_um)
        return float(self.source_um)


@dataclass(frozen=True)
class PreparedInput:
    path: Path
    array: np.ndarray
    pixel_um: tuple[float, float, float]
    source: Path
    scale_plan: ScalePlan
    roi: tuple[int, int, int, int] | None = None
    origin_px: tuple[int, int] = (0, 0)
    valid: np.ndarray | None = None
    receipt: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawOutput:
    maps: dict[str, np.ndarray]
    valid: np.ndarray | None = None
    volume: np.ndarray | None = None
    volume_attrs: dict[str, Any] = field(default_factory=dict)
    quantisation: dict[str, Any] = field(default_factory=dict)
    receipt: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InkMap:
    array: np.ndarray
    valid: np.ndarray | None
    space: str
    pixel_um: tuple[float, float]
    origin_px: tuple[int, int]
    direction: str
    model_id: str
    checkpoint_sha256: str
    no_ink_floor: float
    quantisation: dict[str, Any]
    receipt: dict[str, Any]

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.array.shape)  # type: ignore[return-value]


@dataclass(frozen=True)
class InkVolume:
    path: Path
    dims: tuple[str, ...]
    pixel_um: tuple[float, float, float]
    evaluated_z_interval: tuple[int, int] | None
    projection: str | None
    attrs: dict[str, Any]


@dataclass(frozen=True)
class InkResult:
    maps: dict[str, InkMap]
    primary: str
    volume: InkVolume | None
    out_dir: Path | None
    receipt_path: Path | None
    receipt: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_map(self) -> InkMap:
        return self.maps[self.primary]


def jsonable(value: Any) -> Any:
    """Make dataclasses, paths, numpy scalars and arrays safe for json.dumps."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
