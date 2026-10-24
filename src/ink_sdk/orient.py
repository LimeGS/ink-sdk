"""Label-free depth-orientation check for a rendered surface volume.

Two measurements, both without labels or an umbilicus:

1. Sheet side (mesh only). A papyrus sheet in a rolled scroll is concave towards the
   scroll centre. The median sagitta of the tifxyz grid, measured along the SDK mesh
   normal ``n = d/drow x d/dcol``, tells whether ``+n`` points to the concave side
   (inward) or away from it (outward).
2. Stack direction (mesh + CT). The CT is sampled at ``p + t*n`` in a window of the
   mesh and each layer of the surface volume is matched to the offset ``t`` it
   correlates with best. The slope says whether the stored layers grow along ``+n``
   or ``-n``.

Together they say which face of the sheet layer 0 is on. For the 9 um ink models the
SDK ships (ink_9um seed43 and Reader v2), the map that sees ink is the one whose stack
runs from the OUTER face to the INNER face; that rule was calibrated on PHerc1447 w058
and held 36/36 (segment, model) pairs on PHerc0139, PHerc0814 and PHerc1667 under
pre-registered tests (see the ink-or-hallucination record). For other adapters the
report states the geometry but recommends no map.

The check fails closed: if the sheet is too flat to decide, or the CT does not match
the layers well, it says "undetermined" and recommends nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .errors import MissingDependencyError, ValidationError

SCHEMA = "ink-sdk.orient.v1"
MIN_SAGITTA_FRAC = 0.02   # |median sagitta| >= 2% of D to decide the sheet side
MIN_CT_CORR = 0.5         # median best layer-to-CT correlation
MIN_SLOPE = 0.5           # |offset per layer|, in CT voxels

# adapter -> which face layer 0 must be on for the map that sees ink
CALIBRATED = {
    "ink9um": {
        "layer0_should_be": "outer",
        "calibration": "PHerc1447 w058 (8.64 um), koine ink_9um/Reader v2/Reader V3 and ink-sdk seed43",
        "validation": "36/36 (segment, model) pairs, PHerc0139 w035-w043, PHerc0814 46527, PHerc1667 w013-w031",
    },
}


def _scipy():
    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise MissingDependencyError("orient needs scipy; install ink-sdk[render]") from exc
    return ndimage


def load_mesh(mesh: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (P[h, w, 3] in xyz, valid[h, w], grid scale) from a tifxyz directory."""
    import tifffile

    mesh = Path(mesh)
    try:
        P = np.stack([tifffile.imread(mesh / f"{c}.tif").astype(np.float64) for c in "xyz"], -1)
    except FileNotFoundError as exc:
        raise ValidationError(f"tifxyz needs x.tif, y.tif and z.tif: {mesh}") from exc
    if P.ndim != 3 or min(P.shape[:2]) < 3:
        raise ValidationError(f"mesh x/y/z must be compatible 2D grids: {mesh}")
    valid = np.isfinite(P).all(-1) & (P > 0).all(-1)
    scale = 1.0
    meta = mesh / "meta.json"
    if meta.is_file():
        value = json.loads(meta.read_text(encoding="utf-8")).get("scale")
        if isinstance(value, (list, tuple)) and value:
            scale = float(value[0])
        elif isinstance(value, (int, float)):
            scale = float(value)
    return P, valid, scale


def mesh_normals(P: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals ``d/drow x d/dcol`` (the renderer's convention) and their validity."""
    dr = np.zeros_like(P)
    dc = np.zeros_like(P)
    dr[1:-1] = P[2:] - P[:-2]
    dc[:, 1:-1] = P[:, 2:] - P[:, :-2]
    ok = valid.copy()
    ok[1:-1] &= valid[2:] & valid[:-2]
    ok[:, 1:-1] &= valid[:, 2:] & valid[:, :-2]
    ok[[0, -1], :] = False
    ok[:, [0, -1]] = False
    n = np.cross(dr, dc)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    ok &= norm[..., 0] > 0
    return n / np.maximum(norm, 1e-12), ok


def sheet_side(P: np.ndarray, valid: np.ndarray, scale: float, *, distance_vx: float = 1000.0,
               samples: int = 20000, seed: int = 0) -> dict[str, Any]:
    """Which way the normal points relative to the sheet's concave side."""
    n, ok = mesh_normals(P, valid)
    H, W = valid.shape
    k = max(2, int(round(distance_vx * scale)))
    ii, jj = np.nonzero(ok)
    if len(ii) == 0:
        raise ValidationError("mesh has no valid stencil for the orientation check")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ii), min(samples, len(ii)), replace=False)
    ii, jj = ii[pick], jj[pick]
    sag = []
    for di, dj in ((k, 0), (-k, 0), (0, k), (0, -k)):
        i2, j2 = ii + di, jj + dj
        m = (i2 >= 0) & (i2 < H) & (j2 >= 0) & (j2 < W)
        m[m] &= valid[i2[m], j2[m]]
        sag.append(((P[i2[m], j2[m]] - P[ii[m], jj[m]]) * n[ii[m], jj[m]]).sum(-1))
    sag = np.concatenate(sag)
    dist = k / scale
    if sag.size == 0:
        return {"decided": False, "normal_points": "undetermined", "reason": "mesh smaller than the test distance",
                "distance_vx": dist, "pairs": 0}
    med = float(np.median(sag))
    decided = abs(med) >= MIN_SAGITTA_FRAC * dist
    return {
        "decided": bool(decided),
        # sheet bends towards +n  ->  +n points to the concave side, i.e. towards the scroll centre
        "normal_points": ("inward" if med > 0 else "outward") if decided else "undetermined",
        "median_sagitta_vx": med,
        "distance_vx": dist,
        "pairs": int(sag.size),
        "threshold_vx": MIN_SAGITTA_FRAC * dist,
    }


def _open_ct(ct: str | Path, level: str):
    import zarr

    text = str(ct)
    try:
        node = zarr.open(text, mode="r")
    except Exception as exc:
        raise ValidationError(f"cannot open CT {ct}: {exc}") from exc
    if hasattr(node, "shape"):
        return node
    if level in node:
        return node[level]
    raise ValidationError(f"CT {ct} has no array {level!r}")


def _open_layers(volume: Path):
    import zarr

    volume = Path(volume)
    if volume.is_dir() and ((volume / "0" / ".zarray").is_file() or (volume / "0" / "zarr.json").is_file()):
        return zarr.open(str(volume), mode="r")["0"]
    from .io import open_volume_array

    return open_volume_array(volume)


def stack_direction(P: np.ndarray, valid: np.ndarray, volume: Path, ct: str | Path, *, ct_level: str = "0",
                    mesh_divisor: float = 1.0, win: int = 48, T: int = 16,
                    at: tuple[int, int] | None = None) -> dict[str, Any]:
    """Do the stored layers grow along +n or -n? Measured against the CT."""
    ndi = _scipy()
    P = np.where(valid[..., None], P / mesh_divisor, P)
    n, ok = mesh_normals(P, valid)
    H, W = valid.shape
    if at is None:
        full = ndi.uniform_filter(ok.astype(np.float32), win) > 0.999
        ii, jj = np.nonzero(full)
        if len(ii) == 0:
            return {"decided": False, "reason": f"no fully valid {win}x{win} window on the mesh"}
        c = np.argmin((ii - H / 2) ** 2 + (jj - W / 2) ** 2)
        i0, j0 = int(ii[c] - win // 2), int(jj[c] - win // 2)
    else:
        i0, j0 = at
    p = P[i0:i0 + win, j0:j0 + win]
    nw = n[i0:i0 + win, j0:j0 + win]
    reach = T * np.abs(nw)
    lo = np.floor(np.minimum((p - reach).reshape(-1, 3).min(0), (p + reach).reshape(-1, 3).min(0)) - 2).astype(int)
    hi = np.ceil(np.maximum((p - reach).reshape(-1, 3).max(0), (p + reach).reshape(-1, 3).max(0)) + 2).astype(int)
    lo = np.maximum(lo, 0)
    arr = _open_ct(ct, ct_level)
    box = np.asarray(arr[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]).astype(np.float32)
    ts = np.arange(-T, T + 1)
    S = []
    for t in ts:
        q = p + t * nw - lo
        S.append(ndi.map_coordinates(box, [q[..., 2].ravel(), q[..., 1].ravel(), q[..., 0].ravel()],
                                     order=1).reshape(win, win))
    S = np.array(S)
    layers = _open_layers(volume)
    nl, Hr, Wr = layers.shape
    fy, fx = Hr / H, Wr / W
    ys = slice(int(round(i0 * fy)), int(round((i0 + win) * fy)))
    xs = slice(int(round(j0 * fx)), int(round((j0 + win) * fx)))
    L = []
    for k in range(nl):
        im = np.asarray(layers[k, ys, xs]).astype(np.float32)
        L.append(ndi.zoom(im, (win / im.shape[0], win / im.shape[1]), order=1)[:win, :win])
    L = np.array(L)

    def z(x):
        x = x.reshape(len(x), -1)
        x = x - x.mean(1, keepdims=True)
        return x / (x.std(1, keepdims=True) + 1e-6)

    C = z(L) @ z(S).T / (win * win)
    best = ts[C.argmax(1)]
    slope = float(np.polyfit(np.arange(nl), best, 1)[0]) if nl > 1 else 0.0
    corr = float(np.median(C.max(1)))
    decided = corr >= MIN_CT_CORR and abs(slope) >= MIN_SLOPE
    return {
        "decided": bool(decided),
        "layers_grow_along": ("+n" if slope > 0 else "-n") if decided else "undetermined",
        "slope_vx_per_layer": slope,
        "median_best_corr": corr,
        "window_grid": [i0, i0 + win, j0, j0 + win],
        "render_px_per_grid": [fy, fx],
        "ct": str(ct), "ct_level": ct_level, "mesh_divisor": mesh_divisor,
        "thresholds": {"median_best_corr": MIN_CT_CORR, "abs_slope": MIN_SLOPE},
    }


def orient(mesh: Path, volume: Path, ct: str | Path, *, adapter: str | None = None, ct_level: str = "0",
           mesh_divisor: float = 1.0, distance_vx: float = 1000.0, win: int = 48, T: int = 16) -> dict[str, Any]:
    """Full report: sheet side, stack direction, which face layer 0 is on, recommended map."""
    P, valid, scale = load_mesh(mesh)
    side = sheet_side(P, valid, scale * mesh_divisor, distance_vx=distance_vx)
    stack = stack_direction(P, valid, volume, ct, ct_level=ct_level, mesh_divisor=mesh_divisor, win=win, T=T)
    layer0 = "undetermined"
    if side["decided"] and stack["decided"]:
        grows_outward = (stack["layers_grow_along"] == "+n") == (side["normal_points"] == "outward")
        layer0 = "inner" if grows_outward else "outer"
    rec, why = None, None
    rule = CALIBRATED.get(adapter or "")
    if layer0 == "undetermined":
        why = "geometry undetermined; nothing recommended"
    elif rule is None:
        why = f"no calibrated rule for adapter {adapter!r}; geometry reported only"
    else:
        rec = "forward" if layer0 == rule["layer0_should_be"] else "reverse"
    return {
        "schema": SCHEMA,
        "mesh": str(mesh), "surface_volume": str(volume),
        "sheet_side": side,
        "stack": stack,
        "layer0_face": layer0,
        "adapter": adapter,
        "recommended_primary": rec,
        "reason": why,
        "rule": rule,
        "non_claims": ["Picks a depth direction only. Says nothing about whether the map shows ink or letters."],
    }
