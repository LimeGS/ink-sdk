"""Z-score ensemble of battery cells, after Erwin Nieuwlaar's PHerc0343 recipe.

Source: ``ink/scripts/ensemble_maps.py`` in github.com/Nieuwlaar/pherc343-first-letters (MIT).
For each depth width (``step``) of a battery:

  valid pixels  CT > 0 in every layer of the thick surface volume.
  map values    ink9um (VILLA) maps: the SDK's p = v / 255 is mapped back to Erwin's clip((v - 64) / 128, 0, 1);
                other adapters: p as written. Every map is resized bilinearly to the surface-volume grid.
  statistics    per profile, mean and SD pooled over the READING side at every centre, valid pixels only.
                The reading side is chosen by the caller (``reverse`` or ``forward``); the other side (the blind
                side) is shown with the reading side's statistics, never with its own.
  ensemble      mean of the per-profile z-scores at centres where every profile ran.
  stretch       ONE stretch per step: p1..p99.5 of the reading-side ensemble over a fixed grid (every 8th row and
                column, starting at 4) of valid pixels at all those centres. Never per centre, so brightness
                differences between depths stay visible.

Not included: Erwin's tile-phase merge (each map run twice with shifted tiles and the halves stitched), which is
specific to his two-part PHerc0343 mesh. Images are in render orientation (no rotation).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

P_LO, P_HI = 1.0, 99.5
SUB = 8
SIDES = ("forward", "reverse")
_FILES = {"forward": "probability.npy", "reverse": "probability_reverse.npy"}


def resize_bilinear(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear resize with pixel-centre alignment (as cv2.INTER_LINEAR for upsampling)."""
    a = np.asarray(a, dtype=np.float32)
    if a.shape == tuple(shape):
        return a

    def axis(n_in: int, n_out: int):
        x = (np.arange(n_out, dtype=np.float64) + 0.5) * n_in / n_out - 0.5
        x = np.clip(x, 0, n_in - 1)
        i0 = np.floor(x).astype(int)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, (x - i0).astype(np.float32)

    y0, y1, fy = axis(a.shape[0], shape[0])
    x0, x1, fx = axis(a.shape[1], shape[1])
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    return (top * (1 - fy[:, None]) + bot * fy[:, None]).astype(np.float32)


def map_values(p: np.ndarray, adapter: str) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    if adapter == "ink9um":
        return np.clip((p * 255.0 - 64.0) / 128.0, 0.0, 1.0)
    return p


def load_side(cell_dir: Path, side: str, adapter: str, shape: tuple[int, int]) -> np.ndarray | None:
    path = Path(cell_dir) / _FILES[side]
    if not path.is_file():
        return None
    return resize_bilinear(map_values(np.load(path), adapter), shape)


def stretch8(m: np.ndarray, lo: float, hi: float, valid: np.ndarray) -> np.ndarray:
    b = (np.clip((m - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    b[~valid] = 0
    return b


def ensemble_battery(
    thick,
    results: Sequence[dict],
    adapters: dict[str, str],
    out_dir: Path,
    side: str,
) -> dict:
    """Write ensemble/<step>/ensemble_<side>_c<centre>.png (+ _blind) and return the statistics."""
    from .battery import png_bytes  # local import: battery imports this module

    if side not in SIDES:
        raise ValueError(f"ensemble side must be forward or reverse, not {side!r}")
    blind = "forward" if side == "reverse" else "reverse"
    n, h, w = (int(x) for x in thick.shape)
    valid = np.ones((h, w), bool)
    for r0 in range(0, h, 256):
        valid[r0:r0 + 256] = (np.asarray(thick[:, r0:r0 + 256, :]) > 0).all(0)
    grid = np.zeros_like(valid)
    grid[SUB // 2::SUB, SUB // 2::SUB] = True
    grid &= valid

    ok = [r for r in results if r.get("ok")]
    report: dict = {"reading_side": side, "blind_side": blind, "valid_px": int(valid.sum()), "steps": {},
                    "source": "Erwin Nieuwlaar, pherc343-first-letters ink/scripts/ensemble_maps.py (MIT)",
                    "not_included": "tile-phase merge"}
    for step in sorted({r["step"] for r in ok}):
        rows = [r for r in ok if r["step"] == step]
        profiles = sorted({r["profile"] for r in rows})
        stats: dict[str, dict] = {}
        for pid in profiles:
            s1 = s2 = 0.0
            count = 0
            centres = []
            for r in rows:
                if r["profile"] != pid:
                    continue
                m = load_side(Path(r["dir"]), side, adapters[pid], (h, w))
                if m is None:
                    continue
                v = m[valid].astype(np.float64)
                s1 += float(v.sum()); s2 += float(np.square(v).sum()); count += v.size
                centres.append(r["centre"])
            if count:
                mu = s1 / count
                stats[pid] = {"mean": mu, "sd": math.sqrt(max(s2 / count - mu * mu, 0.0)) + 1e-6,
                              "centres": sorted(centres), "n_px": count}
        members = sorted(stats)
        common = sorted(set.intersection(*[set(stats[p]["centres"]) for p in members])) if members else []
        entry: dict = {"profiles": members, "centres": common, "per_profile": stats, "images": {}}
        report["steps"][str(step)] = entry
        if len(members) < 2 or not common:
            entry["note"] = "fewer than two profiles with a common centre; no ensemble"
            continue
        by_key = {(r["profile"], r["centre"]): Path(r["dir"]) for r in rows}

        def ens(centre: int, which: str) -> np.ndarray:
            acc = np.zeros((h, w), np.float32)
            for pid in members:
                m = load_side(by_key[(pid, centre)], which, adapters[pid], (h, w))
                if m is None:
                    raise FileNotFoundError(f"{pid} centre {centre}: no {which} map")
                acc += (m - stats[pid]["mean"]) / stats[pid]["sd"]
            return acc / len(members)

        # one map in memory at a time: the stretch needs only the grid samples
        lo, hi = (float(x) for x in np.percentile(np.concatenate([ens(c, side)[grid] for c in common]), [P_LO, P_HI]))
        entry.update(p1=lo, p99_5=hi, n_grid=int(grid.sum()) * len(common))
        sdir = Path(out_dir) / "ensemble" / f"step{step}"
        sdir.mkdir(parents=True, exist_ok=True)
        for c in common:
            for which, suffix in ((side, ""), (blind, "_blind")):
                try:
                    e = ens(c, which)
                except FileNotFoundError as exc:
                    entry.setdefault("missing", []).append(str(exc))
                    continue
                name = f"ensemble_{side}_c{c:+d}{suffix}"
                np.save(sdir / f"{name}.npy", e.astype(np.float32))
                g = stretch8(e, lo, hi, valid)
                (sdir / f"{name}.png").write_bytes(png_bytes(np.repeat(g[..., None], 3, -1)))
                entry["images"][f"c{c:+d}{suffix}"] = str((sdir / f"{name}.png").relative_to(out_dir))
    return report
