"""Battery: every profile x depth centre x depth width on one segment, with a mosaic.

One thick surface volume (rendered once) is sliced into depth windows. A window is the profile's own number of
layers (``input.depth``), centred ``centre`` layers away from the middle of the thick volume, with ``step`` layers
between consecutive inputs (step 1 = contiguous; step 2 = the same number of layers spread over twice the depth).
Each window runs through the normal ``infer`` path, so scale checks, resampling, liveness gates and receipts are
the same as a single ``ink-sdk infer``.

Outputs (``OUT``):
  cells/<profile>/c<centre>_s<step>/     probability*.npy, preview.tif, INK_SDK_RECEIPT.json (one infer each)
  mosaic_<bg>_t<thr>.png                 rows = profile x step, columns = centre; bg = ct | black
  stability_t<thr>.png                   fraction of all cells where the map is >= thr (what holds across settings)
  index.html, BATTERY.json               everything above, plus every skipped profile or failed cell and why

The battery does not pick a best cell. It shows all of them, and the stability map shows what survives the
choice of settings.
"""

from __future__ import annotations

import base64
import dataclasses
import html
import json
import struct
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .contracts import RunContext, jsonable
from .errors import InkSDKError
from .io import open_volume_array, write_json, write_zarr
from .prepare import plan_scale
from .profiles import ModelProfile


# ---------- tiny PNG writer (no Pillow dependency) ----------
def png_bytes(rgb: np.ndarray) -> bytes:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _resize_nearest(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ys = (np.arange(shape[0]) * a.shape[0] / shape[0]).astype(int)
    xs = (np.arange(shape[1]) * a.shape[1] / shape[1]).astype(int)
    return a[np.ix_(ys, xs)]


def _thumb(a: np.ndarray, width: int) -> np.ndarray:
    h = max(1, int(round(a.shape[0] * width / a.shape[1])))
    return _resize_nearest(a, (h, width))


# ---------- planning ----------
@dataclasses.dataclass(frozen=True)
class Cell:
    profile_id: str
    centre: int
    step: int
    layers: tuple[int, ...]


def window(n_layers: int, depth: int, centre: int, step: int) -> list[int] | None:
    mid = n_layers // 2
    idx = [mid + centre + step * (k - depth // 2) for k in range(depth)]
    return idx if min(idx) >= 0 and max(idx) < n_layers else None


def plan_battery(
    n_layers: int,
    pixel_um: float,
    profiles: Sequence[ModelProfile],
    centres: Sequence[int],
    steps: Sequence[int],
    resample_from_um: float | None,
) -> tuple[list[Cell], list[dict]]:
    cells, skipped = [], []
    for p in profiles:
        try:
            plan_scale(pixel_um, p, resample_from_um=resample_from_um if p.adapter == "ink9um" else None)
        except Exception as exc:  # incompatible scale: report, do not run
            skipped.append({"profile": p.id, "reason": str(exc)})
            continue
        for s in steps:
            for c in centres:
                idx = window(n_layers, int(p.input.depth), int(c), int(s))
                if idx is None:
                    skipped.append({"profile": p.id, "centre": c, "step": s,
                                    "reason": f"window needs layers outside 0..{n_layers - 1}; render more slices"})
                    continue
                cells.append(Cell(p.id, int(c), int(s), tuple(idx)))
    return cells, skipped


# ---------- running ----------
def run_battery(
    sdk,
    surface_volume: Path,
    pixel_um: float,
    profiles: Sequence[ModelProfile],
    out_dir: Path,
    ctx: RunContext,
    *,
    centres: Sequence[int] = (-8, -4, 0, 4, 8),
    steps: Sequence[int] = (1, 2),
    thresholds: Sequence[float] = (0.5, 0.7),
    resample_from_um: float | None = None,
    thumb_px: int = 360,
    say: Callable[[str], None] = print,
) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    thick = open_volume_array(Path(surface_volume))
    if thick.dtype != np.uint8 or thick.ndim != 3:
        raise InkSDKError(f"surface volume must be uint8 ZYX, got {thick.dtype} {thick.shape}")
    n = int(thick.shape[0])
    by_id = {p.id: p for p in profiles}
    cells, skipped = plan_battery(n, pixel_um, profiles, centres, steps, resample_from_um)
    say(f"battery: {n} layers, {len(cells)} cells, {len(skipped)} skipped")
    started = time.time()
    if ctx.dry_run:
        return {"layers": n, "cells": [dataclasses.asdict(c) for c in cells], "skipped": skipped}

    work = out_dir / "_windows"; work.mkdir(exist_ok=True)
    results: list[dict] = []
    from .resolver import build_segment

    for i, cell in enumerate(cells, 1):
        p = by_id[cell.profile_id]
        tag = f"c{cell.centre:+d}_s{cell.step}"
        win = work / f"d{p.input.depth}_{tag}.zarr"
        if not win.exists():
            write_zarr(win, np.ascontiguousarray(thick[list(cell.layers)]),
                       attrs={"pixel_um": [pixel_um] * 3, "battery_layers": list(cell.layers)}, overwrite=True)
        cdir = out_dir / "cells" / p.id.replace("@", "_") / tag
        row = {"profile": p.id, "centre": cell.centre, "step": cell.step, "layers": list(cell.layers), "dir": str(cdir)}
        say(f"[{i}/{len(cells)}] {p.id} {tag}")
        try:
            cctx = dataclasses.replace(ctx, on_degenerate="warn", overwrite=True,
                                       resample_from_um=resample_from_um if p.adapter == "ink9um" else None)
            seg = build_segment(profile=p, ctx=cctx, surface_volume=win, voxel_um=pixel_um)
            res = sdk.infer(seg, p, out_dir=cdir, ctx=cctx)
            rec = res.receipt
            row.update(primary=res.primary, ok=True,
                       liveness=rec.get("liveness", {}).get(res.primary, {}).get("verdict"),
                       p99=rec.get("maps", {}).get(res.primary, {}).get("statistics", {}).get("p99"),
                       reverse_summary=rec.get("reverse_summary"))
        except Exception as exc:  # one failed cell must not stop the battery
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:500])
        results.append(row)

    report = {"schema": "ink-sdk.battery.v1", "surface_volume": str(surface_volume), "pixel_um": pixel_um,
              "layers": n, "centres": list(centres), "steps": list(steps), "thresholds": list(thresholds),
              "resample_from_um": resample_from_um, "cells": results, "skipped": skipped,
              "runtime_seconds": round(time.time() - started, 1),
              "non_claims": ["A probability map is not OCR and does not accept ink, text or letters.",
                             "The battery shows every cell; it does not select a best one."]}
    report["images"] = make_images(thick, results, out_dir, thresholds, thumb_px)
    write_json(out_dir / "BATTERY.json", report)
    (out_dir / "index.html").write_text(to_html(report, out_dir), encoding="utf-8")
    say(f"battery: {out_dir / 'index.html'}")
    return report


def _load_primary(row: dict) -> np.ndarray | None:
    d = Path(row["dir"])
    for name in ("probability_primary.npy", "probability.npy"):
        if (d / name).is_file():
            return np.load(d / name)
    return None


def make_images(thick: np.ndarray, results: list[dict], out_dir: Path, thresholds: Sequence[float], thumb_px: int) -> dict:
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return {}
    maps = {(r["profile"], r["centre"], r["step"]): _load_primary(r) for r in ok}
    maps = {k: v for k, v in maps.items() if v is not None}
    if not maps:
        return {}
    shape = next(iter(maps.values())).shape
    rows = sorted({(k[0], k[2]) for k in maps}); cols = sorted({k[1] for k in maps})
    n = thick.shape[0]
    images: dict[str, Any] = {"rows": [f"{p} step {s}" for p, s in rows], "cols": [f"centre {c:+d}" for c in cols]}
    gap = 6
    for thr in thresholds:
        for bg in ("ct", "black"):
            tiles = []
            for p, s in rows:
                line = []
                for c in cols:
                    m = maps.get((p, c, s))
                    if bg == "ct":
                        base = _resize_nearest(thick[min(n - 1, max(0, n // 2 + c))], m.shape if m is not None else shape)
                        rgb = np.repeat(base[..., None], 3, -1).astype(np.float32) * 0.85
                    else:
                        rgb = np.zeros((*(m.shape if m is not None else shape), 3), np.float32)
                    if m is not None:
                        hit = m >= thr
                        rgb[hit] = rgb[hit] * 0.25 + np.array([255, 170, 40], np.float32) * 0.75
                    else:
                        rgb[:] = 40
                    line.append(_thumb(rgb.clip(0, 255).astype(np.uint8), thumb_px))
                hmax = max(t.shape[0] for t in line)
                line = [np.pad(t, ((0, hmax - t.shape[0]), (0, gap), (0, 0)), constant_values=20) for t in line]
                tiles.append(np.pad(np.concatenate(line, 1), ((0, gap), (0, 0), (0, 0)), constant_values=20))
            wmax = max(t.shape[1] for t in tiles)
            mosaic = np.concatenate([np.pad(t, ((0, 0), (0, wmax - t.shape[1]), (0, 0)), constant_values=20) for t in tiles], 0)
            name = f"mosaic_{bg}_t{thr:g}.png"; (out_dir / name).write_bytes(png_bytes(mosaic))
            images[name] = {"threshold": thr, "background": bg}
        same = [v for v in maps.values() if v.shape == shape]
        frac = np.mean([v >= thr for v in same], 0)
        heat = np.stack([frac * 255, frac * 200, frac * 60], -1).astype(np.uint8)
        name = f"stability_t{thr:g}.png"; (out_dir / name).write_bytes(png_bytes(_thumb(heat, 1200)))
        images[name] = {"threshold": thr, "cells": len(same),
                        "note": "fraction of cells (all profiles, centres and widths) where the map is >= threshold"}
    return images


def _p99(r: dict) -> str:
    return "" if r.get("p99") is None else "%.3f" % r["p99"]


def _where(s: dict) -> str:
    return "" if "centre" not in s else " centre %+d step %d" % (s["centre"], s["step"])


def to_html(report: dict, out_dir: Path) -> str:
    def img(name: str) -> str:
        p = out_dir / name
        return (f'<img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" alt="{html.escape(name)}">'
                if p.is_file() else "")
    ims = report.get("images", {})
    rows = "".join(
        f"<tr><td>{html.escape(r['profile'])}</td><td>{r['centre']:+d}</td><td>{r['step']}</td>"
        f"<td>{'ok' if r.get('ok') else 'FAILED'}</td><td>{html.escape(str(r.get('liveness') or r.get('error', '')))[:160]}</td>"
        f"<td>{_p99(r)}</td></tr>" for r in report["cells"])
    skipped = "".join(f"<li>{html.escape(s['profile'])}{_where(s)}: "
                      f"{html.escape(s['reason'])}</li>" for s in report["skipped"])
    mos = "".join(f"<h3>threshold {v['threshold']:g}, over {v['background']}</h3>{img(k)}"
                  for k, v in ims.items() if k.startswith("mosaic_"))
    stab = "".join(f"<h3>threshold {v['threshold']:g} ({v['cells']} cells)</h3>{img(k)}"
                   for k, v in ims.items() if k.startswith("stability_"))
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ink-sdk battery</title><style>body{{font:14px/1.5 system-ui,sans-serif;background:#17140f;color:#eee5d8;margin:0;padding:20px}}
h1{{font:600 26px Georgia,serif}} img{{max-width:100%;border:1px solid #3a3229;margin:6px 0}} table{{border-collapse:collapse}}
td,th{{border-bottom:1px solid #3a3229;padding:4px 10px;text-align:left}} .k{{color:#a99f91}}</style></head><body>
<h1>ink-sdk battery</h1><p class="k">{html.escape(report['surface_volume'])} · {report['layers']} layers · {report['pixel_um']} um ·
centres {report['centres']} · steps {report['steps']} · {report['runtime_seconds']} s</p>
<p class="k">Rows: {html.escape(', '.join(ims.get('rows', [])))}. Columns: {html.escape(', '.join(ims.get('cols', [])))}.
Orange = map ≥ threshold. Every cell is shown; none is selected.</p>
<h2>What holds across settings</h2>{stab}<h2>Mosaics</h2>{mos}
<h2>Skipped</h2><ul>{skipped or '<li>none</li>'}</ul>
<h2>Cells</h2><table><tr><th>profile</th><th>centre</th><th>step</th><th></th><th>liveness / error</th><th>p99</th></tr>{rows}</table>
</body></html>"""
