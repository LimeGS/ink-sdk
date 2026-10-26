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
  ensemble/step<s>/                      with ``ensemble``: z-score ensemble per centre, reading and blind side
                                         (see ensemble.py; after Erwin Nieuwlaar's PHerc0343 recipe)

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
            sp = plan_scale(pixel_um, p, resample_from_um=resample_from_um if p.adapter == "ink9um" else None)
        except Exception as exc:  # incompatible scale: report, do not run
            skipped.append({"profile": p.id, "reason": str(exc)})
            continue
        # The window is cut at the SOURCE scale. When the plan pools Z by f, the model's
        # depth needs depth * f source layers, or it receives depth / f layers after pooling.
        fz = int(sp.factors_zyx[0]) if sp.mode == "pool" else 1
        depth = int(p.input.depth) * fz
        for s in steps:
            for c in centres:
                idx = window(n_layers, depth, int(c), int(s))
                if idx is None:
                    skipped.append({"profile": p.id, "centre": c, "step": s,
                                    "reason": f"needs {depth} source layers"
                                              + (f" ({p.input.depth} x pool {fz})" if fz > 1 else "")
                                              + f", outside 0..{n_layers - 1}; render more slices"})
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
    ensemble: str | None = None,
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
    if ensemble is not None and ensemble not in ("forward", "reverse"):
        raise InkSDKError(f"--ensemble must be forward or reverse (the reading side), not {ensemble!r}")
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
            # Keep the end: a traceback's cause and the SDK's hint are on its last lines.
            message = f"{type(exc).__name__}: {exc}"
            if len(message) > 2000:
                message = message[:300] + " [...] " + message[-1700:]
            row.update(ok=False, error=message)
        results.append(row)

    report = {"schema": "ink-sdk.battery.v1", "surface_volume": str(surface_volume), "pixel_um": pixel_um,
              "layers": n, "centres": list(centres), "steps": list(steps), "thresholds": list(thresholds),
              "resample_from_um": resample_from_um, "cells": results, "skipped": skipped,
              "runtime_seconds": round(time.time() - started, 1),
              "non_claims": ["A probability map is not OCR and does not accept ink, text or letters.",
                             "The battery shows every cell; it does not select a best one."]}
    report["images"] = make_images(thick, results, out_dir, thresholds, thumb_px)
    if ensemble:
        from .ensemble import ensemble_battery
        say(f"battery: ensemble, reading side {ensemble}")
        report["ensemble"] = ensemble_battery(thick, results, {p.id: p.adapter for p in profiles}, out_dir, ensemble)
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


_CSS = """
:root{--bg:#14110d;--panel:#1d1915;--line:#352d25;--ink:#efe6d8;--mute:#a59a8b;--accent:#f0a830;
--ok:#7fc48a;--warn:#e7b04a;--bad:#e8735f;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:28px clamp(16px,4vw,48px) 18px;border-bottom:1px solid var(--line)}
h1{font:600 30px/1.2 Georgia,"Times New Roman",serif;margin:0 0 4px;letter-spacing:.2px}
h1 span{color:var(--accent)}
.src{font:13px var(--mono);color:var(--mute);word-break:break-all}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 14px;min-width:96px}
.stat b{display:block;font-size:20px;font-variant-numeric:tabular-nums}.stat small{color:var(--mute);font-size:12px}
nav{position:sticky;top:0;z-index:5;background:rgba(20,17,13,.94);backdrop-filter:blur(6px);
border-bottom:1px solid var(--line);padding:0 clamp(16px,4vw,48px);display:flex;gap:4px;overflow-x:auto}
nav a{color:var(--mute);text-decoration:none;padding:11px 12px;font-size:14px;white-space:nowrap;border-bottom:2px solid transparent}
nav a:hover{color:var(--ink);border-color:var(--accent)}
main{padding:8px clamp(16px,4vw,48px) 48px;max-width:1500px}
section{padding-top:22px}h2{font:600 21px Georgia,serif;margin:10px 0 6px}
h3{font-size:15px;margin:18px 0 8px;color:var(--ink)}
.note{color:var(--mute);font-size:13.5px;max-width:880px;margin:0 0 12px}
.profiles{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 10px}
.chip{font:12px var(--mono);background:#2a231c;border:1px solid var(--line);border-radius:999px;padding:2px 10px}
.pairs{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,560px),1fr));gap:16px}
.pair{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 12px 12px}
.pair h4{margin:0 0 8px;font-size:14px;font-weight:600}.pair h4 span{color:var(--mute);font-weight:400}
.two{display:grid;grid-template-columns:1fr 1fr;gap:8px}
figure{margin:0}figcaption{font-size:12px;color:var(--mute);margin-bottom:4px}
figcaption.read{color:var(--accent)}
img{display:block;max-width:100%;height:auto;border-radius:6px;background:#000}
a.zoom{display:block;cursor:zoom-in}
.frame{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px}
.stab{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,420px),1fr));gap:16px;max-width:1100px}
.mosaic{display:grid;grid-template-columns:auto 1fr;gap:6px 10px;align-items:stretch}
.cols{grid-column:2;display:flex}.cols span{flex:1;text-align:center;font-size:12px;color:var(--mute)}
.rows{display:flex;flex-direction:column}.rows span{flex:1;display:flex;align-items:center;justify-content:flex-end;
text-align:right;font:12px var(--mono);color:var(--mute);max-width:230px;overflow-wrap:anywhere}
.mosaic img{width:100%}
details{margin:8px 0}summary{cursor:pointer;font-weight:600;padding:6px 0}summary span{color:var(--mute);font-weight:400}
table{border-collapse:collapse;width:100%;font-size:14px}
th{text-align:left;color:var(--mute);font-weight:500;font-size:12.5px;border-bottom:1px solid var(--line);padding:6px 10px}
td{border-bottom:1px solid #2a231c;padding:6px 10px;vertical-align:middle}
td.num{font-variant-numeric:tabular-nums;font-family:var(--mono)}td.p{font:13px var(--mono)}
.tw{overflow-x:auto}.badge{font:600 11.5px var(--mono);padding:2px 8px;border-radius:999px;border:1px solid}
.b-ok{color:var(--ok);border-color:#3c5e41}.b-warn{color:var(--warn);border-color:#6a5326}.b-bad{color:var(--bad);border-color:#6d3a31}
.bar{display:inline-block;width:70px;height:6px;background:#2a231c;border-radius:3px;margin-left:8px;vertical-align:middle}
.bar i{display:block;height:100%;background:var(--accent);border-radius:3px}
.err{font:12px var(--mono);color:var(--bad);max-width:520px;word-break:break-word}
ul.skip{padding-left:18px;color:var(--mute)}ul.skip b{color:var(--ink);font-weight:500}
footer{border-top:1px solid var(--line);padding:18px clamp(16px,4vw,48px);color:var(--mute);font-size:13px}
@media(max-width:640px){.two{grid-template-columns:1fr}.mosaic{grid-template-columns:1fr}.rows,.cols{display:none}}
"""


def _esc(x: Any) -> str:
    return html.escape(str(x))


def _b64(path: Path) -> str | None:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}" if path.is_file() else None


def _badge(row: dict) -> str:
    if not row.get("ok"):
        return '<span class="badge b-bad">FAILED</span>'
    v = str(row.get("liveness") or "-")
    cls = "b-ok" if v == "ALIVE" else "b-warn"
    return f'<span class="badge {cls}">{_esc(v)}</span>'


def _p99_cell(row: dict) -> str:
    p = row.get("p99")
    if p is None:
        return '<td class="num"></td>'
    w = max(0.0, min(1.0, float(p))) * 100
    return f'<td class="num">{float(p):.3f}<span class="bar"><i style="width:{w:.0f}%"></i></span></td>'


def _ensemble_html(ens: dict | None) -> str:
    if not ens:
        return ""
    read, blind = _esc(ens["reading_side"]), _esc(ens.get("blind_side", ""))
    out = ['<section id="ensemble"><h2>Ensemble</h2>',
           f'<p class="note">Mean of the per-profile z-scores. Statistics and one contrast stretch per depth width come '
           f'from the reading side (<b>{read}</b>) and are applied to the blind side ({blind}) as well, so the two '
           "can be compared directly. After Erwin Nieuwlaar's PHerc0343 recipe; his tile-phase merge is not included. "
           "Click an image for full size.</p>"]
    for step, e in ens.get("steps", {}).items():
        chips = "".join(f'<span class="chip">{_esc(p)}</span>' for p in e.get("profiles", []))
        out.append(f'<h3>Depth width: step {_esc(step)}</h3><div class="profiles">{chips}</div>')
        if e.get("note"):
            out.append(f'<p class="note">{_esc(e["note"])}</p>')
            continue
        out.append('<div class="pairs">')
        for c in e.get("centres", []):
            figs = []
            for key, label, cls in ((f"c{c:+d}", f"reading side · {read}", "read"),
                                    (f"c{c:+d}_blind", f"blind side · {blind}", "")):
                src = e.get("images", {}).get(key)
                if src:
                    figs.append(f'<figure><figcaption class="{cls}">{label}</figcaption>'
                                f'<a class="zoom" href="{_esc(src)}" target="_blank" rel="noopener">'
                                f'<img loading="lazy" src="{_esc(src)}" alt="ensemble {_esc(key)}"></a></figure>')
            out.append(f'<div class="pair"><h4>centre {c:+d} <span>layers from the middle</span></h4>'
                       f'<div class="two">{"".join(figs)}</div></div>')
        out.append("</div>")
    out.append("</section>")
    return "".join(out)


def _mosaics_html(ims: dict, out_dir: Path) -> str:
    rows, cols = ims.get("rows", []), ims.get("cols", [])
    keys = [k for k in ims if k.startswith("mosaic_")]
    if not keys:
        return ""
    out = ['<section id="mosaics"><h2>Mosaics</h2><p class="note">One tile per cell: rows are profile × depth width, '
           "columns are depth centres. Orange marks map ≥ threshold, over the CT layer at that centre or over black."
           "</p>"]
    for i, k in enumerate(keys):
        v, src = ims[k], _b64(out_dir / k)
        if not src:
            continue
        lab_r = "".join(f"<span>{_esc(r)}</span>" for r in rows)
        lab_c = "".join(f"<span>{_esc(c)}</span>" for c in cols)
        out.append(f'<details{" open" if i == 0 else ""}><summary>threshold {v["threshold"]:g} '
                   f'<span>over {_esc(v["background"])}</span></summary><div class="frame mosaic">'
                   f'<div class="cols">{lab_c}</div><div class="rows">{lab_r}</div>'
                   f'<img src="{src}" alt="{_esc(k)}"></div></details>')
    out.append("</section>")
    return "".join(out)


def to_html(report: dict, out_dir: Path) -> str:
    ims = report.get("images", {})
    cells = report["cells"]
    n_ok = sum(1 for r in cells if r.get("ok"))
    n_alive = sum(1 for r in cells if r.get("ok") and r.get("liveness") == "ALIVE")
    profiles = sorted({r["profile"] for r in cells})
    name = Path(str(report["surface_volume"]).rstrip("/")).name
    stats = [(len(cells), "cells"), (n_ok, "ran"), (n_alive, "ALIVE"), (len(cells) - n_ok, "failed"),
             (len(report["skipped"]), "skipped"), (len(profiles), "profiles"),
             (f'{report["pixel_um"]:g} µm', f'{report["layers"]} layers'),
             (f'{report["runtime_seconds"]:g} s', "runtime")]
    stat_html = "".join(f'<div class="stat"><b>{_esc(v)}</b><small>{_esc(k)}</small></div>' for v, k in stats)

    stab = [f'<figure class="frame"><figcaption>threshold {v["threshold"]:g} · {v["cells"]} cells</figcaption>'
            f'<img src="{_b64(out_dir / k)}" alt="{_esc(k)}"></figure>'
            for k, v in ims.items() if k.startswith("stability_") and (out_dir / k).is_file()]
    stab_html = ('<section id="stability"><h2>What holds across settings</h2><p class="note">Fraction of all cells '
                 "(every profile, centre and width) in which the map is at or above the threshold. Bright = holds "
                 "whatever the setting.</p>" + f'<div class="stab">{"".join(stab)}</div></section>') if stab else ""

    trs = []
    for r in cells:
        detail = "" if r.get("ok") else f'<div class="err">{_esc(str(r.get("error", ""))[-300:])}</div>'
        trs.append(f'<tr><td class="p">{_esc(r["profile"])}</td><td class="num">{r["centre"]:+d}</td>'
                   f'<td class="num">{r["step"]}</td><td>{_badge(r)}{detail}</td>{_p99_cell(r)}</tr>')
    skipped = "".join(f"<li><b>{_esc(s['profile'])}</b>{_esc(_where(s))}: {_esc(s['reason'])}</li>"
                      for s in report["skipped"])
    nav = [("ensemble", "Ensemble", report.get("ensemble")), ("stability", "Across settings", stab),
           ("mosaics", "Mosaics", any(k.startswith("mosaic_") for k in ims)), ("cells", "Cells", True),
           ("skipped", "Skipped", True)]
    nav_html = "".join(f'<a href="#{i}">{t}</a>' for i, t, on in nav if on)
    claims = "".join(f"<li>{_esc(c)}</li>" for c in report.get("non_claims", []))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ink-sdk battery · {_esc(name)}</title>
<style>{_CSS}</style></head><body>
<header><h1>ink-sdk <span>battery</span></h1><div class="src">{_esc(report["surface_volume"])}</div>
<div class="stats">{stat_html}</div>
<p class="note" style="margin-top:12px">Every profile × depth centre {_esc(report["centres"])} × width (step)
{_esc(report["steps"])}. Every cell is shown; none is selected.</p></header>
<nav>{nav_html}</nav><main>
{_ensemble_html(report.get("ensemble"))}{stab_html}{_mosaics_html(ims, out_dir)}
<section id="cells"><h2>Cells</h2><div class="tw"><table><tr><th>profile</th><th>centre</th><th>step</th>
<th>liveness</th><th>p99 of primary map</th></tr>{"".join(trs)}</table></div></section>
<section id="skipped"><h2>Skipped</h2><ul class="skip">{skipped or "<li>none</li>"}</ul></section>
</main><footer><ul style="margin:0;padding-left:18px">{claims}</ul></footer></body></html>"""


def _where(s: dict) -> str:
    return "" if "centre" not in s else " centre %+d step %d" % (s["centre"], s["step"])
