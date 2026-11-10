"""Combine the maps of two or more profiles run on the same segment.

product  pixel-wise product of the probabilities: ink only where every model agrees. This is the rule of the
         PHerc. 0306B First Letters claim (Liam Warner & Nicola Bodill): TRACE v3B x A-hn, pixel by pixel
         (https://github.com/SirProdigle/pherc0306b-first-letters). On PHerc. 0841 known text their records give
         TRACE v3B x A-hn AUC 0.874 / 6.1 % false ink and TRACE v3B x v8-in 0.871 / 5.9 %, against 0.840 / 12.1 %
         for v8-in alone.
mean     pixel-wise mean, e.g. the mean of A-hn's three seeds (the claim's A-hn map).

Inputs are `ink-sdk infer` output folders; the side (forward/reverse) is read from each folder's
probability.npy / probability_reverse.npy. Maps are resized bilinearly to the first map's grid. Probabilities are
used as written (no floor subtraction), as in the claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .ensemble import _FILES, resize_bilinear

MODES = ("product", "mean")


def combine(run_dirs: Sequence[Path], mode: str, side: str, out_dir: Path) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    if side not in _FILES:
        raise ValueError(f"side must be forward or reverse, not {side!r}")
    if len(run_dirs) < 2:
        raise ValueError("combine needs at least two infer folders")
    maps, shape = [], None
    for d in run_dirs:
        path = Path(d) / _FILES[side]
        if not path.is_file():
            raise FileNotFoundError(f"no {side} map in {d} ({path.name})")
        m = np.load(path).astype(np.float32)
        shape = shape or m.shape
        maps.append(resize_bilinear(m, shape))
    out = np.prod(maps, axis=0) if mode == "product" else np.mean(maps, axis=0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / _FILES[side], out.astype(np.float32))
    receipt = {"mode": mode, "side": side, "inputs": [str(Path(d)) for d in run_dirs], "shape": list(shape),
               "p50": float(np.percentile(out, 50)), "p99": float(np.percentile(out, 99)), "max": float(out.max())}
    (out_dir / f"combine_{side}.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt
