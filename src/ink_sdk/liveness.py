"""The liveness gate, ported from Campaign X's calibrated thresholds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MIN_SPREAD_P99_P50 = 0.05
MIN_STD = 0.02
MAX_FRACTION_NEAR_HALF = 0.90
NEAR_HALF = 0.05

FORWARD_REVERSE_THRESHOLDS = (0.5, 0.6, 0.7)
MIN_OVER_THRESHOLD_PIXELS = 300
SUSTAINED_ASYMMETRY_RATIO = 1.5


def assess_liveness(probability: np.ndarray, *, valid: np.ndarray | None = None) -> dict:
    array = np.asarray(probability)
    if valid is not None:
        values = array[np.asarray(valid, bool)]
    else:
        values = array[array > 0]
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return {"verdict": "EMPTY", "reason": "no valid pixels", "metrics": {}}

    p50, p99 = (float(x) for x in np.percentile(values, [50, 99]))
    spread = p99 - p50
    std = float(values.std())
    near_half = float((np.abs(values - 0.5) <= NEAR_HALF).mean())

    failures = []
    if spread < MIN_SPREAD_P99_P50:
        failures.append(f"p99-p50 {spread:.4f} < {MIN_SPREAD_P99_P50}")
    if std < MIN_STD:
        failures.append(f"std {std:.4f} < {MIN_STD}")
    if near_half > MAX_FRACTION_NEAR_HALF:
        failures.append(f"{near_half:.1%} of pixels within {NEAR_HALF} of 0.5")

    metrics = {
        "p50": p50,
        "p99": p99,
        "spread_p99_p50": spread,
        "std": std,
        "fraction_near_half": near_half,
        "valid_pixels": int(values.size),
    }
    if failures:
        return {
            "verdict": "DEGENERATE",
            "reason": "; ".join(failures),
            "metrics": metrics,
            "interpretation": (
                "the map carries no decision: the output head is untrained, "
                "collapsed, or being fed an input outside its training distribution. "
                "Do not screen this map."
            ),
        }
    return {"verdict": "ALIVE", "reason": "", "metrics": metrics}


def require_alive(probability: np.ndarray, *, lane: str, valid: np.ndarray | None = None) -> dict:
    report = assess_liveness(probability, valid=valid)
    if report["verdict"] != "ALIVE":
        raise RuntimeError(
            f"lane {lane} produced a {report['verdict']} map: {report['reason']}. "
            f"{report.get('interpretation', '')}"
        )
    return report


def refuse_if_not_alive(
    report: dict,
    *,
    lane: str,
    output: Path,
    on_degenerate: str = "fail",
    stream=None,
) -> int:
    stream = stream or sys.stderr
    if report.get("verdict") == "ALIVE":
        return 0
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "LANE_NOT_USABLE").write_text(
        f"{report.get('verdict')}: {report.get('reason', '')}\n"
        f"{report.get('interpretation', '')}\n",
        encoding="utf-8",
    )
    message = f"lane {lane} produced a {report.get('verdict')} map: {report.get('reason', '')}"
    if on_degenerate == "fail":
        print(f"\nREFUSED: {message}", file=stream)
        print(
            "The receipt and the map were kept for diagnosis. Re-run with "
            "--on-degenerate warn to continue anyway.",
            file=stream,
        )
        return 3
    print(f"\nWARNING: {message}", file=stream)
    return 0


def forward_reverse_asymmetry(
    forward: np.ndarray | None,
    reverse: np.ndarray | None,
    *,
    valid: np.ndarray | None = None,
) -> dict | None:
    if forward is None or reverse is None:
        return None
    forward = np.asarray(forward, dtype=np.float64)
    reverse = np.asarray(reverse, dtype=np.float64)
    if forward.shape != reverse.shape:
        return {"shape_differs": [list(forward.shape), list(reverse.shape)]}
    mask = np.ones(forward.shape, bool) if valid is None else np.asarray(valid, bool)
    mask = mask & np.isfinite(forward) & np.isfinite(reverse)
    thresholds: dict[str, dict] = {}
    ratios: dict[float, float | None] = {}
    for threshold in FORWARD_REVERSE_THRESHOLDS:
        fwd = int((mask & (forward >= threshold)).sum())
        rev = int((mask & (reverse >= threshold)).sum())
        entry: dict = {"forward_over_px": fwd, "reverse_over_px": rev}
        if fwd < MIN_OVER_THRESHOLD_PIXELS or rev < MIN_OVER_THRESHOLD_PIXELS:
            entry["ratio"] = None
            entry["reason"] = (
                f"fewer than {MIN_OVER_THRESHOLD_PIXELS} px over {threshold}"
            )
        else:
            entry["ratio"] = fwd / rev
        thresholds[str(threshold)] = entry
        ratios[threshold] = entry["ratio"]
    sustained = (
        ratios[0.6] is not None
        and ratios[0.7] is not None
        and ratios[0.6] > SUSTAINED_ASYMMETRY_RATIO
        and ratios[0.7] > SUSTAINED_ASYMMETRY_RATIO
    )
    return {"thresholds": thresholds, "sustained_above_1_5": sustained}
