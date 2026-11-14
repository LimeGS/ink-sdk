"""Known training overlap: was this checkpoint, or a model it was fine-tuned from, trained on this segment?

A map on a segment the checkpoint (or its base) was trained on measures fit, not
generalisation. This module keeps a small, sourced registry of profile/segment
pairs and answers

  lookup_profile(profile_id, scroll, segment, paths=...) -> dict

with ``status`` True (in-sample), False (held out) or None (unknown). It is
metadata only: it never changes a probability map, a liveness verdict or an exit
code.

Rules (first that applies):

* **in-sample** -- the segment is in the training list of the checkpoint itself
  (``seen_by: "checkpoint"``) or of a model it was fine-tuned from
  (``seen_by: "base model"``, ``inherited: True``). The whole lineage is
  walked; every hit is listed under ``evidence``.
* **held out** -- only when a held-out record with an accepted ``basis`` covers
  every model in the lineage: ``stated`` (the author's card or training code
  says so) or ``complete list`` (not in a training list whose whole chain of
  start weights is published and read; by definition this covers the lineage).
* **unknown** -- everything else: a profile or segment that is not listed, a
  model whose start weights are not identified, a held-out record whose basis
  is only a later segment timestamp (``after upload``) or a near-identical id or
  re-render of the same sheet, a segment used for checkpoint selection, and an
  ambiguous input (conflicting scroll / segment ids). "Not listed" is never
  evidence of held out.

Segments are named three ways and all are matched: the public 14-digit segment
id (20250108000005), the public w-number in a segment folder name
(20250108000005-w030_2025010818 -> w030), and the case name of the
scrollprize/ink_9um label dataset (pherc0139-w017). The label dataset uses its
own w-numbers, which do not follow the public ones, so a label case name is
only matched as a label case name. Matching is by segment key only: a
neighbouring wrap or another segment over the same area is not caught.

Scroll and segment identifiers are resolved once, from the ORIGINAL input
(surface volume, mesh, CT, segment id) plus the optional ``--scroll`` /
``--segment`` overrides, with :func:`resolve_input`. The battery resolves them
before it writes its temporary depth windows and hands the result to every
cell, so a single ``infer`` and a battery cell report the same status. Only
identifiers live here; no scroll data.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

# ---------- sources ----------
JO_TUD_2349 = (
    "jo-tud, ScrollPrize/villa#2349 "
    "(https://github.com/ScrollPrize/villa/pull/2349#issuecomment-6548401630): "
    "model x labelled-set table for ink_9um s42/s43, ink9um-dense-native and v8-in on PHerc0139 and 0343P"
)
JO_TUD_17 = (
    "jo-tud, LimeGS/ink-sdk#17 (https://github.com/LimeGS/ink-sdk/issues/17#issuecomment-6876506166): "
    "lineage and training lists read from embedded configs, meta.json files and model cards"
)
INK9_CARD = (
    "scrollprize/ink_9um model card @7109667e (https://huggingface.co/scrollprize/ink_9um), 'Training data': "
    "PHerc0139 (9), Scroll 1667 (6), PHerc Paris 4 (8), PHerc0814 (1) plus 5 native 0139 segments; "
    "one config, the two runs differ only by seed"
)
DENSE_CONFIG = (
    "Nieuwlaar/ink9um-dense-native @b894a049, training/configs/train_dense_native.json "
    "(https://huggingface.co/Nieuwlaar/ink9um-dense-native)"
)
V8IN_LIST = (
    "v8-in training list = patch pack segments.json [train]; v8-in starts from Kinetics video weights "
    "(no CT start checkpoint), per " + JO_TUD_17
)
TRACE_CARD = (
    "TRACE v3B model card, Jashann/vesuvius-scrolling @4f4997ad MODEL_CARD.md "
    "(https://github.com/Jashann/vesuvius-scrolling/blob/4f4997adcccaa9124ab5af02605233f7e3388c69/MODEL_CARD.md)"
)
AHN_CARD = (
    "A-hn model card and checkpoints/hn4_s{0,1,2}_final/meta.json, Prodigle/pherc0306b-ink-ahn @e57c4d54 "
    "(https://huggingface.co/Prodigle/pherc0306b-ink-ahn)"
)

# ---------- vocabulary ----------
USE_TRAINING = "training"
USE_SELECTION = "checkpoint selection"
BASIS_STATED = "stated"
BASIS_COMPLETE = "complete list"
BASIS_AFTER_UPLOAD = "after upload"
#: held-out bases that establish held out on their own; anything else stays unknown
ACCEPTED_HELD_OUT_BASES = (BASIS_STATED, BASIS_COMPLETE)

LABEL = {True: "in-sample (training overlap)", False: "held out", None: "unknown"}

# ---------- built-in table ----------
# PHerc0139: public w-number -> (public segment id, ink_9um label case name or None).
# Case -> segment from the label .zattrs (source_surface_volume), per jo-tud on villa#2349.
# The label case numbers do NOT follow the public ones: label pherc0139-w016 is public
# w029, pherc0139-w028 is public w044, pherc0139-w029 is public w045.
_P0139 = {
    "w029": ("20250108000004", "pherc0139-w016"),
    "w030": ("20250108000005", "pherc0139-w017"),
    "w033": ("20260219000000", None),
    "w035": ("20260317000000", "pherc0139-w035"),
    "w039": ("20260302000000", "pherc0139-w039"),
    "w040": ("20250831000000", "pherc0139-w040"),
    "w041": ("20260108000000", "pherc0139-w041"),
    "w043": ("20260112000000", "pherc0139-w043"),
    "w044": ("20260115000000", "pherc0139-w028"),
    "w045": ("20260126000000", "pherc0139-w029"),
}


def _seg(w, **extra):
    sid, case = _P0139[w]
    spec = {"scroll": "PHerc0139", "segment_id": sid, "public_w": w}
    if case:
        spec["label_case"] = case
    return {**spec, **extra}


_INK9_CASES = ["w029", "w030", "w035", "w039", "w040", "w041", "w043", "w044", "w045"]
_ZATTRS = "label case in the ink_9um training config; case -> segment from .zattrs source_surface_volume"
_INK9_0139_TRAIN = [_seg(w, evidence=_ZATTRS) for w in _INK9_CASES]
_INK9_0139_TRAIN[0]["note"] = "part of public w029 is the ink_9um online-validation region"
_INK9_HELD = [
    {
        "scroll": "PHerc0343P",
        "segment_id": "*",
        "basis": BASIS_COMPLETE,
        "note": "ink_9um trains on PHerc0139, Scroll 1667, Paris 4 and PHerc0814 only, from no start checkpoint; "
                "0343P is a separately scanned fragment (no shared voxel with PHerc0343)",
        "source": INK9_CARD + "; " + JO_TUD_2349 + "; " + JO_TUD_17,
    }
]

# dense-native: the nine label cases, w033, and native-scan config entries read as public
# w-numbers (only the w-number is known for those, so they match on it alone).
_DENSE_CONFIG_ONLY = ["w032", "w034", "w046", "w049", "w050"]
_DENSE_TRAIN = (
    [_seg(w, evidence=_ZATTRS) for w in _INK9_CASES]
    + [_seg("w033", evidence="native-scan config entry read as public w033 (jo-tud)")]
    + [
        {
            "scroll": "PHerc0139",
            "public_w": w,
            "evidence": "native-scan config entry read as public w-number; mapping not checked",
        }
        for w in _DENSE_CONFIG_ONLY
    ]
)

_CANVAS = "patch pack 0139_<w>v* has exactly the canvas size of the public 9.362 um segment (not a byte check)"
_V8IN_TRAIN = [_seg(w, evidence=_CANVAS) for w in ("w033", "w035", "w041", "w044")]
_V8IN_HELD = [
    _seg(w, basis=BASIS_COMPLETE, source=JO_TUD_2349 + "; " + V8IN_LIST)
    for w in ("w029", "w030", "w039", "w040", "w043", "w045")
] + [
    {
        "scroll": "PHerc0343P",
        "segment_id": "20250511003658",
        "basis": BASIS_COMPLETE,
        "note": "0343P ir-truth-9um",
        "source": JO_TUD_2349 + "; " + V8IN_LIST,
    }
]

# TRACE v3B used these for checkpoint selection (card, 'Checkpoint selection (disclosure)').
_TRACE_SELECTION = [
    _seg("w043", evidence="scored every 1000-1500 steps; best mean AUC kept", source=TRACE_CARD),
    {"scroll": "PHerc0841", "segment_id": "20260220213127", "public_w": "w00",
     "evidence": "PHerc0841 w00; 'never trained on, not never looked at'", "source": TRACE_CARD + "; " + JO_TUD_17},
    {"scroll": "PHerc0841", "segment_id": "20260220214732",
     "evidence": "one of the card's ag144 / ag174; segment id per jo-tud", "source": TRACE_CARD + "; " + JO_TUD_17},
    {"scroll": "PHerc0841", "segment_id": "20260221022814",
     "evidence": "one of the card's ag144 / ag174; segment id per jo-tud", "source": TRACE_CARD + "; " + JO_TUD_17},
]

_AHN_ANCHORS = [
    {"scroll": "PHerc1447", "public_w": w, "evidence": "meta.json data.anchor (PHerc1447 team labels)"}
    for w in ("w058", "w060")
]

# Exact ink-sdk profile ids only. ``fine_tuned_from`` names the ``key`` of the entry whose
# weights the checkpoint started from; ``start_weights`` says what is known about the start
# when there is no such entry. ``complete`` is True only when the training list of this model
# and of every model before it is published and read in full.
BUILTIN: list[dict] = [
    {
        "key": "ink_9um-s42",
        "model": "scrollprize ink_9um seed 42",
        "profiles": ["ink_9um@scrollprize-seed42-step75000"],
        "fine_tuned_from": None,
        "start_weights": "none (trained from scratch; the embedded config names no start checkpoint)",
        "in_sample": _INK9_0139_TRAIN,
        "held_out": _INK9_HELD,
        "source": JO_TUD_2349 + "; " + INK9_CARD,
        "note": "all nine labelled 0139 cases are ink_9um training cases, so ink_9um has no held-out labelled "
                "0139 segment",
    },
    {
        "key": "ink_9um-s43",
        "model": "scrollprize ink_9um seed 43",
        "profiles": ["ink_9um@scrollprize-seed43-step75000"],
        "fine_tuned_from": None,
        "start_weights": "none (same config as seed 42, only the seed differs)",
        "in_sample": _INK9_0139_TRAIN,
        "held_out": _INK9_HELD,
        "source": JO_TUD_2349 + "; " + INK9_CARD,
        "note": "all nine labelled 0139 cases are ink_9um training cases, so ink_9um has no held-out labelled "
                "0139 segment",
    },
    {
        "key": "dense-native",
        "model": "Nieuwlaar/ink9um-dense-native",
        "profiles": ["ink_9um@dense-native-step016000"],
        "fine_tuned_from": None,
        "start_weights": "unidentified (config 'checkpoint': /workspace/job/init.pth); its training list is not read",
        "in_sample": _DENSE_TRAIN,
        "held_out": [],
        "source": JO_TUD_2349 + "; " + DENSE_CONFIG,
        "note": "trains on the nine 0139 label cases plus native-scan 0139 entries; held out stays unknown "
                "because the start weights are not identified",
    },
    {
        "key": "v8-in",
        "model": "YoussefMoNader/ink-8um-v8in",
        "profiles": ["v8in@ink-8um-v8in"],
        "fine_tuned_from": None,
        "start_weights": "Kinetics video weights (no CT data)",
        "complete": True,
        "in_sample": _V8IN_TRAIN,
        "held_out": _V8IN_HELD,
        "source": JO_TUD_2349 + "; " + JO_TUD_17,
        "note": "only v8-in's PHerc0139 entries are recorded here; other scrolls stay unknown",
    },
    {
        "key": "trace-v3b",
        "model": "TRACE v3B",
        "profiles": ["ink_9um@trace-v3b-step5000"],
        "fine_tuned_from": "ink_9um-s42",
        "in_sample": [],  # TRACE's own fine-tuning manifests are not recorded here
        "selection": _TRACE_SELECTION,
        "held_out": [],
        "source": TRACE_CARD + " ('Base model': ink_9um hybrid_3d2d-seed42/step-075000.pth); " + JO_TUD_17,
        "note": "only the lineage and the checkpoint-selection segments are recorded; TRACE's own fine-tuning "
                "list is not, so nothing is held out here",
    },
    {
        "key": "a-hn",
        "model": "A-hn s0-s2",
        "profiles": ["v8in@pherc0306b-ahn-s0", "v8in@pherc0306b-ahn-s1", "v8in@pherc0306b-ahn-s2"],
        "fine_tuned_from": "v8-in",
        "in_sample": _AHN_ANCHORS,  # the PHerc0306B pseudo-label surfaces are not recorded here
        "held_out": [],
        "source": AHN_CARD + " (start: v8-in, init_sha256 3b94548d... = the v8in@ink-8um-v8in pin); " + JO_TUD_17,
        "note": "only the lineage and the PHerc1447 anchor labels are recorded; the 0306B pseudo-label surfaces "
                "are not, so nothing is held out here. Teacher models (distillation) are not lineage.",
    },
]

_ID_KEYS = ("scroll", "segment_id", "public_w", "label_case", "label_w")
_CONFLICT_KEYS = ("scroll", "segment_id", "public_w", "label_case")


# ---------- identifiers ----------
def norm_scroll(s) -> str | None:
    """'PHerc0139', 'pherc0139', 'PHerc. 139', '0139' -> '139'; 'PHerc0343P' -> '343p' (fragment != scroll 343)."""
    if not s:
        return None
    t = re.sub(r"[^a-z0-9]", "", str(s).lower()).removeprefix("pherc")
    m = re.fullmatch(r"0*(\d+)([a-z0-9]*)", t)
    return (m.group(1) + m.group(2)) if m else t


def _scroll_from_part(part: str) -> str | None:
    """Accept a path component as a scroll name only when it looks like PHerc..., not a bare number."""
    m = re.match(r"pherc[ ._]*((?:paris)?[0-9]+[a-z]?[0-9]?)(?![0-9a-z])", str(part or "").lower())
    return norm_scroll(m.group(1)) if m else None


def _blank() -> dict:
    return dict.fromkeys(_ID_KEYS)


def segment_ids(segment, scroll=None, *, anywhere: bool = True) -> dict:
    """Identifiers in one name: {'scroll', 'segment_id', 'public_w', 'label_case', 'label_w'} (None if absent).

    With ``anywhere=False`` (path components) a 14-digit id only counts at the start of the
    name, as in a segment folder (``20250108000005-w030_...``), so a volume file named
    ``...-volume-20250728140407.zarr`` is not read as a segment id.
    """
    s = str(segment or "").strip()
    out = _blank()
    out["scroll"] = norm_scroll(scroll)
    m = re.fullmatch(r"(pherc[0-9a-z]+)-(w\d+)", s.lower())
    if m:  # ink_9um label case name
        out.update(label_case=m.group(0), label_w=m.group(2))
        out["scroll"] = out["scroll"] or norm_scroll(m.group(1))
        return out
    pattern = r"(?<!\d)(\d{14})(?!\d)" if anywhere else r"^(\d{14})(?!\d)"
    m = re.search(pattern, s)
    if m:
        out["segment_id"] = m.group(1)
    m = re.search(r"(?<![a-z0-9])(w\d{2,3})(?![0-9])", s.lower())
    if m:
        out["public_w"] = m.group(1)
    out["scroll"] = out["scroll"] or _scroll_from_part(s)
    return out


def resolve_input(*paths, scroll=None, segment=None) -> dict:
    """Resolve scroll/segment identifiers once, from the original input paths plus explicit overrides.

    Returns the identifier dict plus ``from`` (the inputs read) and ``ambiguous`` (keys
    whose path components disagree, e.g. two different scrolls). Explicit ``scroll`` /
    ``segment`` values settle their keys. An ambiguous input looks up as unknown.
    """
    seen: dict[str, set] = {k: set() for k in _ID_KEYS}
    used: list[str] = []
    for raw in paths:
        if raw is None or raw == "":
            continue
        text = str(raw)
        used.append(text)
        p = Path(text)
        names = [part for part in p.parts if part not in ("/", "\\")]
        if p.suffix and p.stem not in names:
            names.append(p.stem)
        for name in names:
            ids = segment_ids(name, anywhere=False)
            for key in _ID_KEYS:
                if ids.get(key):
                    seen[key].add(ids[key])
    merged = _blank()
    ambiguous = []
    for key in _ID_KEYS:
        values = sorted(seen[key])
        if len(values) == 1:
            merged[key] = values[0]
        elif len(values) > 1 and key in _CONFLICT_KEYS:
            ambiguous.append(key)
    if segment:
        override = segment_ids(segment, scroll=scroll or merged.get("scroll"))
        if any(override.get(k) for k in ("segment_id", "public_w", "label_case")):
            for key in ("segment_id", "public_w", "label_case", "label_w"):
                merged[key] = override.get(key)
            ambiguous = [k for k in ambiguous if k == "scroll"]
        if override.get("scroll") and not merged.get("scroll"):
            merged["scroll"] = override["scroll"]
    if scroll:
        merged["scroll"] = norm_scroll(scroll)
        ambiguous = [k for k in ambiguous if k != "scroll"]
    merged["from"] = used + ([f"--scroll {scroll}"] if scroll else []) + ([f"--segment {segment}"] if segment else [])
    merged["ambiguous"] = ambiguous
    return merged


def ids_from_paths(*paths, scroll=None, segment=None) -> dict:
    """Backwards-compatible name for :func:`resolve_input`."""
    return resolve_input(*paths, scroll=scroll, segment=segment)


def segment_paths(segment) -> list:
    """The original-input paths of a Segment that identify it (not render outputs or windows)."""
    if segment is None:
        return []
    return [
        getattr(segment, "surface_volume", None),
        getattr(segment, "surface_mesh", None),
        getattr(segment, "ct", None),
    ]


# ---------- matching ----------
def _spec_matches(spec: dict, ids: dict) -> str | None:
    if ids.get("scroll") is None or norm_scroll(spec.get("scroll")) != ids["scroll"]:
        return None
    if spec.get("segment_id") == "*":
        return f"{spec['scroll']} (all segments)"
    if spec.get("segment_id") and spec["segment_id"] == ids.get("segment_id"):
        return spec["segment_id"]
    if spec.get("label_case") and ids.get("label_case") and spec["label_case"].lower() == ids["label_case"]:
        return spec["label_case"]
    # a public w-number only counts when there is no public id to contradict it
    if (
        spec.get("public_w")
        and spec["public_w"] == ids.get("public_w")
        and not (spec.get("segment_id") and ids.get("segment_id"))
    ):
        return spec["public_w"]
    return None


def _contradicts(spec: dict, ids: dict) -> bool:
    """A known segment id whose recorded w-number differs from the input's w-number."""
    return bool(
        ids.get("scroll") is not None
        and norm_scroll(spec.get("scroll")) == ids["scroll"]
        and spec.get("segment_id") not in (None, "*")
        and spec.get("segment_id") == ids.get("segment_id")
        and spec.get("public_w")
        and ids.get("public_w")
        and spec["public_w"] != ids["public_w"]
    )


def _basis(spec: dict) -> str | None:
    raw = str(spec.get("basis") or spec.get("confidence") or "").strip().lower()
    for basis in (BASIS_STATED, BASIS_COMPLETE, BASIS_AFTER_UPLOAD):
        if raw.startswith(basis):
            return basis
    return raw or None


# ---------- registry ----------
def load_registry(path=None, include_builtin: bool = True) -> list[dict]:
    """Built-in entries plus a user JSON file (a list of entries, or {"entries": [...]}).

    User entries come first, so they win when both list a profile. Each entry should list
    exact ``profiles`` ids (a ``profile`` string is also accepted), and may name the entry it
    was fine-tuned from with ``fine_tuned_from`` (a ``key``). Held-out records need a
    ``basis`` (or ``confidence``) of ``stated`` or ``complete list``; ``after upload`` and
    anything else stay unknown.
    """
    reg: list[dict] = []
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data = data.get("entries", []) if isinstance(data, dict) else data
        for entry in data:
            entry = dict(entry)
            if "profiles" not in entry:
                ident = entry.get("profile") or entry.get("model")
                entry["profiles"] = [ident] if ident else []
            reg.append({"source": f"user file {path}", **entry})
    return reg + (list(BUILTIN) if include_builtin else [])


def _profiles_of(entry: dict) -> set[str]:
    names = list(entry.get("profiles") or [])
    if entry.get("profile"):
        names.append(entry["profile"])
    return {str(name) for name in names if name}


def lineage(profile_id, registry=None) -> list[dict]:
    """The registry entry for ``profile_id`` followed by the entries it was fine-tuned from."""
    reg = BUILTIN if registry is None else registry
    by_key = {e["key"]: e for e in reg if e.get("key")}
    chain: list[dict] = []
    entry = next((e for e in reg if profile_id in _profiles_of(e)), None)
    while entry is not None and entry not in chain:
        chain.append(entry)
        parent = entry.get("fine_tuned_from")
        entry = by_key.get(parent) if parent else None
    return chain


def _lineage_known(chain: list[dict]) -> bool:
    """Every link resolves to an entry (no dangling fine_tuned_from)."""
    if not chain:
        return False
    last = chain[-1].get("fine_tuned_from")
    return not last


def _public_input(ids: Mapping) -> dict:
    return {k: ids.get(k) for k in (*_ID_KEYS, "from", "ambiguous")}


def _empty(profile_id=None, ids: Mapping | None = None) -> dict:
    return dict(
        status=None,
        flag=LABEL[None],
        profile_id=profile_id,
        model=None,
        seen_by=None,
        inherited=False,
        lineage=[],
        matched=None,
        basis=None,
        confidence=None,
        source=None,
        note=None,
        evidence=[],
        input=_public_input(ids or {}),
    )


def _evidence(entry: dict, depth: int, use: str, spec: dict, hit: str) -> dict:
    return {
        "model": entry.get("model"),
        "relation": "checkpoint" if depth == 0 else "base model (fine-tuned from)",
        "use": use,
        "matched": hit,
        "basis": _basis(spec) if use == "held out" else None,
        "evidence": spec.get("evidence") or spec.get("confidence"),
        "note": spec.get("note"),
        "source": spec.get("source") or entry.get("source"),
    }


def lookup_ids(profile_id, ids: Mapping, registry=None) -> dict:
    """Training-overlap status for one (profile id, resolved identifier dict)."""
    base = _empty(profile_id, ids)
    if not profile_id:
        return base
    chain = lineage(profile_id, registry)
    if not chain:
        return {**base, "note": "profile not in the training-overlap table: training status unknown"}
    out = {
        **base,
        "model": chain[0].get("model"),
        "lineage": [e.get("model") for e in chain],
        "source": chain[0].get("source"),
    }
    if ids.get("ambiguous"):
        keys = ", ".join(ids["ambiguous"])
        return {**out, "note": f"ambiguous input (conflicting {keys} in the paths): pass --scroll / --segment"}
    if ids.get("scroll") is None:
        return {**out, "note": "no scroll name in the input: pass --scroll / --segment"}

    evidence: list[dict] = []
    for depth, entry in enumerate(chain):
        for key, use in (("in_sample", USE_TRAINING), ("selection", USE_SELECTION), ("held_out", "held out")):
            for spec in entry.get(key) or []:
                if _contradicts(spec, ids):
                    return {**out, "note": f"segment id {spec['segment_id']} is {spec['public_w']}, "
                                           f"not {ids.get('public_w')}: ambiguous input"}
                hit = _spec_matches(spec, ids)
                if hit:
                    evidence.append(_evidence(entry, depth, use, spec, hit))
                    break
    out["evidence"] = evidence
    trained = [e for e in evidence if e["use"] == USE_TRAINING]
    if trained:
        first = trained[0]
        own = first["relation"] == "checkpoint"
        selection = [e for e in evidence if e["use"] == USE_SELECTION]
        conflict = [e for e in evidence if e["use"] == "held out"]
        note = first.get("note")
        extra = []
        if not own:
            extra.append(f"inherited: {first['model']} (a model this checkpoint was fine-tuned from) trained on it")
        if selection:
            extra.append("this checkpoint also used it for checkpoint selection")
        if conflict:
            extra.append("a held-out record for it is overridden by the training list")
        return {
            **out,
            "status": True,
            "flag": LABEL[True] if own else "in-sample (inherited from base model)",
            "seen_by": "checkpoint" if own else "base model",
            "inherited": not own,
            "matched": first["matched"],
            "confidence": first.get("evidence"),
            "source": first["source"],
            "note": "; ".join([n for n in (note, *extra) if n]) or None,
        }
    if any(e["use"] == USE_SELECTION for e in evidence):
        return {**out, "matched": evidence[0]["matched"],
                "note": "used for checkpoint selection (not training): neither in-sample nor a clean held out"}

    held = [e for e in evidence if e["use"] == "held out"]
    accepted = [e for e in held if e["basis"] in ACCEPTED_HELD_OUT_BASES]
    if not held:
        return {**out, "note": "model known, segment not listed: training status unknown"}
    if not accepted:
        return {**out, "matched": held[0]["matched"], "basis": held[0]["basis"],
                "note": f"held-out record with basis {held[0]['basis']!r} does not establish held out: unknown"}
    covered = {e["model"] for e in accepted}
    complete_at_top = any(e["relation"] == "checkpoint" and e["basis"] == BASIS_COMPLETE for e in accepted)
    whole_chain = _lineage_known(chain) and all(e.get("model") in covered for e in chain)
    if not (complete_at_top or whole_chain):
        missing = [e.get("model") for e in chain if e.get("model") not in covered]
        return {**out, "matched": accepted[0]["matched"], "basis": accepted[0]["basis"],
                "note": "held out by part of the lineage only; no held-out evidence for "
                        + ", ".join(m or "?" for m in missing)}
    top = accepted[0]
    return {
        **out,
        "status": False,
        "flag": f"held out ({top['basis']})",
        "matched": top["matched"],
        "basis": top["basis"],
        "confidence": top["basis"],
        "source": top["source"],
        "note": top.get("note"),
    }


def lookup_profile(profile_id, scroll=None, segment=None, *, paths: Iterable[Any] = (), registry=None) -> dict:
    """Look up training overlap by exact profile id, original input paths and optional overrides."""
    ids = resolve_input(*tuple(paths), scroll=scroll, segment=segment)
    return lookup_ids(profile_id, ids, registry=registry)


def resolve_for(segment=None, ctx=None) -> dict:
    """The resolved identifiers for a run: ``ctx.overlap_input`` when the caller (the battery)
    already resolved them from the original input, else from the segment's input paths."""
    pre = getattr(ctx, "overlap_input", None) if ctx is not None else None
    if pre:
        return dict(pre)
    scroll = getattr(ctx, "scroll", None) if ctx is not None else None
    segment_name = getattr(ctx, "segment", None) if ctx is not None else None
    return resolve_input(*segment_paths(segment), scroll=scroll, segment=segment_name)


def lookup_segment(profile_id, segment=None, ctx=None, *, registry=None) -> dict:
    """Look up training overlap for a Segment and a RunContext (single ``infer`` / ``inspect``)."""
    return lookup_ids(profile_id, resolve_for(segment, ctx), registry=registry)


def is_in_sample(profile_id, scroll=None, segment=None, *, paths: Iterable[Any] = (), registry=None):
    """True = in-sample (the checkpoint or its base trained on it), False = held out, None = unknown."""
    return lookup_profile(profile_id, scroll, segment, paths=paths, registry=registry)["status"]


def short_flag(overlap: dict | None) -> str:
    """Compact label for tables: in-sample / in-sample (inherited) / held out / unknown."""
    if not overlap:
        return "unknown"
    status = overlap.get("status")
    if status is True:
        return "in-sample (inherited)" if overlap.get("inherited") else "in-sample"
    if status is False:
        return "held out"
    return "unknown"
