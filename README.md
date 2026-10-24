# ink-sdk

Standalone inference for Vesuvius ink models. Give it a `tifxyz` mesh plus a
local OME-Zarr CT, or an already rendered surface volume. It returns a 2D
probability map, a preview, a 3D zarr when the model produces one, receipts,
and a liveness verdict.

This package is inference only. It does not train, fine-tune, OCR, recognise
letters, or evaluate scientific accuracy.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Inputs](#inputs)
- [Models](#models)
- [Default profile templates](#default-profile-templates)
- [Scale, orientation and ROI](#scale-orientation-and-roi)
- [Outputs](#outputs)
- [Liveness and exit codes](#liveness-and-exit-codes)
- [CLI reference](#cli-reference)
- [Environment variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Status and limits](#status-and-limits)
- [License](#license)

## Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/LimeGS/ink-sdk.git
cd ink-sdk
python -m pip install -e '.[all]'
```

`[all]` is the easiest choice for a workstation or GPU host. If you want a
smaller install, use only the extras you need:

| Extra | Adds | Needed for |
|---|---|---|
| `render` | `scipy` | `--umbilicus` orientation and `--resample-from-um` |
| `canonical` | `torch` | `canonical_2um@...` |
| `hecate` | `torch`, `scipy` | `hecate@2.4um`, `hecate@9.6um` |
| `ink9um` | `torch`, `imagecodecs` | both `ink_9um@...` profiles |
| `fetch` | `huggingface_hub` | `ink-sdk fetch` |
| `test` | `pytest`, `pytest-cov` | the test suite |

The core package has no `torch` dependency. The model checkpoints are not
bundled; `ink-sdk fetch` downloads and verifies them. The external
`vc_render_tifxyz` binary is not installed by pip; see
[Inputs](#inputs).

## Quick start

```bash
# 1. See the five built-in, pinned profiles.
ink-sdk models

# 2. Download all pinned checkpoints and verify size + SHA-256.
ink-sdk fetch

# 3. Check Python, CUDA, renderer, VILLA and checkpoint presence.
ink-sdk doctor

# 4. Run a model on an already rendered surface volume.
ink-sdk infer \
  --profile hecate@2.4um \
  --surface-volume /data/segments/PHercParis4/surface-volumes/2.4um.zarr \
  --pixel-um 2.4 \
  --out runs/paris4-hecate
```

The default model directory is `~/.cache/ink-sdk/models`; override it with
`--models-dir` or `INK_SDK_MODELS`. `fetch` is opt-in: inference never
downloads weights.

`doctor` prints each dependency check and each profile. Profiles are
`verified`, `missing` or `present-but-unverified`; the last state means the
checkpoint exists but its size or SHA-256 does not match the profile. Fix that
before running inference.

## Inputs

There are two input modes. They produce the same kind of result, but only the
second one needs the CT renderer.

### A. Rendered surface volume

```bash
ink-sdk infer \
  --profile canonical_2um@r152-3ddec-v2-l5-epoch13 \
  --surface-volume /data/segments/PHAEDRUS/surface-volumes/2.399um.zarr \
  --pixel-um 2.399 \
  --out runs/phaedrus-canonical
```

`--surface-volume` accepts:

- a uint8 ZYX `.npy` file;
- a 3D `.tif` / `.tiff`;
- a directory of numbered TIFF slices (`0.tif`, `1.tif`, ...);
- a local Zarr group whose array is named `0`.

The volume must be unnormalised `uint8`. The scale is read from `pixel_um`,
`voxel_um` or `spacing_um` metadata (or an OME `multiscales` scale). If the
volume has no usable scale, pass `--pixel-um`; `--voxel-um` is accepted as the
same explicit isotropic scale. Explicit scales win over metadata.

`--surface-volume` can also point at a segment container that has a
`surface-volumes/` directory. The SDK chooses the only volume compatible with
the profile; if more than one is compatible it asks you to name one explicitly.

### B. tifxyz + local CT

```bash
ink-sdk infer \
  --profile canonical_2um@r152-3ddec-v2-l5-epoch13 \
  --tifxyz /data/segments/PHAEDRUS/mesh \
  --ct /data/ct/PHAEDRUS/level0.zarr \
  --voxel-um 2.4 \
  --normal outward \
  --out runs/phaedrus-canonical
```

The `tifxyz` directory must contain `x.tif`, `y.tif`, `z.tif` and a complete
`meta.json` (bbox, UUID and scale metadata). A tifxyz with only the TIFFs may
load in other tools but is rejected by the official renderer.

The CT must be a local, uint8 OME-Zarr with a readable level-0 array:

```text
/data/ct/PHAEDRUS/level0.zarr/
├── .zattrs
└── 0/
    ├── .zarray
    └── ... chunks
```

The level-0 `.zarray` must have dtype `|u1`. The `.zattrs` multiscale list must
only announce levels that actually exist; a list that mentions missing levels
makes `vc_render_tifxyz` fail. Zarr v3 CT input is not supported in this
version (surface volumes may be v3).

The renderer is the official `vc_render_tifxyz` from
volume-cartographer/VC3D, usually an AppImage `AppRun` on Linux. Point the SDK
at it in one of these ways:

```bash
--renderer /path/to/AppRun
export INK_SDK_RENDERER=/path/to/AppRun
export VC3D_APPRUN=/path/to/AppRun
```

The render is native level-0 sampling (`--scale 1.0`). The model scale is
reached later in `prepare`, not baked into the renderer. Renders are cached by
mesh, CT, orientation, scale, slice count and binary hash. The render receipt
records the command, binary hash, CT zarray hash, mesh hashes and cache key.

`--normal` is required for single-direction models (canonical and Hecate) when
you render from a mesh: the SDK will not guess which side of the surface is
"inward". `--umbilicus PATH` can supply the reading-frame orientation instead.
For the two-direction 9 µm models, omitting `--normal` is allowed; the receipt
marks the orientation as unverified.

To render once and infer more than once:

```bash
ink-sdk render \
  --profile canonical_2um@r152-3ddec-v2-l5-epoch13 \
  --tifxyz /data/segments/PHAEDRUS/mesh \
  --ct /data/ct/PHAEDRUS/level0.zarr \
  --voxel-um 2.4 \
  --normal outward \
  --out renders/phaedrus

ink-sdk infer \
  --profile canonical_2um@r152-3ddec-v2-l5-epoch13 \
  --surface-volume renders/phaedrus/render \
  --pixel-um 2.4 \
  --out runs/phaedrus-canonical
```

The render command writes `<out>/render` and does not load model weights.

`--roi y0,y1,x0,x1` is applied after scaling, in output-map pixels. The full
render is still cached; the receipt says
`roi_application = "post-render-in-prepare"`. The v2 pre-render mesh crop is not
implemented.

## Models

`ink-sdk models` prints the source of truth for these IDs; `ink-sdk models
--json` also shows the checkpoint path and SHA-256:

| Profile | Adapter | Native scale and tolerance | Depth | Direction | 3D output |
|---|---|---|---:|---|---|
| `ink_9um@scrollprize-seed43-step75000` | `ink9um` | 9.6 ± 0.5 µm | 17 | both | no |
| `ink_9um@reader-v2-step040000` | `ink9um` | 9.6 ± 0.5 µm | 17 | both | no |
| `canonical_2um@r152-3ddec-v2-l5-epoch13` | `canonical2um` | 2.399 ± 0.05 µm | 62 | forward | no |
| `hecate@2.4um` | `hecate` | 2.4 ± 0.048 µm | 64 | forward | with `--output-3d` |
| `hecate@9.6um` | `hecate` | 9.6 ± 0.192 µm | 16 | forward | with `--output-3d` |

The two 9 µm profiles use the same adapter and preprocessing; they differ in
checkpoint and source metadata (seed43 also declares `no_ink_floor = 0.25`).
The canonical profile is a ResNet-152 + 3D decoder. The Hecate profiles share
one architecture and checkpoint format at two sampling scales.

### 9 µm: seed43 and reader-v2

Use either profile ID:

```bash
export VILLA=/opt/ScrollPrize-villa

ink-sdk infer \
  --profile ink_9um@scrollprize-seed43-step75000 \
  --surface-volume /data/segments/PHerc0306B/surface-volumes/9.362um.zarr \
  --pixel-um 9.362 \
  --villa "$VILLA" \
  --out runs/seed43

ink-sdk infer \
  --profile ink_9um@reader-v2-step040000 \
  --surface-volume /data/segments/PHerc0306B/surface-volumes/9.362um.zarr \
  --pixel-um 9.362 \
  --villa "$VILLA" \
  --out runs/reader-v2
```

These models run `koine_machines.inference.infer` from a
`ScrollPrize/villa` checkout. The checkout must expose
`ink-detection/koine_machines/inference/infer.py`; `--villa` or `VILLA`
selects it. The VILLA environment must also make the `vesuvius` package
importable, because the 9 µm model builder imports it. `ink-sdk doctor --villa
"$VILLA"` checks both imports.

`imagecodecs` is required for the upstream TIFF maps and is installed by the
`ink9um` extra. If you use a separate upstream interpreter, pass it with
`--python /path/to/python`.

The default execution is CUDA with `bf16`. On CPU, use `--amp fp32` (the
adapter maps it to upstream's `default` path).

Both profiles produce `forward`, `reverse` and `mean` maps. The primary output
is `forward` unless you pass `--primary reverse` or `--primary mean`.

For an 8.64 µm CT, the only supported route is the opt-in XY resample:

```bash
ink-sdk infer \
  --profile ink_9um@scrollprize-seed43-step75000 \
  --tifxyz /data/segments/PHerc0306B/mesh \
  --ct /data/ct/PHerc0306B/level0.zarr \
  --voxel-um 8.64 \
  --resample-from-um 8.64 \
  --villa "$VILLA" \
  --out runs/seed43-8.64
```

This does XY linear interpolation from 8.64 to 9.362 µm with no prefilter; Z
is untouched. It needs `scipy`, which is in the `render` extra, not the
`ink9um` extra. A 2.4 µm source does not need this flag: the 9 µm profiles
accept it through 4× mean pooling to 9.6 µm.

`seed43` declares a `no_ink_floor` of 0.25 in its profile. That value is
recorded in the map receipt; the SDK does **not** threshold or rescale the
probability map with it.

### canonical 2 µm

```bash
ink-sdk infer \
  --profile canonical_2um@r152-3ddec-v2-l5-epoch13 \
  --surface-volume /data/segments/PHAEDRUS/surface-volumes/2.399um.zarr \
  --pixel-um 2.399 \
  --out runs/phaedrus-canonical
```

From a mesh and CT, supply the normal explicitly because this profile has one
reading direction:

```bash
ink-sdk infer \
  --profile canonical_2um@r152-3ddec-v2-l5-epoch13 \
  --tifxyz /data/segments/PHAEDRUS/mesh \
  --ct /data/ct/PHAEDRUS/level0.zarr \
  --voxel-um 2.4 \
  --normal outward \
  --out runs/phaedrus-canonical
```

The adapter runs the vendored ResNet-152 + 3D decoder in-process with 62
central layers, 256 px tiles, stride 128 and a Hann window. The profile accepts
2.399 ± 0.05 µm; a 2.4 µm surface is within tolerance. It does not accept
8.64 or 9.x µm input and has no pooling path. The output is forward-only.
`--deterministic` enables PyTorch deterministic algorithms for this lane.

The input must have at least 62 layers; the adapter selects the central stack.
For shorter volumes, render more layers first.

### Hecate 2.4 µm

```bash
ink-sdk infer \
  --profile hecate@2.4um \
  --surface-volume /data/segments/PHercParis4/surface-volumes/2.4um.zarr \
  --pixel-um 2.4 \
  --output-3d \
  --out runs/paris4-hecate-2.4
```

From a mesh and CT:

```bash
ink-sdk infer \
  --profile hecate@2.4um \
  --tifxyz /data/segments/PHercParis4/mesh \
  --ct /data/ct/PHercParis4/level0.zarr \
  --voxel-um 2.4 \
  --normal outward \
  --output-3d \
  --out runs/paris4-hecate-2.4
```

Hecate 2.4 uses 64 layers and 256 px patches with stride 128. It accepts
2.4 ± 0.048 µm and does not resample. `--output-3d` also writes `ink_3d.zarr`
alongside the 2D map. The 3D volume is uint8 probability; only the depth
interval that was evaluated is populated, and the attrs state the interval.

The checkpoint normally runs in fp32. `--amp bf16` is available on CUDA;
`--device cpu --amp fp32` is the safe CPU path.

### Hecate 9.6 µm

```bash
ink-sdk infer \
  --profile hecate@9.6um \
  --surface-volume /data/segments/PHerc0306B/surface-volumes/9.6um.zarr \
  --pixel-um 9.6 \
  --out runs/phaedrus-hecate-9.6
```

Hecate 9.6 uses 16 layers and 64 px patches with stride 32. It accepts
9.6 ± 0.192 µm natively. A 2.4 µm source can also be used: the shared scale
policy pools it 4× in Z, Y and X to 9.6 µm. A native 8.64 µm source is not
accepted by this profile.

`--normal` is required when rendering from a mesh because the profile emits
one reading direction. `--output-3d` writes a 3D zarr as with Hecate 2.4.

## Default profile templates

There is no Jinja/YAML model template and no per-model run config. The defaults
are split into three layers:

| Layer | Where it lives | What it controls |
|---|---|---|
| Adapter defaults | adapter code in `src/ink_sdk/adapters/` | tiling, preprocessing, checkpoint loading, limits |
| Render defaults | `src/ink_sdk/renderers/vc_render_tifxyz.py` and CLI flags | native scale, slice step, orientation, cache |
| Model profile | a TOML in `src/ink_sdk/profiles/` | checkpoint identity, model scale/depth, execution and output |

The five TOMLs are the default **profile templates**. They are data, not code:
you can copy one, change model facts, and use it without forking the SDK.
Changing an adapter's preprocessing is a code change, not a profile edit.

### Built-in templates

```text
src/ink_sdk/profiles/
├── ink_9um-scrollprize-seed43-step75000.toml
├── ink_9um-reader-v2-step040000.toml
├── canonical-2um.toml
├── hecate-2.4um.toml
└── hecate-9.6um.toml
```

In a source checkout, the top-level `profiles/` symlink points at this same
directory. In an installed wheel, use the packaged path:

```bash
python -c "import ink_sdk, pathlib; print(pathlib.Path(ink_sdk.__file__).parent / 'profiles')"
```

`ink-sdk models` lists the profiles currently visible to the process. The
default templates are always available; no copy is required for normal use.

### Use a profile by ID, suffix or file

`--profile` accepts:

- an exact ID: `hecate@2.4um`;
- a unique suffix: `2.4um`;
- a path to a TOML file: `./profiles/my-hecate.toml`.

A custom file is loaded directly and does not need to be in any search
directory:

```bash
ink-sdk inspect --profile ./profiles/my-hecate.toml \
  --surface-volume /data/segments/PHercParis4/surface-volumes/2.4um.zarr \
  --pixel-um 2.4
```

`inspect` is always a dry run: it loads the profile and prints the scale,
render and inference plan without loading checkpoint weights.

### Modify a template

1. Copy a TOML out of the installed package or source checkout. Do not edit
   `site-packages`; the next install will overwrite it.
2. Change `id` to a new unique value. Do not reuse a built-in ID.
3. Keep `adapter` and `family` compatible. The allowed pairs are
   `ink9um`/`surface-volume-2d`, `canonical2um`/`surface-volume-2d`, and
   `hecate`/`surface-volume-3d2d`.
4. Put the checkpoint under your models directory and update
   `checkpoint.path`, `checkpoint.upstream_path`, `checkpoint.sha256` and
   `checkpoint.bytes`. The digest and byte count are mandatory; there are no
   default checkpoints.
5. Update `source.repo`, `source.revision` and the model facts if you are
   registering a different release.
6. Validate before running:

```bash
ink-sdk inspect --profile ./profiles/my-hecate.toml \
  --surface-volume /data/segments/PHercParis4/surface-volumes/2.4um.zarr \
  --pixel-um 2.4

ink-sdk fetch --profile ./profiles/my-hecate.toml --models-dir ~/.cache/ink-sdk/models
```

A minimal, annotated Hecate profile looks like this. Replace the placeholder
digest and byte count before use:

```toml
schema = "ink-sdk.profile.v1"
id = "my-hecate@2.4um"
adapter = "hecate"
family = "surface-volume-3d2d"

[source]
repo = "scrollprize/hecate"
revision = "9cb86e500e944b11a06a7020403cde5dffb5bcb2"
upstream = "scrollprize/hecate"
upstream_revision = "9cb86e500e944b11a06a7020403cde5dffb5bcb2"
url = "https://huggingface.co/scrollprize/hecate"
license = "MIT"

[checkpoint]
path = "hecate/my-hecate.pth"       # relative to the models dir; no ".."
upstream_path = "my-hecate.pth"     # path inside the Hugging Face repo
sha256 = "<64 lowercase hex>"       # must match the pinned file
bytes = 123456789                   # must be positive and exact

[input]
kind = "surface-volume"
family = "surface-volume-3d2d"
scale_um = 2.4
scale_tolerance_um = 0.048
depth = 64
normalization = "divide-200-no-clip"
divisor = 200.0
tile = 256
stride = 128
min_valid_ratio = 0.0

[execution]
direction = "forward"
batch = 1
amp = "fp32"

[output]
primary = "forward"
no_ink_floor = 0.0
floor_provenance = "not-declared"
```

Important field semantics:

- `input.scale_um` and `scale_tolerance_um` drive the accept/pool/resample
  policy. Do not widen tolerance to force an incompatible stretch; the
  adapter still has its own geometry limits.
- `input.depth` is the model's required layer count. The canonical adapter
  selects the central stack; Hecate uses its patch depth; the 9 µm worker
  receives the prepared volume and its checkpoint owns the model depth.
- `input.normalization` is declarative. Canonical reads `input.divisor`;
  Hecate and 9 µm use the divisor/normalization stored in the checkpoint, so
  the profile `divisor` is metadata for those two adapters.
- `input.tile`, `stride`, `overlap` and `min_valid_ratio` are adapter-specific:
  canonical uses tile/stride/min-valid, Hecate uses stride, and 9 µm passes
  overlap to the upstream worker.
- `execution.direction = "both"` is valid for the 9 µm profiles. Canonical only
  supports `forward`; Hecate profiles are declared `forward`.
- `execution.amp` is the profile default. `--amp` overrides it where the
  adapter supports a precision choice.
- `output.primary` selects the map used for `preview.tif` and the non-forward
  `probability_primary.npy`; `probability.npy` always stays the forward map.
- `output.no_ink_floor` is recorded in the receipt. It is not applied as a
  threshold by this SDK; do not change it expecting a different probability
  map.

### Load a directory of custom templates

Use `INK_SDK_PROFILES` for profiles you want to see in `ink-sdk models` and
`ink-sdk doctor`:

```bash
mkdir -p ~/ink-profiles
cp src/ink_sdk/profiles/hecate-2.4um.toml ~/ink-profiles/my-hecate.toml
# edit id, checkpoint and model facts

INK_SDK_PROFILES=$HOME/ink-profiles ink-sdk models
INK_SDK_PROFILES=$HOME/ink-profiles ink-sdk doctor

INK_SDK_PROFILES=$HOME/ink-profiles ink-sdk infer \
  --profile my-hecate@2.4um \
  --surface-volume /data/segments/PHercParis4/surface-volumes/2.4um.zarr \
  --pixel-um 2.4 \
  --out runs/my-hecate
```

Every `*.toml` in each `INK_SDK_PROFILES` directory is loaded. Directories are
searched before the packaged profiles; duplicate IDs are a configuration error,
not an override. Use a new `id` for a modified template.

For repeatable runs, keep the profile TOML, the command line and the generated
`INK_SDK_RECEIPT.json` together. The receipt is the record of what actually ran,
including the checkpoint digest and scale plan.

## Scale, orientation and ROI

The SDK refuses stretches that the profiles do not support:

| Source | 9 µm profiles | canonical 2.399 | Hecate 2.4 | Hecate 9.6 |
|---|---|---|---|---|
| 2.4 µm | 4× mean pool -> 9.6 | native | native | 4× mean pool -> 9.6 |
| 2.399 µm | 4× mean pool -> 9.596 | native | native | 4× mean pool -> 9.596 |
| 8.64 µm | resample -> 9.362 with `--resample-from-um 8.64` | reject | reject | reject |
| 9.362 µm | native | reject | reject | reject |
| 9.596 µm | native | reject | reject | native |

The 8.64 µm resample is XY linear interpolation to 9.362 µm; Z is untouched
and there is no prefilter. If no row applies, the SDK rejects the source
instead of silently stretching it.

`prepare` pools or resamples first, then applies ROI in output-map pixels.
The render itself is not cropped to the ROI; the full render stays cached and
the receipt says `roi_application = "post-render-in-prepare"`. If a renderer
does not support `--auto-crop`, the full CT-sized canvas is rendered and no
post-render bbox trim is applied.

With `--umbilicus`, normal direction and reading orientation come from the
umbilicus/reading frame. Without one, depth orientation is ambiguous:

- the 9 µm profiles run both directions and record the orientation as
  unverified;
- single-direction profiles require `--umbilicus` or an explicit
  `--normal inward|outward`;
- `--normal` without an umbilicus is recorded as
  `orientation_source = "user-declared"`; reading orientation remains
  unverified.

## Outputs

`ink-sdk infer --out out/` writes:

- `probability.npy` — forward map, float32 in `[0, 1]`, shape `(y, x)`.
- `probability_reverse.npy`, `probability_mean.npy` — when produced.
- `probability_primary.npy` — a copy of the selected map when `--primary` is
  `reverse` or `mean`; `probability.npy` remains the forward map.
- `preview.tif` — uint8 display preview of the primary map.
- `ink_3d.zarr` — when the adapter produces 3D output (currently Hecate).
- `render/0` and `render/RENDER_RECEIPT.json` — when this run rendered from
  `tifxyz` + CT.
- `INK_SDK_RECEIPT.json` — profile, checkpoint, inputs, scale plan, render,
  execution, map statistics, liveness, environment and non-claims.
- `LANE_NOT_USABLE` — written before exit 3 when the primary map is empty or
  degenerate.

The receipt is the source of truth for scale, orientation, ROI origin, model
identity and checkpoint hash. A probability map is not evidence of ink,
letters, or a First Letter.

If `out/` already exists and is not empty, inference refuses to overwrite it
unless you pass `--overwrite`. The CLI also refuses to delete the current
directory, your home directory, or the filesystem root.

## Liveness and exit codes

The liveness gate is ported from Campaign X and is evaluated per map:

- `EMPTY`: no valid pixels.
- `DEGENERATE`: `p99 - p50 < 0.05`, `std < 0.02`, or more than 90% of valid
  pixels within 0.05 of 0.5.
- `ALIVE`: none of the above.

`--on-degenerate fail` (default) exits 3 after keeping the map and receipt for
diagnosis. `--on-degenerate warn` continues and exits 0. The warning is for
diagnosis, not a claim that the map is scientifically usable.

Exit codes: `0` success, `2` validation/dependency/render error, `3` refused
degenerate lane.

## CLI reference

```text
ink-sdk models [--json]
ink-sdk fetch [--profile ID|TOML ...] [--models-dir PATH] [--overwrite]
ink-sdk doctor [--json] [--renderer PATH] [--models-dir PATH] [--villa PATH]
ink-sdk inspect --profile ID|TOML (--surface-volume PATH | --tifxyz MESH --ct CT) ...
ink-sdk render --tifxyz MESH --ct CT [--profile ID|TOML] [--out PATH] [--dry-run]
ink-sdk infer (--surface-volume PATH | --tifxyz MESH --ct CT) --profile ID|TOML ...
```

`inspect` is a planning command: it does not load weights and always emits
JSON. `render --dry-run` prints the render plan without invoking the binary.
`infer --dry-run` prints the receipt plan and does not render or infer.

Useful `infer` options:

| Option | Meaning |
|---|---|
| `--roi y0,y1,x0,x1` | Half-open ROI in output-map pixels, applied after scaling. |
| `--primary forward\|reverse\|mean` | Which produced map is primary (preview and `probability_primary.npy`). |
| `--device auto\|cpu\|cuda\|mps` | Runtime device. |
| `--amp bf16\|fp16\|fp32\|default` | Precision override for adapters that support it. |
| `--resample-from-um 8.64` | Opt-in 8.64 µm -> 9.362 µm XY resample for 9 µm profiles. |
| `--normal auto\|inward\|outward` | Declared depth direction when there is no umbilicus. |
| `--umbilicus PATH` | Use a reading-frame/umbilicus orientation. |
| `--output-3d` | Also write `ink_3d.zarr` for Hecate. |
| `--python PATH` | Interpreter used for the upstream 9 µm worker. |
| `--villa PATH` | VILLA checkout for the 9 µm profiles. |
| `--renderer PATH` | Official `vc_render_tifxyz` binary or `AppRun`. |
| `--no-render-cache` | Use a temporary render cache for this run. |
| `--threads N` | Renderer thread count. |
| `--max-render-gb N` | Refuse a render whose estimated array exceeds this size. |
| `--overwrite` | Replace a non-empty output directory. |
| `--deterministic` | Deterministic PyTorch algorithms for canonical. |
| `--dry-run` | Print the plan without rendering or inferring. |
| `--json` | Print the receipt/report as JSON. |

Use `ink-sdk infer --help`, `ink-sdk render --help`, etc. for the complete
Typer-generated option list.

## Environment variables

| Variable | Used for | Default |
|---|---|---|
| `INK_SDK_MODELS` | Checkpoint root | `~/.cache/ink-sdk/models` |
| `INK_SDK_CACHE` | Inference cache root | `~/.cache/ink-sdk` |
| `INK_SDK_PROFILES` | Extra profile directories (`os.pathsep`-separated) | packaged `profiles/` only |
| `INK_SDK_RENDERER` | `vc_render_tifxyz`/`AppRun` path | unset |
| `VC3D_APPRUN` | Alternative renderer path | unset |
| `VILLA` | VILLA checkout for 9 µm | unset |

`--models-dir`, `--cache-dir`, `--renderer` and `--villa` override the
corresponding environment variables where the command exposes them.

## Troubleshooting

**`GroupNotFoundError` or "not an OME-Zarr with a readable 0/.zarray"**

The CT reader expects OME-Zarr v2 with `<ct>/0/.zarray`. Check the path; do
not point at a `.zattrs` file or at a Zarr v3 CT root. Surface volumes may use
other layouts, but a CT passed to `--ct` must satisfy this contract.

**`vc_render_tifxyz` not found or renderer flags are missing**

Install the official binary and set `--renderer`, `INK_SDK_RENDERER` or
`VC3D_APPRUN`. Run `ink-sdk doctor --renderer /path/to/AppRun` to see which
flags the binary reports. If the binary does not support `--auto-crop`, the SDK
renders the full canvas and records that choice.

**`CT has no scale; pass --voxel-um`**

The `.zattrs` at the CT root and `0/.zattrs` did not contain `pixel_um`,
`voxel_um`, `spacing_um` or an OME scale. Pass `--voxel-um`, or fix the
metadata. A surface volume can use `--pixel-um` instead.

**`needs ... um, but the source is ... um`**

The profile does not accept that scale. Check the scale matrix above. For
8.64 µm CT and a 9 µm profile, add `--resample-from-um 8.64`. For a 2.4 µm
source and a 9.6 µm profile, pooling is automatic; do not add resample flags.

**`unverified-orientation` or a request for `--normal`**

You are rendering a mesh for canonical or Hecate without an umbilicus. Pass
`--normal inward` or `--normal outward` after checking the reading frame, or
pass `--umbilicus`. The receipt records whether the orientation was
user-declared.

**`missing-imagecodecs` / upstream TIFF compression error**

Install the 9 µm extra:

```bash
python -m pip install -e '.[ink9um]'
```

**`missing-villa` or `No module named 'vesuvius'`**

Set `VILLA` or pass `--villa` to a checkout containing
`ink-detection/koine_machines/inference/infer.py`. The VILLA environment must
also provide `vesuvius`. `ink-sdk doctor --villa "$VILLA"` checks the imports.

**Checkpoint size or SHA-256 mismatch**

The profile pins both values. Re-run `ink-sdk fetch --profile ID` or point
`--models-dir` at the verified directory. Do not edit the digest to make a
mismatch pass.

**`DEGENERATE` / exit 3**

The gate says the primary map is empty, collapsed, saturated or otherwise not
decision-bearing. Keep the receipt and map for diagnosis. `--on-degenerate
warn` lets the command finish but does not make the map usable for screening.

**CUDA requested but not available**

Use `--device cpu`, or run on a CUDA host. Canonical and Hecate run on CPU in
fp32; the 9 µm upstream worker is intended for CUDA and may be slow on CPU.

**Output directory is not empty**

Pass `--overwrite` after checking that the directory is the one you want to
replace. The CLI protects the current directory, the home directory and the
filesystem root.

## Development

```bash
python -m pip install -e '.[test,all]'
PYTHONPATH=src python -m pytest -q
python -m build --wheel
```

The default tests use synthetic meshes/CT, a fake `vc_render_tifxyz`
executable, fake model objects, and a fake `koine_machines` invocation.
Real-checkpoint and GPU runs are manual. Synthetic tests prove the plumbing,
not scientific validity.

To add a model, add a profile TOML and use an existing adapter if its
architecture matches. A new architecture needs an adapter class, not a
template string.

## Status and limits

Implemented and tested:

- five pinned profiles and the scale matrix;
- native/pool/resample preparation and ROI cropping;
- dummy end-to-end pipeline, receipts and liveness;
- canonical/Hecate/9 µm adapters against fake models and a fake upstream;
- all five profiles end to end on a Vast.ai CUDA host with the pinned
  checkpoints:
  - both 9 µm profiles on a real PHerc0306B 8.64 µm CT and real tifxyz
    (primary maps ALIVE);
  - canonical 2 µm on a 2.4 µm render (ALIVE on a non-ink window);
  - Hecate 9.6 µm on a synthetic 9.6 µm surface volume (ALIVE; no real
    9.6 µm mesh was available);
  - Hecate 2.4 µm on a real PHercParis4 2.4 µm CT crop, including an
    ink-rich 256×256 window (ALIVE) and `--output-3d`;
- official `vc_render_tifxyz` argv, orientation, cache, receipt and fallbacks
  on the render-dependent runs;
- CLI `models`, `doctor`, `inspect`, `render`, `infer`, `fetch`.

Not verified:

- a real 2.4 µm tifxyz with along-surface geometry; the 2.4 µm GPU test used
  a flat 256×256 mesh inside a real PHercParis4 surface volume;
- scientific correctness, calibration, ink calls, or OCR. Liveness only says
  that a map carries a decision, not that the decision is right;
- Hecate 2.4 µm on the synthetic 2.4 µm volume is DEGENERATE and is treated as
  out of distribution, not as a passing lane;
- a fully inked canonical window saturates near 0.95 and the liveness gate
  flags it DEGENERATE; that is a gate limitation on saturated maps.

The SDK deliberately fails closed when a checkpoint, renderer, scale,
orientation, or dependency is missing.

## License

MIT for this package. Vendored architecture code and checkpoint provenance are
listed in `THIRD_PARTY_NOTICES.md`; the external `vc_render_tifxyz` binary and
model weights keep their own licenses.
