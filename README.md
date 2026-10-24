# ink-sdk

Standalone Python inference for Vesuvius ink models. Give it a `tifxyz` mesh
and a local OME-Zarr CT, or an already rendered surface volume; it returns the
same kind of result every time: a 2D probability map, a preview, a zarr when
the model produces 3D output, two receipts, and a liveness verdict.

This is inference only. There is no training, fine-tuning, OCR, text
recognition, or scientific evaluation in this package.

## Install

```bash
git clone https://github.com/LimeGS/ink-sdk.git
cd ink-sdk
python -m pip install -e '.[all]'
ink-sdk models
```

The model checkpoints are not bundled. Fetch the pinned files explicitly:

```bash
ink-sdk fetch --profile hecate@2.4um
ink-sdk fetch                       # all five profiles
```

Fetch verifies the pinned byte size and SHA-256 before placing a checkpoint in
`~/.cache/ink-sdk/models` (override with `--models-dir`). Inference never
downloads weights.

## Inputs

### Rendered surface volume

```bash
ink-sdk infer \
  --profile canonical_2um@r152-3ddec-v2-l5-epoch13 \
  --surface-volume /data/segments/PHAEDRUS/surface-volumes/2.399um.zarr \
  --pixel-um 2.399 \
  --out out/
```

A surface volume is a uint8 ZYX array in `.npy`, a 3D TIFF, a numbered TIFF
stack, or a zarr group with array `0`. If the scale is not in its metadata,
pass `--pixel-um` (or `--voxel-um`).

### tifxyz + local CT

```bash
ink-sdk render \
  --profile canonical_2um@r152-3ddec-v2-l5-epoch13 \
  --tifxyz /data/segments/PHAEDRUS/mesh \
  --ct /data/ct/PHAEDRUS/level0.zarr \
  --normal outward \
  --out out/

ink-sdk infer \
  --profile canonical_2um@r152-3ddec-v2-l5-epoch13 \
  --tifxyz /data/segments/PHAEDRUS/mesh \
  --ct /data/ct/PHAEDRUS/level0.zarr \
  --normal outward \
  --out out/
```

The renderer is the official `vc_render_tifxyz` from volume-cartographer/VC3D,
usually an AppImage `AppRun` on Linux. Point the SDK at it with one of:

```bash
--renderer /path/to/AppRun
export INK_SDK_RENDERER=/path/to/AppRun
export VC3D_APPRUN=/path/to/AppRun
```

The render is native level-0 sampling (`--scale 1.0`). Model scale is reached
later by `prepare`, not baked into the renderer. The renderer receipt records
the command, binary hash, CT zarray hash, mesh hashes, orientation and cache
key. Renders are cached by all of those inputs.

v0 does not download remote CT. Put the level-0 OME-Zarr on local disk.

## Profiles

| id | adapter | scale | depth | direction |
|---|---|---:|---:|---|
| `ink_9um@scrollprize-seed43-step75000` | `ink9um` | 9.6 µm | 17 | both |
| `ink_9um@reader-v2-step040000` | `ink9um` | 9.6 µm | 17 | both |
| `canonical_2um@r152-3ddec-v2-l5-epoch13` | `canonical2um` | 2.399 µm | 62 | forward |
| `hecate@2.4um` | `hecate` | 2.4 µm | 64 | forward |
| `hecate@9.6um` | `hecate` | 9.6 µm | 16 | forward |

The TOMLs are in `profiles/` (a symlink to the packaged copy). Each profile
pins the upstream repo revision, checkpoint path, byte size, SHA-256, input
scale, tolerance, depth, normalisation, direction and primary output.

The three adapters are deliberately different:

- **ink9um** runs `koine_machines.inference.infer` from a
  `ScrollPrize/villa` checkout. Set `VILLA` or pass `--villa`. It needs the
  upstream inference dependencies and a CUDA-capable environment for its
  default bf16 path. The VILLA environment must also make the `vesuvius`
  package importable (the 9 µm model builder uses it); `doctor --villa` checks
  both imports. On CPU, pass `--amp default` or `--amp fp32`. It produces
  forward, reverse and mean when the profile runs both directions.
- **canonical2um** runs the vendored ResNet-152 + 3D decoder in-process. It
  uses 62 central layers, 256 px tiles with stride 128, a Hann window, and
  `/255` normalisation. It needs `torch`.
- **hecate** runs the vendored Hecate architecture in-process, 64 layers at
  2.4 µm or 16 layers at 9.6 µm. `--output-3d` additionally writes
  `ink_3d.zarr`. It needs `torch`.

## Scale rules

The SDK refuses stretches that the profiles do not support:

- 2.4 µm source -> 9.6 µm profile: 4x mean pooling (output ~9.596 µm).
- 2.399 / 2.4 µm source -> canonical or Hecate 2.4: native.
- 9.362 / 9.596 µm source -> 9.6 µm: native.
- 8.64 µm -> 9.6 µm: only `ink9um`, only with
  `--resample-from-um 8.64`; XY linear interpolation to 9.362 µm, Z untouched.
  There is no prefilter, matching the method Campaign X measured.
- A 9.362 µm source cannot feed canonical or Hecate 2.4/9.6; the tolerance
  rejects it rather than silently stretching.

`prepare` pools or resamples first, then applies ROI in output-map pixels.
The render itself is not cropped to the ROI; the full render stays cached and
the receipt says `roi_application = "post-render-in-prepare"`. This trades peak
disk/RAM for exact output shape and avoids a second geometry mapping. The v2
design's pre-render mesh crop (32 px margin) is therefore not implemented in
this version; it is the main deliberate deviation for ROI. If a binary does not
support `--auto-crop`, the full CT-sized canvas is rendered and no post-render
bbox trim is applied.

## Orientation

With `--umbilicus`, normal direction and reading orientation are ported from
Campaign X's `render_vc3d.py` (`traspuesta`, `filas_invertidas`,
`columnas_invertidas`). Without an umbilicus, depth orientation is ambiguous:

- `ink9um` runs both directions, so it proceeds with both stacks and records
  the orientation as unverified.
- Single-direction profiles (canonical and Hecate) require an umbilicus or an
  explicit `--normal inward|outward`. The SDK will not guess the depth
  direction for a map that has only one direction.

`--normal` without an umbilicus is recorded as `orientation_source =
"user-declared"`; reading orientation remains unverified.

## Outputs

`ink-sdk infer --out out/` writes:

- `probability.npy` — primary forward map, float32 in `[0, 1]`, shape `(y, x)`.
- `probability_reverse.npy`, `probability_mean.npy` — when produced.
- `preview.tif` — uint8 display preview of the primary map.
- `ink_3d.zarr` — when the adapter produces 3D output.
- `render/0` and `render/RENDER_RECEIPT.json` — when this run rendered.
- `INK_SDK_RECEIPT.json` — profile, checkpoint, inputs, scale plan, render,
  execution, map statistics, liveness, environment and non-claims.
- `LANE_NOT_USABLE` — written before exit 3 when the primary map is empty or
  degenerate.

The receipt is the source of truth for scale, orientation, ROI origin, model
identity and checkpoint hash. A probability map is not evidence of ink,
letters, or a First Letter.

## Liveness gate

The gate is ported from Campaign X:

- `EMPTY`: no valid pixels.
- `DEGENERATE`: `p99 - p50 < 0.05`, `std < 0.02`, or more than 90% of valid
  pixels within 0.05 of 0.5.
- `ALIVE`: none of the above.

`--on-degenerate fail` (default) exits 3 after keeping the map and receipt for
diagnosis. `--on-degenerate warn` continues and exits 0.

Exit codes: `0` success, `2` validation/dependency/render error, `3` refused
degenerate lane.

## CLI

```text
ink-sdk models [--json]
ink-sdk fetch [--profile ID ...] [--models-dir PATH] [--overwrite]
ink-sdk doctor [--json] [--renderer PATH] [--models-dir PATH] [--villa PATH]
ink-sdk inspect --profile ID (--surface-volume PATH | --tifxyz MESH --ct CT) ...
ink-sdk render --tifxyz MESH --ct CT [--out PATH] [--dry-run]
ink-sdk infer (--surface-volume PATH | --tifxyz MESH --ct CT) --profile ID ...
```

`inspect` planning does not load weights. `render --dry-run` prints the render
plan without invoking the binary. `infer --dry-run` prints the receipt plan and
does not render or infer.

Useful inference options: `--roi y0,y1,x0,x1`, `--primary forward|reverse|mean`,
`--device auto|cpu|cuda|mps`, `--amp bf16|fp16|fp32`, `--resample-from-um 8.64`,
`--normal auto|inward|outward`, `--umbilicus PATH`, `--output-3d`,
`--no-render-cache`, `--threads N`, `--max-render-gb N`, `--overwrite`,
`--deterministic`.

## Development

```bash
python -m pip install -e '.[test,render,canonical,hecate,ink9um]'
PYTHONPATH=src python -m pytest -q
```

The tests use synthetic meshes/CT, a fake `vc_render_tifxyz` executable, fake
model objects, and a fake `koine_machines` invocation. They do not prove a real
checkpoint produces a scientifically valid map.

## Status and limits

Implemented and tested synthetically:

- five pinned profiles and the scale matrix;
- native/pool/resample preparation and ROI cropping;
- dummy end-to-end pipeline, receipts and liveness;
- canonical/Hecate/9 µm adapters against fake models and a fake upstream;
- the real Hecate 2.4 checkpoint loaded and ran a 2D CPU inference on a
  synthetic 64×256×256 volume (near-zero output, as expected for random CT;
  this is a smoke test, not a scientific result);
- official renderer argv, orientation, cache, receipt and fallbacks;
- CLI `models`, `doctor`, `inspect`, `render`, `infer`, `fetch`.

Not verified in this checkout:

- a real end-to-end golden run, because the pinned checkpoints are not local;
- the real Linux `vc_render_tifxyz` binary, which is not available on macOS;
- a real `koine_machines` run, which needs a VILLA checkout, its `vesuvius`
  dependency, and a compatible runtime;
- real canonical weights, Hecate 9.6 weights, and a GPU.

The SDK is ready to run once those external artifacts are present. It deliberately
fails closed when a checkpoint, renderer, scale, orientation, or dependency is
missing.

## License

MIT for this package. Vendored architecture code and checkpoint provenance are
listed in `THIRD_PARTY_NOTICES.md`; the external `vc_render_tifxyz` binary and
model weights keep their own licenses.
