# Third-party notices

This package is MIT-licensed. The files below are vendored or derived from
MIT-licensed upstream projects. Their notices are reproduced here so the
provenance of the inference code is explicit.

## ScrollPrize/villa

- Source: <https://github.com/ScrollPrize/villa>
- Pinned revision: `3ea17f54a9b3d5fd1aaf73e1d2c8386dbaa9f30e`
- License: MIT (upstream repository)
- Vendored files:
  - `src/ink_sdk/_vendor/canonical/model_resnet3d_3d_decoder.py`
  - `src/ink_sdk/_vendor/canonical/models/resnetall.py`
- Used by: `canonical2um` and the shared Hecate feature extractor.
- The files are copied for standalone inference. The sliding-window runner,
  scale planning, receipts and liveness gate are local `ink-sdk` code, not
  upstream code.

## scrollprize/hecate

- Source: <https://github.com/scrollprize/hecate>
- Pinned revision: `9cb86e500e944b11a06a7020403cde5dffb5bcb2`
- License: MIT (upstream repository)
- Vendored file: `src/ink_sdk/_vendor/hecate.py`
- Used by: `hecate@2.4um` and `hecate@9.6um`.

## Model checkpoints

The five `profiles/*.toml` files pin checkpoint repository revisions, paths,
byte sizes and SHA-256 digests. The checkpoint repositories' model cards and
their own licenses govern the weights. `ink-sdk fetch` only downloads a
checkpoint when explicitly invoked; inference never downloads weights.

| profile | checkpoint repository | revision |
|---|---|---|
| `ink_9um@scrollprize-seed43-step75000` | `scrollprize/ink_9um` | `7109667e2607db1b90c37c8b09cb876ea7fe7bb1` |
| `ink_9um@reader-v2-step040000` | `domenicor046/reader-v2` | `72633d8aa6d40e9e737ccea4c9928fbc624dab45` |
| `canonical_2um@r152-3ddec-v2-l5-epoch13` | `scrollprize/ink_canonical_2um` | `075855bc69317ef6febf39a0d9d687b27d2b7c29` |
| `hecate@2.4um` | `scrollprize/hecate` | `9cb86e500e944b11a06a7020403cde5dffb5bcb2` |
| `hecate@9.6um` | `scrollprize/hecate` | `9cb86e500e944b11a06a7020403cde5dffb5bcb2` |

## vc_render_tifxyz / volume-cartographer

`vc_render_tifxyz` is an external binary from the VC3D / volume-cartographer
project. It is not vendored here and is not redistributed by this package.
Obtain it from its upstream release, comply with its license, and point
`ink-sdk` at it with `--renderer`, `INK_SDK_RENDERER`, or `VC3D_APPRUN`.

## Python dependencies

The runtime dependencies (`numpy`, `pydantic`, `typer`, `zarr`, `tifffile`,
and optionally `scipy`, `torch`, `huggingface_hub`) keep their own licenses.
