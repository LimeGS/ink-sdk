# TimeSformer adapter and profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans
> inline, TDD, then one fresh whole-change review before opening the requested PR.

**Goal:** Make GP Scroll1, Large Scroll1 and Scroll5 available through ink-sdk.
**Architecture:** Add one BaseAdapter implementation and three packaged TOMLs,
using the existing registry, opt-in fetch, preparation, map and receipt paths.
Use the installed, pinned TimeSformer-pytorch architecture and safe tensor loading;
do not depend on the separate PHerc1218 runner or modify its frozen runtime.
**Tech Stack:** Existing Python/Pydantic/NumPy/PyTorch; optional
timesformer-pytorch==0.4.1 and safetensors, torch>=2.6 for safe archive inspection.
**Spec:** User approved the adapter/three-profile scope and requested a new
branch and PR. Checkpoints and preprocessing are those already safely smoke-tested
in the PHerc1218 run; physical transfer remains explicitly experimental.

## Global Constraints
- New branch feat/timesformer-profiles from origin/main1dbfac2. Preserve the
  unrelated private/base-repaired commit/checkout; no merge or release requested.
- Profiles timesformer@gp-scroll1, timesformer@large-scroll1, timesformer@scroll5.
- Exact HF revisions/SHA/bytes already verified:
  GP61b81d31ddb93c89a2b3b00c66769d18bd96ccb2,151853128bytes,
  SHA490a98f9491e1180274ed3a0c0a9c611d73a0109c0e0c0fbba1097562a972488;
  Large7e8b37ce66b2497b4af1ad07ae3d7d8f5c5fa990,336694640bytes,
  SHAc85dd722aad3734476dd6d8f14fae7a249a0a3c019fbd68f894558ae381e3e9d;
  Scroll5396d39b9a66cbd21c9cbe453c4540d9516707020,455711244bytes,
  SHA3c6b0b4d1b766a69cef2ba86b51574d91c71dded7fa4d9ec7435a1a95c61320f.
- Input uint8 ZYX,26central planes,7.91±0.1um on every axis,64px tile,
  stride16,batch16,clip[0,200]/200,sigmoid BEFORE bilinear4→64,Hann64.
- Default reverse/primaryreverse matches the current reference run; support
  existing direction forward/reverse/both overrides, without guessing ink face.
- FP32 only, no TTA or 3D output; reject unsupported settings instead of ignoring.
- Safetensors or weights_only=True only; standard archived training types become
  inert placeholders, unknown pickle globals and any state mismatch fail closed.
- No automatic downloads during inference; use existing fetch and checkpoint
  integrity checks. No implicit resampling/new renderer/scientific accuracy claims.
- Preserve existing SDK behavior; keep accumulation on CPU, not full GPU rasters.

## Review Focus
- Wrong checkpoint/config/learned keys must fail, never leave random parameters.
- Thick/short stacks, nonfinite logits and edge/partial batches must be explicit.
- Reverse changes only Z; XY, Hann support and fixed probabilities stay unchanged.
- Package installs/core imports must not require optional torch/TimeSformer.
- CLI/fetch/receipts must expose three usable profiles, not merely TOML filenames.

### Task 1: Safe TimeSformer adapter and packaged profiles
**Files:** src/ink_sdk/adapters/timesformer.py;src/ink_sdk/profiles/*.toml;
src/ink_sdk/profiles.py;src/ink_sdk/registry.py;pyproject.toml;
tests/test_timesformer.py;tests/test_profiles.py;tests/test_cli.py;tests/test_fetch.py.
**Interfaces:** TimeSformerAdapter(BaseAdapter), model_factory(checkpoint,device,profile);
_build_model(profile), _checkpoint_state(path), existing RawOutput/RunContext.
- [x] RED: models/get_adapter/fetch expose exactly three new pinned profiles.
- [x] RED: spatial/input-dependent reference tiles verify central26,Z reversal,
  clipping/divisor,sigmoid/interpolation,Hann,edge origins and batches1/3/16.
- [x] RED: unsafe globals/mismatched states/hashes,wrongdtype/depth/config,
  nonfinite/wrong-shape logits and unsupported execution options fail closed.
- [x] GREEN: minimum adapter/registry/optional-extra/profile changes; no shared
  sliding-window abstraction or source-registry mutation.
- [x] Run PYTHONPATH=src python -m pytest -q. Expected: all tests pass.
- [x] Commit only this feature's code/tests/profiles and this plan.

### Task 2: Verify real weights, document and open PR
**Files:** README.md;THIRD_PARTY_NOTICES.md;tests/test_timesformer.py;this plan;
src/ink_sdk/battery.py;src/ink_sdk/runner.py;tests/test_battery.py (review fixes).
**Interfaces:** consume Task1 profiles/adapter through existing API/CLI.
- [x] Existing CLI/dummy pipeline/scale/orientation/fetch suite remains green;
  full SDK infer with a small model verifies files, scale, mask and receipt.
- [x] Real3checkpoint architecture/load/forward smoke in a separate fresh remote
  code root on CPU or an idle explicitly approved GPU, no downloads to inference
  caches and no replacement of the PHerc1218 runtime/artifacts.
- [x] Document install extra, three IDs, fetch/infer example, exact physical
  contract/unsupported scales and experimental limitations; acknowledge sources.
- [x] One fresh read-only review; one Important/Critical TDD fix pass if needed;
  green full suite/build/packaged profiles verification.
- [ ] Commit docs/fixes, push only feat/timesformer-profiles, open PR againstmain;
  do not merge. Include tests, limitations and real-weight smoke evidence.

## Pre-flight / rulings
- User requested branch/PR, superseding the earlier no-commit choice for this
  SDK task only. Existing recto work stays uncommitted and protected.
- Native worktree is isolated; original SDK checkout remains private/base-repaired.
- Base origin/main avoids including that unrelated unpublished ResNet change.
- Baseline PYTHONPATH=src python -m pytest -q:70passed8.08s. No installs needed.
- Hindsight list/search/capture refused localhost8888; keep a local ledger.
- Shared interface Task1 RawOutput/profile IDs feeds Task2 API/docs/smoke; no conflict.
- User approved inline execution without more question rounds; proceed to PR.

## Verification evidence
- Task1 RED31failed6passed, then full99passed; existing70tests stay green.
- GP/Large/Scroll5 real weights loaded strictly on authorized GPU1 with
  timesformer-pytorch0.4.1/torch2.6.0+cu124. A neutral26×87×101 crop from the
  verified nativew064 render matched the separate reference runner exactly:
  maximum probability/u16 error0 for all3models; support identical.
- Independent smoke code/models links/core-dependency site live under
  /ssd/draft/ink-sdk-timesformer-smoke-20261030, not the PHerc1218 runtime.
  ReceiptSHA c7f8443cb04db1171132de71bf1654600ac97bce9390d32df0c5120a456dbd61.
  Original runtime-sourceSHA remained unchanged. No training/TEST/accuracy claim.
- Fresh whole-change review found two Important battery integration bugs. RED→GREEN
  regressions now skip TimeSformer depth steps other than 1 and publish non-forward
  primary copies, restoring reverse-only mosaics/stability images. Full suite: 101 passed.
- Rebuilt wheel/sdist contain the reviewed sources and all 11 profiles; running
  the full suite against the extracted final wheel also passed all 101 tests.
- Adapter/registry/tests are Ruff-clean. Across all touched Python files, baseline
  comparison found 0 new findings; 13 pre-existing core warnings remain untouched.
- The real-weight smoke predates the battery/runner-only fixes. Adapter, profile
  and other source hashes still match that receipt; both fixes have source/wheel regressions.
