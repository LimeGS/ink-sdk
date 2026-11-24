# ink-sdk with everything its mesh-to-ink-map path needs, Linux x86-64. TORCH_BACKEND selects the
# PyTorch build of both environments: cu128 (CUDA 12.8, the default published image) or cpu.
#   - ink-sdk and its optional extras (Python 3.12, zarr 3);
#   - VILLA at the revision every 9 um profile pins, in its own environment (it pins zarr 2),
#     found through VILLA and INK_SDK_PYTHON;
#   - the official vc_render_tifxyz renderer from the VC3D AppImage, through INK_SDK_RENDERER.
# Model weights are not included: `ink-sdk fetch` stores them in /models (mount a volume there).
# Any --user works: PyTorch's compile cache is set explicitly, so a UID with no passwd entry
# never needs a username lookup.
#
#   docker build -t ink-sdk:cpu .                                      # CPU variant
#   docker build -t ink-sdk:cuda --build-arg TORCH_BACKEND=cu128 .     # GPU (run with --gpus all)
#   docker run --rm -v "$PWD:/work" -v ink-sdk-models:/models ink-sdk:cpu doctor

ARG PYTHON_IMAGE=python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
FROM ghcr.io/astral-sh/uv:0.9.17@sha256:5cb6b54d2bc3fe2eb9a8483db958a0b9eebf9edff68adedb369df8e7b98711a2 AS uv
FROM ${PYTHON_IMAGE}

ARG TORCH_BACKEND=cpu
ARG VILLA_REVISION=3ea17f54a9b3d5fd1aaf73e1d2c8386dbaa9f30e
ARG VC3D_URL=https://github.com/ScrollPrize/villa/releases/download/vc3d-archive-stable-363877681/VC3D-fc25b4d-2026-07-31-linux-x86_64.AppImage
ARG VC3D_SHA256=5da27c3df44c0958390ea752b550756f9d7e2c104840a95c0ba8b8651c7aee9f
LABEL org.opencontainers.image.source="https://github.com/LimeGS/ink-sdk" \
      org.opencontainers.image.description="ink-sdk with VILLA and the vc_render_tifxyz renderer (CPU)" \
      org.opencontainers.image.licenses="MIT"

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never

# git for the VILLA checkout (ink-sdk records its revision); curl for the renderer;
# the shared libraries the VC3D tools load outside the AppImage.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git libfontconfig1 libfreetype6 libglib2.0-0 libglx0 libgomp1 libopengl0 && \
    rm -rf /var/lib/apt/lists/*

# The renderer: the VC3D AppImage, checked and unpacked (no FUSE needed).
RUN curl -fsSL -o /tmp/vc3d.AppImage "$VC3D_URL" && \
    echo "$VC3D_SHA256  /tmp/vc3d.AppImage" | sha256sum -c - && \
    chmod +x /tmp/vc3d.AppImage && cd /opt && /tmp/vc3d.AppImage --appimage-extract > /dev/null && \
    mv /opt/squashfs-root /opt/vc3d && rm /tmp/vc3d.AppImage && \
    /opt/vc3d/AppRun vc_render_tifxyz --help > /dev/null

# VILLA's 9 um inference code, at the pinned revision, in its own environment.
RUN git clone --quiet --filter=blob:none --no-checkout https://github.com/ScrollPrize/villa.git /opt/villa && \
    git -C /opt/villa sparse-checkout set vesuvius ink-detection && \
    git -C /opt/villa checkout --quiet "$VILLA_REVISION" && \
    uv venv /opt/villa/ink-detection/.venv --python /usr/local/bin/python3.12 && \
    uv pip install --python /opt/villa/ink-detection/.venv/bin/python --torch-backend "$TORCH_BACKEND" \
      -e /opt/villa/vesuvius -e /opt/villa/ink-detection && \
    /opt/villa/ink-detection/.venv/bin/python -c "import koine_machines.inference.infer, vesuvius"

# ink-sdk itself.
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md /tmp/ink-sdk/
COPY src /tmp/ink-sdk/src
RUN uv venv /opt/ink-sdk --python /usr/local/bin/python3.12 && \
    uv pip install --python /opt/ink-sdk/bin/python --torch-backend "$TORCH_BACKEND" "/tmp/ink-sdk[all]" && \
    rm -rf /tmp/ink-sdk && \
    mkdir -p /models /cache /work && chmod 1777 /models /cache /work

ENV PATH=/opt/ink-sdk/bin:$PATH \
    VILLA=/opt/villa \
    INK_SDK_PYTHON=/opt/villa/ink-detection/.venv/bin/python \
    INK_SDK_RENDERER=/opt/vc3d/AppRun \
    INK_SDK_MODELS=/models \
    INK_SDK_CACHE=/cache \
    TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HOME=/tmp
WORKDIR /work
ENTRYPOINT ["ink-sdk"]
CMD ["--help"]
