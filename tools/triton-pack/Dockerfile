# syntax=docker/dockerfile:1.7
# The two stages are independently pinned to linux/amd64 OCI manifests. The
# release/3.8.x source archive is supplied by the build context only after its
# toolchain.lock SHA-256 has been checked by lock_verify.py.
FROM python:3.12.3-slim-bookworm@sha256:fd3817f3a855f6c2ada16ac9468e5ee93e361005bd226fd5a5ee1a504e038c84 AS python-runtime
RUN mkdir -p /python-runtime-libs && \
    (ldd /usr/local/bin/python3.12; \
     find /usr/local/lib/python3.12/lib-dynload -type f -name '*.so' \
       -exec ldd {} \; 2>/dev/null) | \
    sed -n 's/.*=> \(\/[^ ]*\).*/\1/p' | \
    grep -Ev '/(libc\.so|libm\.so|libpthread|libdl\.so|ld-linux)' | \
    sort -u | xargs -r cp -L -t /python-runtime-libs
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04@sha256:e542739fcaa4f45da5add8c4cf5769783a61628b2518304a5bbe4ace468b8c8f AS triton-build

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
      git ca-certificates zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*
COPY --from=python-runtime /usr/local /usr/local
COPY --from=python-runtime /python-runtime-libs/ /opt/python-runtime-libs/
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    SOURCE_DATE_EPOCH=1787606393 \
    CMAKE_BUILD_PARALLEL_LEVEL=2 \
    MAX_JOBS=2 \
    CPATH=/usr/local/cuda/targets/x86_64-linux/include \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/opt/python-runtime-libs:/usr/local/cuda/lib64 \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump \
    TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm \
    TRITON_CUDACRT_PATH=/usr/local/cuda/include \
    TRITON_CUDART_PATH=/usr/local/cuda \
    TRITON_CODEGEN_BACKENDS=nvidia

ADD --checksum=sha256:1e25ae6d0dd4a1172002c53bb516b5a3c0ceb33c72900e7ad70394c58e1bd6a5 \
    https://github.com/triton-lang/triton/archive/b252c7c4cea49216b27e5a47db061f9d0dbf49f3.tar.gz \
    /opt/triton-source.tar.gz
COPY requirements-build.lock /opt/requirements-build.lock

# These packages are image-build inputs only. The resulting image identity and
# recipe digest are recorded by toolchain.lock; Program Pack builds never pip
# install or access the network.
RUN python3 -m pip install --no-cache-dir --require-hashes \
      -r /opt/requirements-build.lock && \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps \
      /opt/triton-source.tar.gz && \
    python3 -c "import sys,triton; assert triton.__version__ == '3.8.0'; assert 'torch' not in sys.modules"

FROM nvidia/cuda:12.9.1-devel-ubuntu24.04@sha256:e542739fcaa4f45da5add8c4cf5769783a61628b2518304a5bbe4ace468b8c8f
RUN apt-get update && \
    apt-get install --yes --no-install-recommends openssl ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY --from=python-runtime /usr/local /usr/local
COPY --from=python-runtime /python-runtime-libs/ /opt/python-runtime-libs/
COPY --from=triton-build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    SOURCE_DATE_EPOCH=1787606393 \
    LD_LIBRARY_PATH=/opt/python-runtime-libs:/usr/local/cuda/lib64 \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump \
    TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm \
    TRITON_CUDACRT_PATH=/usr/local/cuda/include \
    TRITON_CUDART_PATH=/usr/local/cuda \
    IMPARO_PTXAS_VERSION=12.9.86 \
    IMPARO_CUOBJDUMP_VERSION=12.9.82 \
    IMPARO_CUOBJDUMP=/usr/local/cuda/bin/cuobjdump

RUN python3 -c "import platform,sys,triton; assert platform.python_version() == '3.12.3'; assert triton.__version__ == '3.8.0'; assert 'torch' not in sys.modules" && \
    openssl version && \
    /usr/local/cuda/bin/ptxas --version 2>&1 | grep -F 'V12.9.86' && \
    /usr/local/cuda/bin/cuobjdump --version 2>&1 | grep -F 'V12.9.82'
WORKDIR /workspace
