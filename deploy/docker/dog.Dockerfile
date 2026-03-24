# deploy/docker/dog.Dockerfile
ARG BASE=nvidia/cuda:12.2.2-cudnn9-runtime-ubuntu22.04
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip git curl ca-certificates \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/legs
COPY pyproject.toml /workspace/legs/
RUN python3 -m pip install -U pip && \
    python3 -m pip install msgpack grpcio grpcio-tools protobuf prometheus-client

COPY packages /workspace/legs/packages
COPY api /workspace/legs/api

ENV PYTHONPATH="/workspace/legs/packages/legs_common/src:/workspace/legs/packages/legs_dog/src:/workspace/legs/packages/legs_server/src"

CMD ["bash", "-lc", "python3 -m legs_dog.main --help || bash"]
