# Agent environment. Base pinned by digest; a tag can be re-pointed and would make this bundle
# unreproducible after freeze. The Qwen2.5-1.5B-Instruct backbone (pinned revision
# 989aa7980e4cf806f80c7fef2b1adb7bc71aa306) is VENDORED under environment/qwen2.5-1.5b-instruct
# and lands in the image through the COPY below, so the build reaches no model-zoo host at all
# (G-EGRESS-CLOSURE / G-DEL-PIN repair 2026-08-19; client C2, PRD 5.1).
FROM pytorch/pytorch@sha256:68c022c2f4627943a6f3e574cfd2c8ae4256210d5f66ae2b117942e0a8d4fa9d
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 TZ=UTC
RUN apt-get update && apt-get install -y --no-install-recommends \
        coreutils bash tar gzip curl ca-certificates git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "transformers==5.5.3" "tokenizers>=0.20" "numpy>=1.26"
WORKDIR /workspace
# Brings in the vendored backbone at environment/qwen2.5-1.5b-instruct; no build-time fetch.
COPY . /workspace/environment
RUN mkdir -p /logs /logs/verifier /logs/agent /submission && chmod 777 /submission
