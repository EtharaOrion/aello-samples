# Agent environment. Pinned by digest, not by tag: a tag can be re-pointed and would make
# this bundle unreproducible after freeze. Harbor builds this image from the bundle alone;
# no private registry and no authoring-host state is needed (client C2).
FROM pytorch/pytorch@sha256:68c022c2f4627943a6f3e574cfd2c8ae4256210d5f66ae2b117942e0a8d4fa9d

ENV DEBIAN_FRONTEND=noninteractive

# Agent bootstrap dependencies pre-installed, so the coding-agent setup does not need
# runtime apt-get. npm is NOT upgraded: npm@latest is 12.x and requires node >= 22, so
# upgrading on a node 20 base fails EBADENGINE and takes the build down.
RUN apt-get update && apt-get install -y --no-install-recommends \
        coreutils bash tar gzip curl ca-certificates git \
        procps xz-utils gnupg2 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# The scoring engine is NOT installed here. Grading runs in the separate verifier
# container built from tests/ (tests/Dockerfile), so the agent-visible surface carries
# neither the aello_eval implementation nor the judge configuration it ships.

WORKDIR /workspace
COPY . /workspace/environment
RUN mkdir -p /logs /logs/verifier /logs/agent /submission && chmod 777 /submission
ENV PYTHONUNBUFFERED=1 TZ=UTC
