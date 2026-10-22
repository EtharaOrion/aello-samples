FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel@sha256:a55ff10111eb11f998884327d37361592e632899edd24fce99886b69289e33e6

# System deps (curl/ca-certificates for the build-time dataset fetch).
# System deps: curl/ca-certificates for build-time dataset fetch;
# nodejs/npm/procps pre-installed so claude-code-opus5 agent bootstrap does NOT
# need runtime apt-get (which would require adding archive.ubuntu.com to network
# allowlist and reaching Ubuntu apt mirrors at each agent-install). Option 1 per
# owner directive m1046+m1047: pre-install agent bootstrap deps in the image to
# preserve the anti-cheat network policy (still blocks download.pytorch.org /
# huggingface.co).
RUN apt-get update && apt-get install -y --no-install-recommends \
        coreutils bash tar gzip curl ca-certificates \
        nodejs npm procps xz-utils gnupg2 && \
    rm -rf /var/lib/apt/lists/*

# CIFAR-10 fetched at build time and STREAMED straight into the data dir
# (fast.ai S3 mirror; ImageFolder PNG layout). No system temp dir is used; no
# pre-staged file is required, so the image builds on any machine with network.
# TRAIN SPLIT ONLY. This one image serves both phases (the separate verifier reuses
# it), but the TEST/eval split is deliberately NOT shipped and is removed here: it is
# the VERIFIER-ONLY held-out set (fetched by the verifier at verify time). The agent
# therefore never has the eval set and its network cannot reach the source, so it
# cannot train on it -- no_test_set_leakage enforced by construction. No system temp
# dir is used.
ARG CIFAR10_URL=https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz
RUN mkdir -p /workspace/data && \
    curl -fSL --retry 5 --retry-delay 3 --connect-timeout 20 "${CIFAR10_URL}" \
        | tar -xz -C /workspace/data && \
    rm -rf /workspace/data/cifar10/test && \
    test -d /workspace/data/cifar10/train && \
    [ ! -e /workspace/data/cifar10/test ] && \
    [ "$(find /workspace/data/cifar10/train -name '*.png' | wc -l)" -eq 50000 ]

# Build wheel from source and install; smoke-test fails the build if broken.
RUN pip install --no-cache-dir build && \
    python -m build --wheel --outdir /opt/aello /opt/aello/aello_eval_src && \
    pip install --no-cache-dir /opt/aello/aello_eval-*.whl && \
    python -c 'import aello_eval; print("aello_eval OK:", aello_eval.__file__)'

WORKDIR /workspace
RUN mkdir -p /workspace/periodic_checkpoints /workspace/verifier_output /logs /logs/verifier

# ENTRYPOINT stamps container_launch.ts ONLY IF ABSENT, so the verifier
# container's own start does not overwrite the agent's launch time (both share
# the bind-mounted /logs). Compose overrides CMD, not ENTRYPOINT.
ENTRYPOINT ["/bin/bash", "-c", "[ -f /logs/container_launch.ts ] || date +%s.%N > /logs/container_launch.ts; exec \"$@\"", "--"]
CMD ["/bin/bash"]
