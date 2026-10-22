# Agent environment AND separate-verifier environment. 
#
# ONE Dockerfile now serves both containers. task.toml pins [environment].docker_image, and
# Harbor keys its tests-upload decision off exactly that field
# (harbor/trial/trial.py:626 -- `skip_tests_upload = env_config.docker_image is None`).
# With the pin set, Harbor uploads tests/ into the SEPARATE verifier container at grade
# time instead of expecting the image to already carry /tests, which is what tests/Dockerfile
# used to provide. [verifier].environment_mode stays "separate", so the graded labels in
# tests/heldout/ still land ONLY in the verifier container and the agent never sees them.
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

# The scoring engine. NO .whl SHIPS IN THIS BUNDLE. The literal token ".aello_eval_wheel"
# below is what arms harness/scaffold/deliver.py:_stage_aello_eval_wheel (it greps this
# Dockerfile for that string): at build time the scaffold builds the wheel from
# harness/packages/aello-eval into ./.aello_eval_wheel/, docker COPYs it, and the scaffold's
# cleanup callback deletes the staged directory afterwards. The delivered bundle therefore
# contains no wheel, and no wheel is tracked in git (.gitignore carries the directory).
#
# The staging hook only fires from _build_tagged_image, which _ensure_env_image reaches
# because [environment].docker_image is pinned -- the same pin that lets Harbor upload
# tests/ into the separate verifier. The two decisions are load-bearing together.
#
# It is installed HERE because the separate verifier runs in this same image and needs it.
# What that costs: the agent can read the aggregation code. What it does NOT cost: the
# graded labels, which never enter this image (see the [verifier] note in task.toml). The
# formula the engine implements is already stated openly in task.toml and instruction.md.
# The import smoke test fails the BUILD rather than letting a broken engine reach a trial.
COPY .aello_eval_wheel/ /opt/eval_wheel/
RUN pip install --no-cache-dir /opt/eval_wheel/*.whl pytest && \
    rm -rf /opt/eval_wheel && \
    python -c 'import aello_eval; print("aello_eval OK:", aello_eval.__file__)' && \
    pytest --version

# HARDENING. One image serves BOTH the agent and the separate verifier, so anything left in
# site-packages is readable by the agent. The council runs host-side and its config names the
# judge panel, the quorum and the per-rubric scoring policy -- an agent that reads judges.yaml
# can tailor its trajectory to the judges that will grade it. Strip the council surface; keep
# only the deterministic path the verifier actually calls (aggregate/checks/score/schema/
# outcome). aello_eval/__init__.py imports the council lazily (PEP 562) precisely so this
# deletion leaves `import aello_eval` working. The smoke test below is the guard: it asserts
# the deterministic imports still resolve AND that judges.yaml is gone.
RUN SITE="$(python -c 'import site; print(site.getsitepackages()[0])')" && \
    rm -f "$SITE"/aello_eval/judges.yaml* \
          "$SITE"/aello_eval/council.py \
          "$SITE"/aello_eval/creds.py \
          "$SITE"/aello_eval/context.py \
          "$SITE"/aello_eval/preflight.py && \
    rm -rf "$SITE"/aello_eval/__pycache__ && \
    python -c 'import aello_eval, aello_eval.checks; from aello_eval import RubricResult, aggregate, load_check_results, write_reward; print("deterministic engine OK")' && \
    python -c 'import pathlib, aello_eval, sys; d = pathlib.Path(aello_eval.__file__).parent; leaked = sorted(p.name for p in d.iterdir() if p.name.startswith("judges") or p.name in {"council.py", "creds.py"}); sys.exit("COUNCIL SURFACE LEAKED: %s" % leaked) if leaked else print("council surface absent OK")'

WORKDIR /workspace
# `COPY .` verbatim, as every recorded trajectory in trajectories/ was produced against:
# narrowing what the agent sees would silently invalidate those 40 runs as a baseline. The
# transient .aello_eval_wheel/ staging dir rides along in the build context, so it is deleted
# immediately after -- a bare COPY would otherwise hand the agent the engine wheel, and
# tests/isolation.py walks this same tree.
COPY . /workspace/environment
RUN rm -rf /workspace/environment/.aello_eval_wheel

# /logs/verifier and /submission are ALSO created by Harbor at grade time -- it calls
# empty_dirs([verifier_dir], chmod=True) before the verifier (trial.py:599) and creates
# artifact parent dirs on upload (artifact_handler.py:248-250). Kept here anyway so the
# image is runnable standalone. The old tests/Dockerfile's `chmod +x /tests/test.sh` has
# NO equivalent here and needs none: /tests does not exist at build time, and Harbor
# chmods the uploaded script as root before running it (verifier.py:190-195).
RUN mkdir -p /logs /logs/verifier /logs/agent /submission && chmod 777 /submission
ENV PYTHONUNBUFFERED=1 TZ=UTC
