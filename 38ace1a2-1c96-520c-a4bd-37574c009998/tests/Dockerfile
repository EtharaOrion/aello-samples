# Verifier environment. Harbor builds the SEPARATE verifier image from this tests/
# directory (its build context) and does not upload tests/ at runtime, so /tests must be
# baked in here — see harness docs "Verifier environment (Shared vs Separate)".
# Same digest-pinned base as the agent image: it is public, already proven pullable, and
# ships the numpy the grader needs. Grading itself is CPU-only (integer comparisons).
FROM pytorch/pytorch@sha256:68c022c2f4627943a6f3e574cfd2c8ae4256210d5f66ae2b117942e0a8d4fa9d

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 TZ=UTC

# The scoring engine, vendored. Installing it from the shipped wheel means the grading
# contract is implemented by the package the scaffold expects rather than reimplemented
# against it. pytest runs the deterministic layer (aello_eval.checks plugin) that writes
# check_results.json. The import smoke test fails the BUILD rather than letting a broken
# engine reach a trial.
COPY aello_eval-0.1.0-py3-none-any.whl /opt/aello/aello_eval-0.1.0-py3-none-any.whl
RUN pip install --no-cache-dir /opt/aello/aello_eval-0.1.0-py3-none-any.whl pytest && \
    python -c 'import aello_eval; print("aello_eval OK:", aello_eval.__file__)' && \
    pytest --version

COPY . /tests
RUN chmod +x /tests/test.sh && mkdir -p /logs/verifier /submission
