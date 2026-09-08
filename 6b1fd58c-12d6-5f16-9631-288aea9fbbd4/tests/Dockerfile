# Separate verifier environment. Harbor builds this from tests/ as its build context and does
# not upload tests/ at runtime, so /tests must be baked in. Same digest-pinned base as the agent
# image. Grading is CPU-only (integer confusion-matrix counts over the private labels).
FROM pytorch/pytorch@sha256:68c022c2f4627943a6f3e574cfd2c8ae4256210d5f66ae2b117942e0a8d4fa9d
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 TZ=UTC
COPY aello_eval-0.1.0-py3-none-any.whl /opt/aello/aello_eval-0.1.0-py3-none-any.whl
RUN pip install --no-cache-dir /opt/aello/aello_eval-0.1.0-py3-none-any.whl pytest && \
    python -c 'import aello_eval; print("aello_eval OK")' && pytest --version
COPY . /tests
