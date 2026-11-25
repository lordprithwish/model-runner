FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy HF_HOME=/cache/huggingface

# GPU=0 builds the API-only image (no torch), handy for testing without a GPU.
ARG GPU=1
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project $([ "$GPU" = 1 ] && echo --extra gpu)

COPY app app
COPY config config

EXPOSE 8000
# Single worker only: the job queue lives in memory (see README).
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
