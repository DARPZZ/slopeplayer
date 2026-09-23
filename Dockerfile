# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ARG TORCH_VERSION=2.8.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    SLOPE_AUTO_RESUME=1

WORKDIR /app

COPY requirements.txt ./

# Installing the CPU wheel first keeps the default image substantially smaller
# than a Linux PyPI install that may pull CUDA runtime packages.
RUN python -m pip install --upgrade pip \
    && python -m pip install "torch==${TORCH_VERSION}" \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt \
    && python -m playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml slope.py ./
COPY slope_core ./slope_core
COPY docker-entrypoint.sh /usr/local/bin/slope-entrypoint

RUN python -m pip install --no-deps . \
    && chmod +x /usr/local/bin/slope-entrypoint \
    && mkdir -p /app/runs /app/artifacts

VOLUME ["/app/runs", "/app/artifacts"]

ENTRYPOINT ["/usr/local/bin/slope-entrypoint"]
CMD ["train", "--model", "/app/runs/slope_qrdqn", "--steps", "500000", "--device", "cpu", "--headless", "--report-every", "100"]
