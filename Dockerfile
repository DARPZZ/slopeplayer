# syntax=docker/dockerfile:1
FROM python:3.13-slim

ARG TORCH_VERSION=2.14.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt requirements-rl.txt ./

RUN python -m pip install --upgrade pip \
    && python -m pip install "torch==${TORCH_VERSION}+cpu" \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements-rl.txt \
    && python -m playwright install --with-deps chromium \
    && apt-get update \
    && apt-get install -y --no-install-recommends xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY . .
COPY docker-entrypoint.sh /usr/local/bin/slope-entrypoint

RUN chmod +x /usr/local/bin/slope-entrypoint \
    && mkdir -p /app/models

ENTRYPOINT ["/usr/local/bin/slope-entrypoint"]
CMD ["train", "--url", "https://da.y8.com/games/slope", "--browser-channel", "bundled", "--instances", "1", "--steps", "100000", "--resume", "--device", "cpu"]
