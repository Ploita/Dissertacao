FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    python3-dev \
    xvfb \
    xserver-xorg-core \
    libglx-mesa0 \
    xfonts-base \
    freeglut3-dev \
    ffmpeg \
    git \
    xauth \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv pip install --system -r pyproject.toml gymnasium[box2d]

COPY . .

CMD ["sh", "-c", "Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset & sleep 2 && DISPLAY=:99 python -m src.run_experiments"]