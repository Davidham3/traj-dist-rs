FROM --platform=linux/amd64 mcr.microsoft.com/devcontainers/base:ubuntu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

USER vscode
WORKDIR /home/vscode

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH="/home/vscode/.cargo/bin:/home/vscode/.local/bin:${PATH}"

RUN bash -lc "rustup default stable"
