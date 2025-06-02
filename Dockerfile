# Build stage
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS builder
LABEL maintainer="prime intellect"
LABEL repository="prime-rl"

# Set en_US.UTF-8 locale by default
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

# Set CUDA_HOME and update PATH
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:/usr/local/cuda/bin

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends --force-yes \
  build-essential \
  curl \
  wget \
  git \
  vim \
  htop \
  nvtop \
  iperf \
  tmux \
  openssh-server \
  git-lfs \
  sudo \
  gpg \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# install gsutil
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y

# # Install Rust
# RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
# ENV PATH="/root/.cargo/bin:${PATH}"
# RUN echo "export PATH=\"/opt/conda/bin:/root/.cargo/bin:\$PATH\"" >> /root/.bashrc

# 1. Create the non-root account (UID 1000 / GID 1000)  ────────────────
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd --gid $GROUP_ID appuser  && \
    useradd  --uid $USER_ID --gid appuser --create-home --shell /bin/bash appuser


# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Install Python dependencies (The gradual copies help with caching)
WORKDIR /appuser/prime-rl

COPY ./pyproject.toml ./pyproject.toml
COPY ./uv.lock ./uv.lock
COPY ./README.md ./README.md
COPY ./src/ ./src/

# Create venv and install dependencies
RUN uv sync && uv sync --extra fa

# Runtime stage
FROM python:3.11-slim

# 2. Put your code under /appuser not /opt, then hand ownership to appuser ─
WORKDIR /appuser/prime-rl

# still need a compiler for wheels that ship only as source
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential wget clang && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user in runtime stage
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd --gid $GROUP_ID appuser && \
    useradd --uid $USER_ID --gid appuser --create-home --shell /bin/bash appuser

# Copy the virtual-env and fix permissions
COPY --from=builder /appuser/prime-rl/.venv /appuser/prime-rl/.venv
RUN ln -sf /usr/local/bin/python /appuser/prime-rl/.venv/bin/python && \
    chown -R appuser:appuser /appuser

# Copy sources and configs
COPY --from=builder /appuser/prime-rl/src ./src
COPY ./configs ./configs


# 4. Activate the venv + switch user before ENTRYPOINT ──────────────────
ENV PATH="/appuser/prime-rl/.venv/bin:${PATH}"

USER appuser         

ENTRYPOINT ["python", "src/zeroband/infer.py"]
