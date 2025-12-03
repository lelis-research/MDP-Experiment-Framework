# ---------------------------------------------------------
# CUDA + Ubuntu base image (works on Compute Canada via Apptainer)
# ---------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------
# System dependencies for RL + Python tooling
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # basic build + tools
    build-essential \
    cmake \
    wget curl git \
    ffmpeg \
    # OpenGL / rendering stuff
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libosmesa6 \
    libosmesa6-dev \
    libglfw3 libglfw3-dev \
    libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 libxxf86vm1 \
    libsdl2-dev \
    # Python 3.10 + venv + pip
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    # SSL & compression libs (already used by system Python)
    ca-certificates \
    zlib1g-dev \
    libbz2-dev \
    libncurses5-dev \
    libncursesw5-dev \
    liblzma-dev \
    libffi-dev \
    libreadline-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Make sure "python" points to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# ---------------------------------------------------------
# Python virtual environment
# ---------------------------------------------------------
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade pip/setuptools/wheel in the venv
RUN pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------
# Install your Python dependencies from requirements.txt
# ---------------------------------------------------------
WORKDIR /workspace
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------
# Default: drop into a shell (you'll override with 'python train.py' etc.)
# ---------------------------------------------------------
CMD ["/bin/bash"]