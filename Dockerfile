# Base image with Python 3.9 and Linux
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH \
    MUJOCO_GL=osmesa

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    git \
    wget \
    curl \
    cmake \
    ca-certificates \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libosmesa6-dev \
    libglfw3-dev \
    libxext-dev \
    patchelf && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# Create and activate robomimic conda environment with Python 3.9
RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN /opt/conda/bin/conda create -n robomimic_venv python=3.10 -y

# Install PyTorch and torchvision with CPU fallback
RUN /opt/conda/bin/conda run -n robomimic_venv pip install \
    torch==2.0.0 torchvision==0.15.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install robomimic from source
RUN /opt/conda/bin/conda run -n robomimic_venv pip install "cmake<4" setuptools wheel && \
    /opt/conda/bin/conda run -n robomimic_venv pip install --no-build-isolation egl_probe

    # torch 2.0 needs NumPy 1.x; newer opencv needs NumPy 2
RUN /opt/conda/bin/conda run -n robomimic_venv pip install "numpy<2" "opencv-python<4.12"

# Install robomimic from source
WORKDIR /opt
RUN git clone https://github.com/ARISE-Initiative/robomimic.git && \
    /opt/conda/bin/conda run -n robomimic_venv pip install -e ./robomimic

USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Install robosuite
RUN git clone https://github.com/ARISE-Initiative/robosuite.git && \
    cd robosuite && \
    /opt/conda/bin/conda run -n robomimic_venv pip install -r requirements.txt

# Optional: Install robomimic documentation dependencies
WORKDIR /opt/robomimic
RUN /opt/conda/bin/conda run -n robomimic_venv pip install -r requirements-docs.txt

# Set the working directory
WORKDIR /workspace

# Activate Conda environment and start bash when container starts
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && bash"]
