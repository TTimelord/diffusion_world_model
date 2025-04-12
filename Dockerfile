FROM continuumio/miniconda3

WORKDIR /diffusion_world_model

# Install compilers, OpenGL/EGL dev libs, and SWIG
RUN apt-get update && apt-get install -y \
    gcc g++ gfortran \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    python3-dev \
    build-essential \
    git \
    vim \
    wget \
    ca-certificates \
    swig \
    && rm -rf /var/lib/apt/lists/*

COPY ./ ./

RUN conda env create -f conda_environment.yaml

SHELL ["conda", "run", "-n", "robodiff", "/bin/bash", "-c"]

RUN pip install -e ./escnn


# docker run -it --name diffusion_world_model -v $(pwd):/diffusion_world_model diffusion_wm_docker_img bash
