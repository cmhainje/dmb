FROM ubuntu:22.04

# no interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# install packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gfortran \
    pkg-config \
    wget \
    curl \
    git \
    vim \
    gdb \
    valgrind \
    ca-certificates \
    libopenmpi-dev \
    openmpi-bin \
    libgsl-dev \
    libfftw3-dev \
    libfftw3-mpi-dev \
    libhdf5-dev \
    libhdf5-openmpi-dev \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libtool \
    libtool-bin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

WORKDIR /workspace

# Keep container running
CMD ["/bin/bash"]

