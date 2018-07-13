FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

RUN apt-get update && apt-get install -y \
    libatlas-base-dev \
    libopencv-dev \
    lsb-release \
    python-numpy \
    python3-numpy \
    sudo \
    wget \
    python-pip \
    cmake

WORKDIR /usr/local

COPY . .

RUN ./install.sh
