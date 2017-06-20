FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

RUN apt-get update
RUN apt-get install -y \
    libatlas-base-dev \
    libopencv-dev \
    lsb-release \
    python-numpy \
    python3-numpy \
    sudo
    wget

WORKDIR /usr/local

COPY . .

RUN ./install_caffe_and_openpose.sh
