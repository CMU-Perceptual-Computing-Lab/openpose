# TODO: build from BVLC's gpu caffe image instead
# FROM bvlc/caffe:gpu
FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# TODO: avoid moving around caffe source build and relinking
# WORKDIR $CAFFE_ROOT/build
# RUN make install

# TODO: omit duplicate dependencies inherited from bvlc/caffe:gpu
# as well as weaning off non essential packages such as sudo, lsb-release, etc...
RUN apt-get update && apt-get install -y \
      build-essential \
      cmake \
      git \
      libatlas-base-dev \
      libatlas-dev \
      libboost-all-dev \
      libgflags-dev \
      libgoogle-glog-dev \
      libhdf5-dev \
      libleveldb-dev \
      liblmdb-dev \
      libopencv-dev \
      libprotobuf-dev \
      lsb-release \
      protobuf-compiler \
      python-dev \
      python-numpy \
      python-pip \
      python-setuptools \
      python-scipy \
      sudo \
      wget \
    && rm -rf /var/lib/apt/lists/*

ENV OPENPOSE_ROOT=/opt/caffe
WORKDIR $OPENPOSE_ROOT

# TODO: Add ARG for choosing a release
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git .

# TODO: Avoide using nested bash scrips and call cmake directly for transparency
# also, avoid baking models binaries into docker images
RUN chmod +x install_caffe_and_openpose.sh && \
    chmod +x 3rdparty/caffe/install_caffe.sh && \
    chmod +x models/getModels.sh && \
    chmod +x install_openpose.sh && \
    bash ./install_caffe_and_openpose.sh
