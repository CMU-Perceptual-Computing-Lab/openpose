FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# install packages
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

WORKDIR /root
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

WORKDIR /root/openpose
RUN chmod u+x install_caffe_and_openpose.sh && \
    chmod u+x 3rdparty/caffe/install_caffe.sh && \
    chmod u+x models/getModels.sh && \
    chmod u+x install_openpose.sh && \
    bash install_caffe_and_openpose.sh
