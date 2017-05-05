FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

ENV CAFFE_VERSION 0.15
LABEL com.nvidia.caffe.version="0.15"
ENV CAFFE_PKG_VERSION 0.15.13-1+cuda7.5
RUN apt-get update && apt-get install -y \
      caffe-nv=$CAFFE_PKG_VERSION \
      caffe-nv-tools=$CAFFE_PKG_VERSION \
      python-caffe-nv=$CAFFE_PKG_VERSION \
      libcaffe-nv0=$CAFFE_PKG_VERSION \
      libcaffe-nv0-dev=$CAFFE_PKG_VERSION \
    && rm -rf /var/lib/apt/lists/*

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
      protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

WORKDIR /root/openpose
RUN bash ./install_openpose.sh
