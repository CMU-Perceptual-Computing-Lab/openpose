FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
MAINTAINER chriamue@gmail.com

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
      libsnappy-dev \
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
RUN pip install --upgrade pip

RUN git clone https://github.com/chriamue/openpose
RUN cd openpose && git checkout cmake && git pull origin cmake && mkdir -p build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j"$(nproc)"

RUN cd openpose/models && ./getModels.sh

WORKDIR /openpose

CMD ./build/openpose
