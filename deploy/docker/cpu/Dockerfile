FROM ubuntu:16.04
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

RUN git clone https://github.com/chriamue/openpose
RUN cd openpose && git checkout cmake && mkdir -p 3rdparty/caffe/build && cd 3rdparty/caffe/build && cmake .. && make
RUN cd openpose && git checkout cmake && mkdir -p build && cd build && cmake .. && make