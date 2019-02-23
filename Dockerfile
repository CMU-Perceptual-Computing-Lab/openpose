# CUDA 8.0 + cuDNN 5 + build dependencies
FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

LABEL maintainer="Timothy Liu <timothy_liu@mymail.sutd.edu.sg>"

USER root

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -yq --no-install-recommends --no-upgrade \
    apt-utils && \
    apt-get install -yq --no-install-recommends --no-upgrade \
    # install system packages
    software-properties-common \
    curl \
    python3-dev \
    python3-tk \
    locales \
    build-essential \
    cmake \
    # for OpenPose
    libatlas-base-dev \
    libprotobuf-dev \
    libleveldb-dev \
    libsnappy-dev \
    libhdf5-serial-dev \
    protobuf-compiler \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    liblmdb-dev \
    opencl-headers \
    ocl-icd-opencl-dev \
    libviennacl-dev \
    libopencv-dev \
    && ldconfig && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

ENV SHELL=/bin/bash \
    OP_USER=openpose \
    OP_UID=1000 \
    OP_GID=100 \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

WORKDIR /openpose

COPY . /openpose

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --force-reinstall && \
    rm get-pip.py && \
    pip install --no-cache-dir Cython && \
    pip install --no-cache-dir -r /openpose/docker_files/requirements.txt && \
    cd /openpose && mkdir build && cd build && \
    # replace CMakeLists.txt with one that specifies BUILD_PYTHON=ON
    cp /openpose/docker_files/CMakeLists.txt /openpose/CMakeLists.txt && \
    cmake .. && make -j && cd python && make install && \
    # in Python code:
    # sys.path.append('/openpose/build/python')
    rm -rf /home/$OP_USER/.cache

USER $OP_UID
