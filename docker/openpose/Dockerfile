FROM bvlc/caffe:gpu

WORKDIR $CAFFE_ROOT/build
RUN make all -j"$(nproc)" && \
    make install

WORKDIR /root
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

WORKDIR /root/openpose
RUN bash ./install_openpose.sh
