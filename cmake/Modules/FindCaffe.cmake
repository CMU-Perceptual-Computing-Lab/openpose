# Copied from
# https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/cnn_3dobj/FindCaffe.cmake

unset(Caffe_FOUND)
unset(Caffe_INCLUDE_DIRS)
unset(Caffe_LIBS)

find_path(Caffe_INCLUDE_DIRS NAMES
    caffe/caffe.hpp
    caffe/common.hpp
    caffe/net.hpp
    caffe/proto/caffe.pb.h
    caffe/util/io.hpp
    HINTS
    ${CMAKE_BINARY_DIR}/caffe/include
    NO_DEFAULT_PATH)
    

find_library(Caffe_LIBS NAMES caffe
    HINTS
    ${CMAKE_BINARY_DIR}/caffe/lib
    ${CMAKE_BINARY_DIR}/caffe/lib/x86_64-linux-gnu
    NO_DEFAULT_PATH)

if (Caffe_LIBS AND Caffe_INCLUDE_DIRS)
  set(Caffe_FOUND 1)
endif (Caffe_LIBS AND Caffe_INCLUDE_DIRS)
