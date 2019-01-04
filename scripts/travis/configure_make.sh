#!/bin/bash

# Raw Makefile configuration

LINE () {
  echo "$@" >> Makefile.config
}

cp Makefile.config.example Makefile.config

if $WITH_PYTHON ; then
  # TODO(lukeyeager) this path is currently disabled because of test errors like:
  #   ImportError: dynamic module does not define init function (PyInit__caffe)
  LINE "PYTHON_LIBRARIES := python3.4m boost_python-py34"
  LINE "PYTHON_INCLUDE := /usr/include/python3.4 /usr/lib/python3/dist-packages/numpy/core/include"
  LINE "INCLUDE_DIRS := \$(INCLUDE_DIRS) \$(PYTHON_INCLUDE)"
fi

if $WITH_CUDA ; then
  # Only build SM50
  LINE "CUDA_ARCH := -gencode arch=compute_52,code=sm_50"
else
  LINE "CPU_ONLY := 1"
fi

if $WITH_CUDNN ; then
  LINE "USE_CUDNN := 1"
fi

