#!/bin/bash

# Set default environment variables
set -e
WITH_CMAKE=${WITH_CMAKE:-true}
WITH_PYTHON=${WITH_PYTHON:-false}
WITH_CUDA=${WITH_CUDA:-true}
WITH_CUDNN=${WITH_CUDNN:-true}
WITH_OPEN_CL=${WITH_OPEN_CL:-false}
WITH_MKL=${WITH_MKL:-false}

# Examples should be run (Travis not compatible with GPU code)
if [[ $WITH_CMAKE ]] && [[ ! $WITH_PYTHON ]] && [[ ! $WITH_CUDA ]] && [[ ! $WITH_OPEN_CL ]] && [[ ! $WITH_MKL ]]; then
  RUN_EXAMPLES=true
else
  RUN_EXAMPLES=false
fi

# Makefile configuration compatible
if [[ ! $WITH_PYTHON ]] ; then
  MAKE_CONFIG_COMPATIBLE=true
else
  MAKE_CONFIG_COMPATIBLE=false
fi
