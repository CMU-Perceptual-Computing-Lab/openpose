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
# if [[ $WITH_CMAKE == true ]] && [[ $WITH_PYTHON == true ]] && [[ $WITH_CUDA == false ]] && [[ $WITH_OPEN_CL == false ]] && [[ $WITH_MKL == false ]]; then
if [[ $WITH_CUDA == false ]] && [[ $WITH_OPEN_CL == false ]]; then
  RUN_EXAMPLES=true
else
  RUN_EXAMPLES=false
fi
echo "RUN_EXAMPLES = ${RUN_EXAMPLES}."

# Makefile configuration compatible
if [[ $WITH_PYTHON == false ]] ; then
  MAKE_CONFIG_COMPATIBLE=true
else
  MAKE_CONFIG_COMPATIBLE=false
fi
echo "MAKE_CONFIG_COMPATIBLE = ${MAKE_CONFIG_COMPATIBLE}."
