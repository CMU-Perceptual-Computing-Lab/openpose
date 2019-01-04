#!/bin/bash

# Set default environment variables
set -e
WITH_CMAKE=${WITH_CMAKE:-true}
WITH_PYTHON=${WITH_PYTHON:-false}
WITH_CUDA=${WITH_CUDA:-true}
WITH_CUDNN=${WITH_CUDNN:-true}
WITH_OPEN_CL=${WITH_OPEN_CL:-false}
WITH_MKL=${WITH_MKL:-false}

RUN_EXAMPLES=$WITH_CMAKE && ! $WITH_PYTHON && ! $WITH_CUDA && ! $WITH_OPEN_CL && ! $WITH_MKL
