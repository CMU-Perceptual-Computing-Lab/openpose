#!/bin/bash

# Set default environment variables

set -e

WITH_CMAKE=${WITH_CMAKE:-true}
WITH_PYTHON=${WITH_PYTHON:-false}
WITH_CUDA=${WITH_CUDA:-true}
WITH_CUDNN=${WITH_CUDNN:-true}
WITH_MKL=${WITH_MKL:-false}
