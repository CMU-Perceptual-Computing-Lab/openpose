#!/bin/bash
# set default environment variables

set -e

WITH_CMAKE=${WITH_CMAKE:-true}
WITH_PYTHON3=${WITH_PYTHON3:-false}
WITH_CUDA=${WITH_CUDA:-true}
WITH_CUDNN=${WITH_CUDNN:-true}
