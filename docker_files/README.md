# OpenPose GPU Docker Image

This folder contains the supporting files to build a GPU-enabled OpenPose docker image that contains:

* CUDA 8.0 + cuDNN 5 from `nvidia/cuda` base image
* OpenPose (built from current repository)
* OpenPose Python bindings
* OpenCV (`opencv-contrib-python` package from PyPI, defined in `requirements.txt`)
* Any required dependencies
* a non-root "openpose" user for executing code

The `Dockerfile` is in the root of the repository.

System requirements:

* NVIDIA drivers >= 375.26
* NVIDIA container runtime ([nvidia-docker](https://github.com/NVIDIA/nvidia-docker))

## Using the Docker Image

### 1. Building the image

Clone this repository (assume into a folder in current working directory called `openpose`)

```
cd openpose
git submodule update --init --recursive

# this command will build a container with tag `openpose`
nvidia-docker build . -t openpose
```

### 2. Testing the image

```
# if you have not yet done so
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose

cd openpose

# this assumes the openpose repository is present at /home/$USER/openpose
# and mounts the examples into the correct folders inside the container
# then runs example in container with tag `openpose`
nvidia-docker run --rm -v /home/$USER/openpose/models:/openpose/models openpose bash -c "python3 /openpose/examples/tutorial_api_python/openpose_python.py"
```