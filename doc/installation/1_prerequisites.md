OpenPose Doc - Installation - Prerequisites
==========================

## Contents
1. [General Tips](#general-tips)
2. [Ubuntu Prerequisites](#ubuntu-prerequisites)
3. [Mac OS Prerequisites](#mac-os-prerequisites)
4. [Windows Prerequisites](#windows-prerequisites)



## General Tips
These tips are **very important** and avoid many bugs:
- Install the latest CUDA version or make sure your GPU is compatible with the CUDA version you have in your system. E.g., Nvidia 30XX GPUs require at least CUDA 11, others (GTX 20XX, V100, Volta or Turing GPUs) require at least CUDA 10.
- CMake automatically downloads all the OpenPose models. However, **some firewall or company networks block these downloads**. If so, you might need to download them manually:
    - [BODY_25 model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel): download in `models/pose/body_25/`.
    - [COCO model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel): download in `models/pose/coco/`.
    - [MPI model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel): download in `models/pose/mpi/`.
    - [Face model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/face/pose_iter_116000.caffemodel): download in `models/face/`.
    - [Hands model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel): download in `models/hand/`.



## Ubuntu Prerequisites
1. **Anaconda should not be installed** on your system **or should be deactivated**. Anaconda includes a Protobuf version that is incompatible with Caffe. Either you uninstall anaconda and install protobuf via apt-get, or you deactivate Conda with the command ```conda deactivate``` (twice if you are not in the base environment).

2. Install **CMake GUI**:
    - Ubuntu 20: Run the command `sudo apt-get install cmake-qt-gui`.
    - Ubuntu 18: **Download and compile CMake-gui from source**. The default CMake-gui version (3.10) installed via `sudo apt-get install cmake-qt-gui` provokes some compiling errors. Required CMake version >= 3.12.
        - Uninstall your current Cmake-gui version by running `sudo apt purge cmake-qt-gui`.
        - Install OpenSSL for building CMake by running `sudo apt install libssl-dev`.
        - Run `sudo apt-get install qtbase5-dev`.
        - Download the `Latest Release` of `CMake Unix/Linux Source` from the [CMake download website](https://cmake.org/download/), called `cmake-X.X.X.tar.gz`.
        - Unzip it and go inside that folder from the terminal.
        - Run `./configure --qt-gui`. Make sure no error occurred.
        - Run ``./bootstrap && make -j`nproc` && sudo make install -j`nproc` ``. Make sure no error occurred.
        - Assuming your CMake downloaded folder is in {CMAKE_FOLDER_PATH}, every time these instructions mentions `cmake-gui`, you will have to replace that line by `{CMAKE_FOLDER_PATH}/bin/cmake-gui`.
    - Ubuntu 14 or 16: Run the command `sudo apt-get install cmake-qt-gui`. Note: If you prefer to use CMake through the command line, see [doc/installation/0_index.md#CMake-Command-Line-Configuration-(Ubuntu-Only)](0_index.md#cmake-command-line-configuration-ubuntu-only).
3. Nvidia GPU version prerequisites:
    1. **Note: OpenPose has been tested extensively with CUDA 11.7.1 (cuDNN 8.5.0) for Ubuntu 20**. Older OpenPose versions (v1.6.X and v1.5.X) were tested with **CUDA 10.1 (cuDNN 7.5.1) for Ubuntu 18 and CUDA 8.0 (cuDNN 5.1) for Ubuntu 14 and 16**. We highly recommend using those combinations to minimize potential installation issues. Other combinations should also work, but we do not provide any support about installation/compilation issues related to CUDA/cuDNN or their integration with OpenPose. Note: If Secure Boot is enabled (by default it is not), the MOK key installation part might be mandatory. For that, record the public key output path and invoke into `sudo mokutil --import PATH_TO_PUBLIC_KEY` manually if automatic install failed.
    2. Upgrade your Nvidia drivers to the latest version.
        - For Ubuntu 20, download ([515.65](https://www.nvidia.com/Download/driverResults.aspx/191961/en-us/))
    3. **CUDA**: You can simply run `sudo bash ./scripts/ubuntu/install_cuda.sh` if you are not too familiar with CUDA. If you are, then you could also do one of the following instead:
        - Ubuntu 20 ([**CUDA 11.7.1**](https://developer.nvidia.com/cuda-11-7-1-download-archive)): Download CUDA 11.7.1 from their [official website](https://developer.nvidia.com/cuda-11-7-1-download-archive). Most Ubuntu computers use the `Architecture` named `x86_64`, and we personally recommend the `Installer Type` named `runfile (local)`. Then, follow the Nvidia website installation instructions. When installing, make sure to enable the symbolic link in `usr/local/cuda` to minimize potential future errors. If the (Nvidia) drivers were installed manually, untick the "install driver" option.
        - Ubuntu 18 ([**CUDA 10.1**](https://developer.nvidia.com/cuda-10.1-download-archive-base)): Analog to the instructions for Ubuntu 20, but using CUDA version 10.1.
        - Ubuntu 14 or 16 ([**CUDA 8**](https://developer.nvidia.com/cuda-80-ga2-download-archive) **or 10**): Run `sudo ./scripts/ubuntu/install_cuda.sh` (if Ubuntu 16 or 14 and for Graphic cards up to 10XX) or alternatively download and install it from their website.
    4. **cuDNN**:
        - Download it (usually called `cuDNN Library for Linux (x86_64)`):
            - Ubuntu 20: [**cuDNN 8.5.0**](https://developer.nvidia.com/cudnn). cuDNN is currently not recommended due to performance degradation issues outlined in [#1864](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1864#issuecomment-774706976).
            - Ubuntu 18: [**cuDNN 7.5.1**](https://developer.nvidia.com/rdp/cudnn-archive).
            - Ubuntu 14 or 16 (**cuDNN 5.1 or 7.2**): Run `sudo ./scripts/ubuntu/install_cudnn_up_to_Ubuntu16.sh` (if Ubuntu 16 or 14 and for Graphic cards up to 10XX) or alternatively [download it from their website](https://developer.nvidia.com/rdp/cudnn-archive).
        - And install it:
            - In order to manually install it (any version), just unzip it and copy (merge) its contents on the CUDA folder, usually `/usr/local/cuda-{version}/` in Ubuntu and `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{version}\` in Windows.
4. OpenCL / AMD GPU version prerequisites (only if you do not have an Nvidia GPU and want to run on AMD graphic cards):
    - Ubuntu 20 or 18: Not tested and not officially supported. Try at your own risk. You might want to use the CPU version if no Nvidia GPU is available.
    - Ubuntu 14 or 16:
        1. Download 3rd party ROCM driver for Ubuntu from [**AMD - OpenCL**](https://rocm.github.io/ROCmInstall.html).
        2. Install `sudo apt-get install libviennacl-dev`.
5. Install **Caffe, OpenCV, and Caffe prerequisites**:
    - OpenCV must be already installed on your machine. It can be installed with `sudo apt-get install libopencv-dev`. You could also use your own compiled OpenCV version.
    - Caffe prerequisites: By default, OpenPose uses Caffe under the hood. If you have not used Caffe previously, install its dependencies by running `sudo bash ./scripts/ubuntu/install_deps.sh` after installing your desired CUDA and cuDNN versions.
    - CMake config generation prerequisites (they might be already installed by default): `sudo apt install protobuf-compiler libgoogle-glog-dev`.
    - OpenPose make prerequisites (they might be already installed by default): `sudo apt install libboost-all-dev libhdf5-dev libatlas-base-dev`.
6. Python prerequisites (optional, only if you plan to use the Python API): python-dev, Numpy (for array management), and OpenCV (for image loading).
```
# Python 3 (default and recommended)
sudo apt-get install python3-dev
sudo pip3 install numpy opencv-python

# Python 2
sudo apt-get install python-dev
sudo pip install numpy opencv-python
```



## Mac OS Prerequisites
1. If you don't have `brew`, install it by running `bash scripts/osx/install_brew.sh` on your terminal.
2. Install **CMake GUI**: Run the command `brew cask install cmake`.
3. Install **Caffe, OpenCV, and Caffe prerequisites**: Run `bash scripts/osx/install_deps.sh`.



## Windows Prerequisites
NOTE: These instructions are only required when compiling OpenPose from source. If you simply wanna use the OpenPose binaries for Windows, skip this step.

1. Install **CMake GUI**: Download and install the `Latest Release` of CMake `Windows win64-x64 Installer` from the [CMake download website](https://cmake.org/download/), called `cmake-X.X.X-win64-x64.msi`.
2. Install **Microsoft Visual Studio (VS) 2019 Enterprise**, **Microsoft Visual Studio (VS) 2017 Enterprise** or **VS 2015 Enterprise Update 3**:
    - **IMPORTANT**: Enable all C++-related flags when selecting the components to install.
    - Different VS versions:
        - If **Visual Studio 2019 Community** (or 2017) is desired, we do not officially support it, but it should run similarly to VS 2017/2019 Enterprise.
3. Nvidia GPU version prerequisites:
    1. **Note: OpenPose has been tested extensively with CUDA 11.1.1 (cuDNN 8.1.0) for VS2019**. Older OpenPose versions (v1.6.X and v1.5.X) were tested with **CUDA 10.1 (cuDNN 7.5.1) for VS2017 and CUDA 8.0 (cuDNN 5.1) for VS2015**. We highly recommend using those combinations to minimize potential installation issues. Other combinations should also work, but we do not provide any support about installation/compilation issues related to CUDA/cuDNN or their integration with OpenPose.
    2. Upgrade your Nvidia drivers to the latest version (in the Nvidia "GeForce Experience" software or its [website](https://www.nvidia.com/Download/index.aspx)).
    3. Install one out of [**CUDA 11.1.1**](https://developer.nvidia.com/cuda-11.1.1-download-archive), [**CUDA 10.1**](https://developer.nvidia.com/cuda-10.1-download-archive-base), or [**CUDA 8**](https://developer.nvidia.com/cuda-80-ga2-download-archive):
        - Install CUDA 11.1.1/10.0/8.0 after Visual Studio 2019/2017/2015 is installed to assure that the CUDA installation will generate all necessary files for VS. If CUDA was installed before installing VS, then re-install CUDA.
        - **Important installation tips**:
            - If CMake returns and error message similar to `CUDA_TOOLKIT_ROOT_DIR not found or specified` or any other CUDA component missing, then: 1) Re-install Visual Studio 2015; 2) Reboot your PC; 3) Re-install CUDA (in this order!).
    4. [**cuDNN 8.1.0**](https://developer.nvidia.com/cudnn), [**cuDNN 7.5.1**](https://developer.nvidia.com/rdp/cudnn-archive), or [**cuDNN 5.1**](https://developer.nvidia.com/rdp/cudnn-archive):
        - In order to manually install it, just unzip it and copy (merge) the contents on the CUDA folder, usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{version}` in Windows and `/usr/local/cuda-{version}/` in Ubuntu.
4. AMD GPU version prerequisites (only if you do not have an Nvidia GPU and want to run on AMD graphic cards):
    1. Download the official AMD drivers for Windows from [**AMD - Windows**](https://support.amd.com/en-us/download).
    2. The libviennacl package comes packaged inside OpenPose for Windows (i.e., no further action required).
5. **Caffe, OpenCV, and Caffe prerequisites**:
    - CMake automatically downloads all the Windows DLLs. Alternatively, you might prefer to download them manually:
        - Dependencies:
            - Note: Leave the zip files in `3rdparty/windows/` so that CMake does not try to download them again.
            - Caffe (if you are not sure which one you need, download the default one):
                - [CUDA Caffe (Default)](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe_16_2020_11_14.zip): Unzip as `3rdparty/windows/caffe/`.
                - [CPU Caffe](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe_cpu_2018_05_27.zip): Unzip as `3rdparty/windows/caffe_cpu/`.
                - [OpenCL Caffe](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe_opencl_2018_02_13.zip): Unzip as `3rdparty/windows/caffe_opencl/`.
            - [Caffe dependencies](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe3rdparty_16_2020_11_14.zip): Unzip as `3rdparty/windows/caffe3rdparty/`.
            - [OpenCV 4.2.0](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/opencv_450_v15_2020_11_18.zip): Unzip as `3rdparty/windows/opencv/`.
7. Python prerequisites (optional, only if you plan to use the Python API): Install any [Python 3.X](https://www.python.org/downloads/windows/) version for Windows, and then:
```
sudo pip install numpy opencv-python
```
