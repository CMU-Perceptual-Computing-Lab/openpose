OpenPose - Prerequisites
==========================

## Contents
1. [General Tips](#general-tips)
2. [Ubuntu Prerequisites](#ubuntu-prerequisites)
3. [Mac OS Prerequisites](#mac-os-prerequisites)
4. [Windows Prerequisites](#windows-prerequisites)



### General Tips
**Very important**: New Nvidia model GPUs (e.g., Nvidia V, GTX 2080, v100, any Nvidia with Volta or Turing architecture, etc.) require (at least) CUDA 10. CUDA 8 would fail!

In addition, CMake automatically downloads all the OpenPose models. However, **some firewall or company networks block these downloads**. You might prefer to download them manually:

    - [BODY_25 model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel): download in `models/pose/body_25/`.
    - [COCO model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel): download in `models/pose/coco/`.
    - [MPI model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel): download in `models/pose/mpi/`.
    - [Face model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/face/pose_iter_116000.caffemodel): download in `models/face/`.
    - [Hands model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel): download in `models/hand/`.



### Ubuntu Prerequisites
1. Ubuntu - **Anaconda should not be installed** on your system. Anaconda includes a Protobuf version that is incompatible with Caffe. Either you uninstall anaconda and install protobuf via apt-get, or you compile your own Caffe and link it to OpenPose.
2. Install **CMake GUI**:
    - Ubuntu 14 or 16: run the command `sudo apt-get install cmake-qt-gui`. Note: If you prefer to use CMake through the command line, see [CMake Command Line Configuration (Ubuntu Only)](#cmake-command-line-configuration-ubuntu-only).
    - Ubuntu 18: **Download and compile CMake-gui from source**. The default CMake-gui version (3.10) installed via `sudo apt-get install cmake-qt-gui` provokes some compiling errors. Required CMake version >= 3.12.
        - Uninstall your current Cmake-gui version by running `sudo apt purge cmake-qt-gui`.
        - Run `sudo apt-get install qtbase5-dev`.
        - Download the `Latest Release` of `CMake Unix/Linux Source` from the [CMake download website](https://cmake.org/download/), called `cmake-X.X.X.tar.gz`.
        - Unzip it and go inside that folder from the terminal.
        - Run `./configure --qt-gui`. Make sure no error occurred.
        - Run `./bootstrap && make -j8 && make install -j8`. Make sure no error occurred.
        - Assuming your CMake downloaded folder is in {CMAKE_FOLDER_PATH}, everytime these instructions mentions `cmake-gui`, you will have to replace that line by `{CMAKE_FOLDER_PATH}/bin/cmake-gui`.
3. Nvidia GPU version prerequisites:
    1. **Note: OpenPose has been tested extensively with CUDA 8.0 (cuDNN 5.1) and CUDA 10.0 (cuDNN 7.5)**. We highly recommend using those versions to minimize potential installation issues. Other versions should also work, but we do not provide support about any CUDA/cuDNN installation/compilation issue, as well as problems relate dto their integration into OpenPose.
    2. **CUDA**:
        - Ubuntu 14 or 16 ([**CUDA 8**](https://developer.nvidia.com/cuda-80-ga2-download-archive) **or 10**): Run `sudo ./scripts/ubuntu/install_cuda.sh` (if Ubuntu 16 or 14 and for Graphic cards up to 10XX) or alternatively download and install it from their website.
        - Ubuntu 18 ([**CUDA 10**](https://developer.nvidia.com/cuda-downloads)): Download the latest Nvidia CUDA version from their [official website](https://developer.nvidia.com/cuda-downloads).
            - Select "Linux" -> "x86_64" -> "Ubuntu" -> "18.04" -> "runtime (local)", and download it.
            - Follow the Nvidia website installation instructions. Make sure to enable the symbolic link in `usr/local/cuda` to minimize potential future errors.
    3. **cuDNN**:
        - Ubuntu 14 or 16 ([**cuDNN 5.1**](https://developer.nvidia.com/rdp/cudnn-archive) **or 7.2**): Run `sudo ./scripts/ubuntu/install_cudnn.sh` (if Ubuntu 16 or 14 and for Graphic cards up to 10XX) or alternatively download and install it from their website.
        - Ubuntu 18 ([**cuDNN 7.2**](https://developer.nvidia.com/cudnn)): Download and install it from the [Nvidia website](https://developer.nvidia.com/cudnn).
        - In order to manually install it (any version), just unzip it and copy (merge) the contents on the CUDA folder, usually `/usr/local/cuda/` in Ubuntu and `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0` in Windows.
5. AMD GPU version prerequisites:
    1. Ubuntu 14 or 16:
        1. Download 3rd party ROCM driver for Ubuntu from [**AMD - OpenCL**](https://rocm.github.io/ROCmInstall.html).
        2. Install `sudo apt-get install libviennacl-dev`.
    2. Ubuntu 18: Not tested and not officially supported. Try at your risk.
6. Install **Caffe, OpenCV, and Caffe prerequisites**:
    - Caffe prerequisites: By default, OpenPose uses Caffe under the hood. If you have not used Caffe previously, install its dependencies by running `sudo bash ./scripts/ubuntu/install_deps_and_cuda.sh` (if Ubuntu 16 or 14 and for Graphic cards up to 10XX) or run `sudo bash ./scripts/ubuntu/install_deps.sh` after installing your desired CUDA and cuDNN versions.
    - OpenCV must be already installed on your machine. It can be installed with `apt-get install libopencv-dev`. You can also use your own compiled OpenCV version.
7. **Eigen prerequisite** (optional, only required for some specific extra functionality, such as extrinsic camera calibration):
    - If you enable the `WITH_EIGEN` flag when running CMake. You can either:
        1. Do not do anything if you set the `WITH_EIGEN` flag to `BUILD`, CMake will automatically download Eigen. Alternatively, you might prefer to download it manually:
            - [Eigen3](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/eigen_2018_05_23.zip): Unzip as `3rdparty/eigen/`.
        2. Run `sudo apt-get install libeigen3-dev` if you prefer to set `WITH_EIGEN` to `FIND`.
        3. Use your own version of Eigen by setting `WITH_EIGEN` to `BUILD`, run CMake so that OpenPose downloads the zip file, and then replace the contents of `3rdparty/eigen/` by your own version.



### Mac OS Prerequisites
1. If you don't have `brew`, install it by running `bash scripts/osx/install_brew.sh` on your terminal.
2. Install **CMake GUI**: Run the command `brew cask install cmake`.
3. Install **Caffe, OpenCV, and Caffe prerequisites**: Run `bash scripts/osx/install_deps.sh`.
4. **Eigen prerequisite** (optional, only required for some specific extra functionality, such as extrinsic camera calibration):
    - Enable the `WITH_EIGEN` flag when running CMake, and set it to `BUILD`.
    - CMake will automatically download Eigen.
    - Alternatively, you can manually download it from the [Eigen3 website](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/eigen_2018_05_23.zip), and unzip as `3rdparty/eigen/`.



### Windows Prerequisites
NOTE: These instructions are only required when compiling OpenPose brom source. If you simply wanna use the OpenPose binaries for Windows, skip this step.

1. Install **CMake GUI**: Download and install the `Latest Release` of CMake `Windows win64-x64 Installer` from the [CMake download website](https://cmake.org/download/), called `cmake-X.X.X-win64-x64.msi`.
2. Install **Microsoft Visual Studio (VS) 2017 Enterprise** or **VS 2015 Enterprise Update 3**:
    - **IMPORTANT**: Enable all C++-related flags when selecting the components to install.
    - Different VS versions:
        - If **Visual Studio 2017 Community** is desired, we do not officially support it, but it should run similarly to VS 2017 Enterprise.
        - VS 2015 Community and Enterprise Update 1 might give some compiler errors. They have not been tested and they are totally not supported (use VS 2017 Community instead).
3. Nvidia GPU version prerequisites:
    1. **Note: OpenPose has been tested extensively with CUDA 10.0 / cuDNN 7.5 for VS2017 and CUDA 8.0 / cuDNN 5.1 for VS 2015**. We highly recommend using those versions to minimize potential installation issues. Other versions should also work, but we do not provide support about any CUDA/cuDNN installation/compilation issue, as well as problems related to their integration into OpenPose.
    2. [**CUDA 10**](https://developer.nvidia.com/cuda-downloads) or [**CUDA 8**](https://developer.nvidia.com/cuda-80-ga2-download-archive):
        - Install CUDA 8.0/10.0 after Visual Studio 2015/2017 is installed to assure that the CUDA installation will generate all necessary files for VS. If CUDA was already installed, re-install it.
        - **Important installation tips**:
            - (Windows issue, reported Sep 2018): If your computer hangs when installing CUDA drivers, try installing first the [Nvidia drivers](http://www.nvidia.com/Download/index.aspx), and then installing CUDA without the Graphics Driver flag.
            - If CMake returns and error message similar to `CUDA_TOOLKIT_ROOT_DIR not found or specified` or any other CUDA component missing, then: 1) Re-install Visual Studio 2015; 2) Reboot your PC; 3) Re-install CUDA (in this order!).
    3. [**cuDNN 7.5**](https://developer.nvidia.com/cudnn) or [**cuDNN 5.1**](https://developer.nvidia.com/rdp/cudnn-archive):
        - In order to manually install it, just unzip it and copy (merge) the contents on the CUDA folder, usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0` in Windows and `/usr/local/cuda/` in Ubuntu.
4. AMD GPU version prerequisites:
    1. Download official AMD drivers for Windows from [**AMD - Windows**](https://support.amd.com/en-us/download).
    2. The libviennacl package comes packaged inside OpenPose for Windows (i.e., no further action required).
5. **Caffe, OpenCV, and Caffe prerequisites**:
    - CMake automatically downloads all the Windows DLLs. Alternatively, you might prefer to download them manually:
        - Dependencies:
            - Note: Leave the zip files in `3rdparty/windows/` so that CMake does not try to download them again.
            - [Caffe](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe_15_2019_03_14.zip): Unzip as `3rdparty/windows/caffe/`.
            - [Caffe dependencies](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe3rdparty_15_2019_03_14.zip): Unzip as `3rdparty/windows/caffe3rdparty/`.
            - [OpenCV 4.0.1](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/opencv_401_v14_15_2019_03_14.zip): Unzip as `3rdparty/windows/opencv/`.
6. **Eigen prerequisite** (optional, only required for some specific extra functionality, such as extrinsic camera calibration):
    - Enable the `WITH_EIGEN` flag when running CMake, and set it to `BUILD`.
    - CMake will automatically download Eigen.
    - Alternatively, you can manually download it from the [Eigen3 website](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/eigen_2018_05_23.zip), run CMake so that OpenPose downloads the zip file, and then replace the contents of `3rdparty/eigen/` by your own version.
