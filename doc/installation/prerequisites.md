OpenPose - Prerequisites
==========================

## Contents
1. [General Tips](#general-tips)
2. [Ubuntu Prerequisites](#ubuntu-prerequisites)
3. [Mac OS Prerequisites](#mac-os-prerequisites)
4. [Windows Prerequisites](#windows-prerequisites)



### General Tips
**Very important**: New Nvidia model GPUs (e.g., Nvidia V, GTX 2080, v100, any Nvidia with Volta or Turing architecture, etc.) require (at least) CUDA 10. CUDA 8 would fail!

In addition, CMake automatically downloads all the OpenPose models. However, **some firewall or company networks block these downloads**.

You might prefer to download them manually:
- [BODY_25 model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel): download in `models/pose/body_25/`.
- [COCO model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel): download in `models/pose/coco/`.
- [MPI model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel): download in `models/pose/mpi/`.
- [Face model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/face/pose_iter_116000.caffemodel): download in `models/face/`.
- [Hands model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel): download in `models/hand/`.



### Ubuntu Prerequisites
1. Ubuntu - **Anaconda should not be installed** on your system. Anaconda includes a Protobuf version that is incompatible with Caffe. Either you uninstall anaconda and install protobuf via apt-get, or you compile your own Caffe and link it to OpenPose.
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
        - Assuming your CMake downloaded folder is in {CMAKE_FOLDER_PATH}, everytime these instructions mentions `cmake-gui`, you will have to replace that line by `{CMAKE_FOLDER_PATH}/bin/cmake-gui`.
    - Ubuntu 14 or 16: Run the command `sudo apt-get install cmake-qt-gui`. Note: If you prefer to use CMake through the command line, see [doc/installation/installation.md#CMake-Command-Line-Configuration-(Ubuntu-Only)](./installation/installation.md#cmake-command-line-configuration-ubuntu-only).
3. Nvidia GPU version prerequisites:
    1. **Note: OpenPose has been tested extensively with CUDA 8.0 (cuDNN 5.1) for Ubuntu 14 and 16, CUDA 10.1 (cuDNN 7.5.1) for Ubuntu 18, and CUDA 11 for Ubuntu 20**. We highly recommend using those versions for those Operating Systems to minimize potential installation issues. Other versions should also work, but we do not provide support about any CUDA/cuDNN installation/compilation issue, as well as problems related to their integration into OpenPose.
    2. **CUDA**:
        - Ubuntu 20 ([**CUDA 11.1**](https://developer.nvidia.com/cuda-downloads)): Download CUDA 11.1 from their [official website](https://developer.nvidia.com/cuda-downloads). Most Ubuntu computers use the `Architecture` named `x86_64`, and we personally recommend the `Installer Type` named `runfile (local)`. Then, follow the Nvidia website installation instructions. When installing, make sure to enable the symbolic link in `usr/local/cuda` to minimize potential future errors.
        - Ubuntu 18 ([**CUDA 10.1**](https://developer.nvidia.com/cuda-10.1-download-archive-base)): Analog to the instructions for Ubuntu 20, but using CUDA version 10.1.
        - Ubuntu 14 or 16 ([**CUDA 8**](https://developer.nvidia.com/cuda-80-ga2-download-archive) **or 10**): Run `sudo ./scripts/ubuntu/install_cuda.sh` (if Ubuntu 16 or 14 and for Graphic cards up to 10XX) or alternatively download and install it from their website.
    3. **cuDNN**:
        - Download it (usually called `cuDNN Library for Linux (x86_64)`):
            - Ubuntu 20: [**cuDNN 8.0.4**](https://developer.nvidia.com/cudnn).
            - Ubuntu 18: [**cuDNN 7.5.1**](https://developer.nvidia.com/rdp/cudnn-archive).
            - Ubuntu 14 or 16 (**cuDNN 5.1 or 7.2**): Run `sudo ./scripts/ubuntu/install_cudnn.sh` (if Ubuntu 16 or 14 and for Graphic cards up to 10XX) or alternatively [download it from their website](https://developer.nvidia.com/rdp/cudnn-archive).
        - And install it:
            - In order to manually install it (any version), just unzip it and copy (merge) its contents on the CUDA folder, usually `/usr/local/cuda-{version}/` in Ubuntu and `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{version}\` in Windows.
5. AMD GPU version prerequisites (only if you do not have an Nvidia GPU and want to run on AMD graphic cards):
    - Ubuntu 20 or 18: Not tested and not officially supported. Try at your own risk. You might want to use the CPU version if no Nvidia GPU is available.
    - Ubuntu 14 or 16:
        1. Download 3rd party ROCM driver for Ubuntu from [**AMD - OpenCL**](https://rocm.github.io/ROCmInstall.html).
        2. Install `sudo apt-get install libviennacl-dev`.
6. Install **Caffe, OpenCV, and Caffe prerequisites**:
    - OpenCV must be already installed on your machine. It can be installed with `sudo apt-get install libopencv-dev`. You could also use your own compiled OpenCV version.
    - Caffe prerequisites: By default, OpenPose uses Caffe under the hood. If you have not used Caffe previously, install its dependencies by running `sudo bash ./scripts/ubuntu/install_deps.sh` after installing your desired CUDA and cuDNN versions. If you are using Ubuntu 14 or 16, you can simply run `sudo bash ./scripts/ubuntu/install_deps_and_cuda.sh` (if Ubuntu 16 or 14 and for Graphic cards up to 10XX).
7. **Eigen prerequisite** (optional, only required for some specific extra functionality, such as extrinsic camera calibration). You can perform any of the 2 following options (but only 1 of them!)
    - Recommended: Simply set the `WITH_EIGEN` flag to `AUTOBUILD`. CMake will automatically download it and configure OpenPose to use it. If you prefer to download it manually (or if your firewall blocks CMake from downloading it):
        - [Eigen 3.3.8](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/eigen_2020_11_18.zip): Unzip as `3rdparty/eigen/`.
    - Advanced (not recommended): If you set `WITH_EIGEN` to `FIND`, you must have Eigen already installed in your system. Note that [Eigen <= 3.3.6 is not supported by CUDA >=9.1](https://bitbucket.org/eigen/eigen/commits/034b6c3e101792a3cc3ccabd9bfaddcabe85bb58?at=default). In order to install it (make sure that Eigen version is compatible with CUDA!):
        - Run `sudo apt-get install libeigen3-dev` and link CMake to the right CMake.
    - Advanced (not recommended): Or you could also use your own version of Eigen by setting `WITH_EIGEN` to `AUTOBUILD`, click `Configure` to let CMake download the zip file, and replace `3rdparty/eigen/` by your own version.



### Mac OS Prerequisites
1. If you don't have `brew`, install it by running `bash scripts/osx/install_brew.sh` on your terminal.
2. Install **CMake GUI**: Run the command `brew cask install cmake`.
3. Install **Caffe, OpenCV, and Caffe prerequisites**: Run `bash scripts/osx/install_deps.sh`.

<a name="problem-with-installing-caffe-from-homebrew"></a>
#### [Problem with installing Caffe from homebrew]:

If you have installed Caffe from homebrew, you might run into an `Attempt to free invalid pointer`. In such a case, the **leveldb** dependency of Caffe is causing an issue. There is a very simple fix.

Note: If you have not, uninstall the current Caffe bottle:
`brew uninstall caffe`

Go to the 3rd party folder and clone the caffe repo:
```
cd 3rdparty
rm -rf caffe
git clone https://github.com/BVLC/caffe.git
```

Now, we have to make a change to the Caffe cmake configurations:

```
vi caffe/cmake/Modules/FindvecLib.cmake
```

Out here, look for the following line:
`find_path(vecLib_INCLUDE_DIR vecLib.h`

and under that there should be a `PATH` variable.

Set the path Variable to `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/` or where ever else you have your vecLib headers. This is cruicial.

Ensure that you have `NO_DEFAULT_PATH` set.

The file should now contain this:

```
find_path(vecLib_INCLUDE_DIR vecLib.h
          DOC "vecLib include directory"
          PATHS /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/
          NO_DEFAULT_PATH)
```
This will fix the `vecLib` not found issues when `BUILD_CAFFE` is run.
Now we have to remove the `leveldb` dependency, and build without it. The `leveldb` dependency is the library that is causing the issue of `Attempt to Free Invalid Pointer`, not the `tcmalloc`. This is a documented issue on their [github repo](https://github.com/google/leveldb/issues/634).

To do this we need to edit the `CMakeLists.txt` file for caffe.

Run `vi caffe/CMakeLists.txt`.

Look for the following line: `caffe_option(USE_LEVELDB "Build with levelDB" ON)`.
We need to set this to off:

`caffe_option(USE_LEVELDB "Build with levelDB" OFF)`

Once you are done with this, you may continue the build process as usual.
Make sure that `BUILD_CAFFE` is set to ON.

4. **Eigen prerequisite** (optional, only required for some specific extra functionality, such as extrinsic camera calibration):
    - Enable the `WITH_EIGEN` flag when running CMake, and set it to `AUTOBUILD`.
    - CMake will automatically download Eigen.
    - Alternatively, you can manually download it from the [Eigen3 website](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/eigen_2020_11_18.zip), and unzip as `3rdparty/eigen/`.



### Windows Prerequisites
NOTE: These instructions are only required when compiling OpenPose brom source. If you simply wanna use the OpenPose binaries for Windows, skip this step.

1. Install **CMake GUI**: Download and install the `Latest Release` of CMake `Windows win64-x64 Installer` from the [CMake download website](https://cmake.org/download/), called `cmake-X.X.X-win64-x64.msi`.
2. Install **Microsoft Visual Studio (VS) 2019 Enterprise**, **Microsoft Visual Studio (VS) 2017 Enterprise** or **VS 2015 Enterprise Update 3**:
    - **IMPORTANT**: Enable all C++-related flags when selecting the components to install.
    - Different VS versions:
        - If **Visual Studio 2019 Community** (or 2017) is desired, we do not officially support it, but it should run similarly to VS 2017/2019 Enterprise.
3. Nvidia GPU version prerequisites:
    1. **Note: OpenPose has been tested extensively with CUDA 11.1.1 / cuDNN 8.0.5 for VS2019, CUDA 10.1 / cuDNN 7.5.1 for VS2017, and CUDA 8.0 / cuDNN 5.1 for VS 2015**. We highly recommend using those versions to minimize potential installation issues. Other versions should also work, but we do not provide support about any CUDA/cuDNN installation/compilation issue, as well as problems related to their integration into OpenPose.
    2. Install one out of [**CUDA 11.1**](https://developer.nvidia.com/cuda-downloads), [**CUDA 10.1**](https://developer.nvidia.com/cuda-10.1-download-archive-base), or [**CUDA 8**](https://developer.nvidia.com/cuda-80-ga2-download-archive):
        - Install CUDA 11.1/10.0/8.0 after Visual Studio 2019/2017/2015 is installed to assure that the CUDA installation will generate all necessary files for VS. If CUDA was installed before installing VS, then re-install CUDA.
        - **Important installation tips**:
            - If CMake returns and error message similar to `CUDA_TOOLKIT_ROOT_DIR not found or specified` or any other CUDA component missing, then: 1) Re-install Visual Studio 2015; 2) Reboot your PC; 3) Re-install CUDA (in this order!).
    3. [**cuDNN 8.0.5**](https://developer.nvidia.com/cudnn), [**cuDNN 7.5.1**](https://developer.nvidia.com/rdp/cudnn-archive), or [**cuDNN 5.1**](https://developer.nvidia.com/rdp/cudnn-archive):
        - In order to manually install it, just unzip it and copy (merge) the contents on the CUDA folder, usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{version}` in Windows and `/usr/local/cuda-{version}/` in Ubuntu.
4. AMD GPU version prerequisites (only if you do not have an Nvidia GPU and want to run on AMD graphic cards):
    1. Download the official AMD drivers for Windows from [**AMD - Windows**](https://support.amd.com/en-us/download).
    2. The libviennacl package comes packaged inside OpenPose for Windows (i.e., no further action required).
5. **Caffe, OpenCV, and Caffe prerequisites**:
    - CMake automatically downloads all the Windows DLLs. Alternatively, you might prefer to download them manually:
        - Dependencies:
            - Note: Leave the zip files in `3rdparty/windows/` so that CMake does not try to download them again.
            - Caffe (if you are not sure which one you need, donwload the default one):
                - [CUDA Caffe (Default)](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe_16_2020_11_14.zip): Unzip as `3rdparty/windows/caffe/`.
                - [CPU Caffe](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe_cpu_2018_05_27.zip): Unzip as `3rdparty/windows/caffe_cpu/`.
                - [OpenCL Caffe](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe_opencl_2018_02_13.zip): Unzip as `3rdparty/windows/caffe_opencl/`.
            - [Caffe dependencies](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe3rdparty_16_2020_11_14.zip): Unzip as `3rdparty/windows/caffe3rdparty/`.
            - [OpenCV 4.2.0](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/opencv_450_v15_2020_11_18.zip): Unzip as `3rdparty/windows/opencv/`.
6. **Eigen prerequisite** (optional, only required for some specific extra functionality, such as extrinsic camera calibration):
    - Set the `WITH_EIGEN` flag in CMake to `AUTOBUILD`.
    - CMake will automatically download Eigen.
    - Alternatively, you can manually download it from the [Eigen3 website](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/eigen_2020_11_18.zip), run CMake so that OpenPose downloads the zip file, and then replace the contents of `3rdparty/eigen/` by your own version.
