OpenPose - Installation (deprecated)
======================================

## Contents
1. [Operating Systems](#operating-systems)
2. [Requirements](#requirements)
3. [Clone OpenPose](#clone-openpose)
4. [Update OpenPose](#update-openpose)
5. [Ubuntu](#ubuntu)
6. [Windows](#windows)
7. [Doxygen Documentation Autogeneration](#doxygen-documentation-autogeneration)
8. [Custom Caffe](#custom-caffe)
9. [Compiling without cuDNN](#compiling-without-cudnn)



## Operating Systems
See [doc/installation.md#operating-systems](./installation.md#operating-systems).



## Requirements
See [doc/installation.md#requirements](./installation.md#requirements).



## Clone OpenPose
See [doc/installation.md#clone-openpose](./installation.md#clone-openpose).



## Update OpenPose
See [doc/installation.md#update-openpose](./installation.md#update-openpose).



## Ubuntu
### Installation - CMake
Recommended installation method, it is simpler and offers more customization settings. See [doc/installation.md](installation.md).



### Prerequisites (Script Compilation or Manual Compilation)
CUDA, cuDNN, OpenCV and Atlas must be already installed on your machine:

    1. [CUDA](https://developer.nvidia.com/cuda-80-ga2-download-archive) must be installed. You should reboot your machine after installing CUDA.
    2. [cuDNN](https://developer.nvidia.com/cudnn): Once you have downloaded it, just unzip it and copy (merge) the contents on the CUDA folder, e.g. `/usr/local/cuda-8.0/`. Note: We found OpenPose working ~10% faster with cuDNN 5.1 compared to cuDNN 6. Otherwise, check [Compiling without cuDNN](#compiling-without-cudnn).
    3. OpenCV can be installed with `apt-get install libopencv-dev`. If you have compiled OpenCV 3 by your own, follow [Manual Compilation](#manual-compilation). After both Makefile.config files have been generated, edit them and uncomment the line `# OPENCV_VERSION := 3`. You might alternatively modify all `Makefile.config.UbuntuXX` files and then run the scripts in step 2.
    4. In addition, OpenCV 3 does not incorporate the `opencv_contrib` module by default. Assuming you have OpenCV 3 compiled with the contrib module and you want to use it, append `opencv_contrib` at the end of the line `LIBRARIES += opencv_core opencv_highgui opencv_imgproc` in the `Makefile` file.
    5. Atlas can be installed with `sudo apt-get install libatlas-base-dev`. Instead of Atlas, you can use OpenBLAS or Intel MKL by modifying the line `BLAS := atlas` in the same way as previosuly mentioned for the OpenCV version selection.



### Installation - Script Compilation
Build Caffe & the OpenPose library + download the required Caffe models for Ubuntu 14.04 or 16.04 (auto-detected for the script) and CUDA 8:
```bash
bash ./ubuntu/install_caffe_and_openpose_if_cuda8.sh
```
**Highly important**: This script only works with CUDA 8 and Ubuntu 14 or 16. Otherwise, see [doc/installation.md](installation.md) or [Installation - Manual Compilation](#installation---manual-compilation).



### Installation - Manual Compilation
Alternatively to the script installation, if you want to use CUDA 7, avoid using sh scripts, change some configuration labels (e.g. OpenCV version), etc., then:
1. Install the [Caffe prerequisites](http://caffe.berkeleyvision.org/installation.html).
2. Compile Caffe and OpenPose by running these lines:
    ```
    ### Install Caffe ###
    git submodule update --init --recursive
    cd 3rdparty/caffe/
    # Select your desired Makefile file (run only one of the next 4 commands)
    cp Makefile.config.Ubuntu14_cuda7.example Makefile.config # Ubuntu 14, cuda 7
    cp Makefile.config.Ubuntu14_cuda8.example Makefile.config # Ubuntu 14, cuda 8
    cp Makefile.config.Ubuntu16_cuda7.example Makefile.config # Ubuntu 16, cuda 7
    cp Makefile.config.Ubuntu16_cuda8.example Makefile.config # Ubuntu 16, cuda 8
    # Change any custom flag from the resulting Makefile.config (e.g. OpenCV 3, Atlas/OpenBLAS/MKL, etc.)
    # Compile Caffe
    make all -j`nproc` && make distribute -j`nproc`

    ### Install OpenPose ###
    cd ../../models/
    bash ./getModels.sh # It just downloads the Caffe trained models
    cd ..
    cp ubuntu/Makefile.example Makefile
    # Same file cp command as the one used for Caffe
    cp ubuntu/Makefile.config.Ubuntu14_cuda7.example Makefile.config
    # Change any custom flag from the resulting Makefile.config (e.g. OpenCV 3, Atlas/OpenBLAS/MKL, etc.)
    make all -j`nproc`
    ```

    NOTE: If you want to use your own Caffe distribution, follow the steps on [Custom Caffe](#custom-caffe) section and later re-compile the OpenPose library:
    ```
    bash ./install_openpose_if_cuda8.sh
    ```
    Note: These steps only need to be performed once. If you are interested in making changes to the OpenPose library, you can simply recompile it with:
    ```
    make clean
    make all -j$(NUM_CORES)
    ```
**Highly important**: There are 2 `Makefile.config.Ubuntu##.example` analogous files, one in the main folder and one in [3rdparty/caffe/](../3rdparty/caffe/), corresponding to OpenPose and Caffe configuration files respectively. Any change must be done to both files (e.g. OpenCV 3 flag, Atlab/OpenBLAS/MKL flag, etc.). E.g. for CUDA 8 and Ubuntu16: [3rdparty/caffe/Makefile.config.Ubuntu16_cuda8.example](../3rdparty/caffe/Makefile.config.Ubuntu16.example) and [ubuntu/Makefile.config.Ubuntu16_cuda8.example](../ubuntu/Makefile.config.Ubuntu16_cuda8.example).



### Reinstallation
If you updated some software that our library or 3rdparty use, or you simply want to reinstall it:
1. Clean the OpenPose and Caffe compilation folders:
```
make clean && cd 3rdparty/caffe && make clean
```
2. Repeat the [Installation](#installation) steps. You do not need to download the models again.



### Uninstallation
You just need to remove the OpenPose folder, by default called `openpose/`. E.g. `rm -rf openpose/`.



## Windows
### Installation - Library
1. Install the pre-requisites:
    1. **Microsoft Visual Studio (VS) 2015 Enterprise Update 3**.
        - If **Visual Studio 2017 Community** is desired, we do not officially support it, but it might be compiled by firstly [enabling CUDA 8.0 in VS2017](https://stackoverflow.com/questions/43745099/using-cuda-with-visual-studio-2017?answertab=active#tab-top) or use **VS2017 with CUDA 9** by checking the `.vcxproj` file and changing the necessary paths from CUDA 8 to 9.
        - VS 2015 Enterprise Update 1 will give some compiler errors and VS 2015 Community has not been tested.
    2. [**CUDA 8**](https://developer.nvidia.com/cuda-80-ga2-download-archive): Install it on the default location, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0`. Otherwise, modify the Visual Studio project solution accordingly. Install CUDA 8.0 after Visual Studio 2015 is installed to assure that the CUDA installation will generate all necessary files for VS. If CUDA was already installed, re-install it after installing VS!
    3. [**cuDNN 5.1**](https://developer.nvidia.com/cudnn): Once you have downloaded it, just unzip it and copy (merge) the contents on the CUDA folder, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0`.

#### CMake Installer
Recommended installation method, it is simpler and offers more customization settings. See [doc/installation.md](installation.md). Note that it is a beta version, post in GitHub any issue you find.


#### Deprecated Windows Installer
Note: This installer will not incorporate any new features, we recommend to use the CMake installer.

1. Download the OpenPose dependencies and models (body, face and hand models) by double-clicking on `{openpose_path}\windows\download_3rdparty_and_models.bat`. Alternatively, you might prefer to download them manually:
    - Models:
        - [COCO model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel): download in `models/pose/coco/`.
        - [MPI model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel): download in `models/pose/mpi/`.
        - [Face model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/face/pose_iter_116000.caffemodel): download in `models/face/`.
        - [Hands model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel): download in `models/hand/`.
    - Dependencies:
        - [Caffe](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe_2018_01_18.zip): Unzip as `3rdparty/windows/caffe/`.
        - [Caffe dependencies](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe3rdparty_2017_07_14.zip): Unzip as `3rdparty/windows/caffe3rdparty/`.
        - [OpenCV 3.1](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/opencv_310.zip): Unzip as `3rdparty/windows/opencv/`.
2. Open the Visual Studio project sln file by double-cliking on `{openpose_path}\windows\OpenPose.sln`.
3. In order to verify OpenPose is working, try compiling and executing the demo:
    1. Right click on `OpenPoseDemo` --> `Set as StartUp Project`.
    2. Change `Debug` by `Release` mode.
    3. Compile it and run it with <kbd>F5</kbd> or the green play icon.
4. If you have a webcam connected, OpenPose will automatically start after being compiled.
5. In order to use the created exe file from the command line (i.e. outside Visual Studio), you have to:
    1. Copy all the DLLs located on `{openpose_folder}\3rdparty\windows\caffe\bin\` on the exe folder: `{openpose_folder}\windows\x64\Release`.
    2. Copy all the DLLs located on `{openpose_folder}\3rdparty\windows\opencv\x64\vc14\bin\` on the exe folder: `{openpose_folder}\windows\x64\Release`.
    3. Open the Windows cmd (Windows button + <kbd>X</kbd>, then <kbd>A</kbd>).
    4. Go to the OpenPose directory, assuming OpenPose has been downloaded on `C:\openpose`: `cd C:\openpose\`.
    5. Run the tutorial commands.
6. Check OpenPose was properly installed by running it on the default images, video or webcam: [doc/quick_start.md#quick-start](./quick_start.md#quick-start).



### Uninstallation
You just need to remove the OpenPose or portable demo folder.



### Reinstallation
If you updated some software that our library or 3rdparty use, or you simply want to reinstall it:
1. Open the Visual Studio project sln file by double-cliking on `{openpose_path}\windows\OpenPose.sln`.
2. Clean the OpenPose project by right-click on `Solution 'OpenPose'` and `Clean Solution`.
3. Compile it and run it with <kbd>F5</kbd>  or the green play icon.





## Doxygen Documentation Autogeneration
You can generate the documentation by running the following command. The documentation will be generated in `doc/doxygen/html/index.html`. You can simply open it with double-click (your default browser should automatically display it).
```
cd doc/
doxygen doc_autogeneration.doxygen
```





## Custom Caffe
We only modified some Caffe compilation flags and minor details. You can use your own Caffe distribution, these are the files we added and modified:

1. Added files: `install_caffe.sh`; as well as `Makefile.config.Ubuntu14.example`, `Makefile.config.Ubuntu16.example`, `Makefile.config.Ubuntu14_cuda_7.example` and `Makefile.config.Ubuntu16_cuda_7.example` (extracted from `Makefile.config.example`). Basically, you must enable cuDNN.
2. Edited file: Makefile. Search for "# OpenPose: " to find the edited code. We basically added the C++11 flag to avoid issues in some old computers.
3. Optional - deleted Caffe file: `Makefile.config.example`.
4. In order to link it to OpenPose:
    1. Run `make all && make distribute` in your Caffe version.
    2. Open the OpenPose Makefile config file: `./Makefile.config.UbuntuX.example` (where X depends on your OS and CUDA version).
    3. Modify the Caffe folder directory variable (`CAFFE_DIR`) to your custom Caffe `distribute` folder location in the previous OpenPose Makefile config file.





## Compiling without cuDNN
The [cuDNN](https://developer.nvidia.com/cudnn) library is not mandatory, but required for full keypoint detection accuracy. In case your graphics card is not compatible with cuDNN, you can disable it by:

- Ubuntu: Disable `USE_CUDNN` in the `Makefile.config` file in `3rdparty/caffe`, and recompiling Caffe.
- Windows: Compiling Caffe by your own with without cuDNN support and replacing the [3rdparty/windows/caffe](../3rdparty/windows/caffe)) folder by your own implementation.

Then, you would have to reduce the `--net_resolution` flag to fit the model into the GPU memory. You can try values like `640x320`, `320x240`, `320x160`, or `160x80` to see your GPU memory capabilities. After finding the maximum approximate resolution that your GPU can handle without throwing an out-of-memory error, adjust the `net_resolution` ratio to your image or video to be processed (see the `--net_resolution` explanation from [doc/demo_overview.md](./demo_overview.md)).
