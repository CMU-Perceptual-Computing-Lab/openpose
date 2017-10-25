OpenPose - Installation and FAQ
====================================

## Contents
1. [Operating Systems](#operating-systems)
2. [Requirements](#requirements)
3. [Clone and Update the Repository](#clone-and-update-the-repository)
4. [Ubuntu](#ubuntu)
5. [Windows](#windows)
6. [OpenPose 3D Demo](#openpose-3d-demo)
7. [Doxygen Documentation Autogeneration](#doxygen-documentation-autogeneration)
8. [Custom Caffe](#custom-caffe)
9. [Compiling without cuDNN](#compiling-without-cudnn)
10. [FAQ](#faq)



## Operating Systems
- **Ubuntu** 14 and 16.
- **Windows** 8 and 10.
- **Nvidia Jetson TX2**, installation instructions in [doc/installation_jetson_tx2.md](./installation_jetson_tx2.md).
- OpenPose has also been used on **Windows 7**, **Mac**, **CentOS**, and **Nvidia Jetson (TK1 and TX1)** embedded systems. However, we do not officially support them at the moment.





## Requirements
- NVIDIA graphics card with at least 1.6 GB available (the `nvidia-smi` command checks the available GPU memory in Ubuntu).
- At least 2 GB of free RAM memory.
- Highly recommended: cuDNN and a CPU with at least 8 cores.

Note: These requirements assume the default configuration (i.e. `--net_resolution "656x368"` and `scale_number 1`). You might need more (with a greater net resolution and/or number of scales) or less resources (with smaller net resolution and/or using the MPI and MPI_4 models).





## Clone and Update the Repository
The first step is to clone the OpenPose repository. It might be done with [GitHub Desktop](https://desktop.github.com/) in Windows and from the terminal in Ubuntu:
```bash
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
```

OpenPose can be easily updated by clicking the `synchronization` button at the top-right part in GitHub Desktop in Windows, or by running `git pull origin master` in Ubuntu. After OpenPose has been updated, just run the `Reinstallation` section described below for your specific Operating System.





## Ubuntu
### Installation - CMake
Recommended installation method. It is simpler and it offers many more customization settings. See [doc/installation_cmake.md](installation_cmake.md). Note that it is a beta version, if it fails, please post in GitHub and use [Installation - Script Compilation](#installation---script-compilation) meanwhile.



### Installation - Script Compilation
**Highly important**: This script only works with CUDA 8 and Ubuntu 14 or 16. Otherwise, see [doc/installation_cmake.md](installation_cmake.md) or [Installation - Manual Compilation](#installation---manual-compilation).
1. Required: CUDA, cuDNN, OpenCV and Atlas must be already installed on your machine.
    1. [CUDA](https://developer.nvidia.com/cuda-80-ga2-download-archive) must be installed. You should reboot your machine after installing CUDA.
    2. [cuDNN](https://developer.nvidia.com/cudnn): Once you have downloaded it, just unzip it and copy (merge) the contents on the CUDA folder, e.g. `/usr/local/cuda-8.0/`. Note: We found OpenPose working ~10% faster with cuDNN 5.1 compared to cuDNN 6. Otherwise, check [Compiling without cuDNN](#compiling-without-cudnn).
    3. OpenCV can be installed with `apt-get install libopencv-dev`. If you have compiled OpenCV 3 by your own, follow [Manual Compilation](#manual-compilation). After both Makefile.config files have been generated, edit them and uncomment the line `# OPENCV_VERSION := 3`. You might alternatively modify all `Makefile.config.UbuntuXX` files and then run the scripts in step 2.
    4. In addition, OpenCV 3 does not incorporate the `opencv_contrib` module by default. Assuming you have OpenCV 3 compiled with the contrib module and you want to use it, append `opencv_contrib` at the end of the line `LIBRARIES += opencv_core opencv_highgui opencv_imgproc` in the `Makefile` file.
    5. Atlas can be installed with `sudo apt-get install libatlas-base-dev`. Instead of Atlas, you can use OpenBLAS or Intel MKL by modifying the line `BLAS := atlas` in the same way as previosuly mentioned for the OpenCV version selection.
2. Build Caffe & the OpenPose library + download the required Caffe models for Ubuntu 14.04 or 16.04 (auto-detected for the script) and CUDA 8:
```bash
bash ./ubuntu/install_caffe_and_openpose_if_cuda8.sh
```



### Installation - Manual Compilation
Alternatively to the script installation, if you want to use CUDA 7, avoid using sh scripts, change some configuration labels (e.g. OpenCV version), etc., then:
1. Install the [Caffe prerequisites](http://caffe.berkeleyvision.org/installation.html).
2. Compile Caffe and OpenPose by running these lines:
    ```
    ### Install Caffe ###
    cd 3rdparty/caffe/
    # Select your desired Makefile file (run only one of the next 4 commands)
    cp Makefile.config.Ubuntu14_cuda7.example Makefile.config # Ubuntu 14, cuda 7
    cp Makefile.config.Ubuntu14_cuda8.example Makefile.config # Ubuntu 14, cuda 8
    cp Makefile.config.Ubuntu16_cuda7.example Makefile.config # Ubuntu 16, cuda 7
    cp Makefile.config.Ubuntu16_cuda8.example Makefile.config # Ubuntu 16, cuda 8
    # Change any custom flag from the resulting Makefile.config (e.g. OpenCV 3, Atlas/OpenBLAS/MKL, etc.)
    # Compile Caffe
    make all -j${number_of_cpus} && make distribute -j${number_of_cpus}

    ### Install OpenPose ###
    cd ../../models/
    bash ./getModels.sh # It just downloads the Caffe trained models
    cd ..
    # Same file cp command as the one used for Caffe
    cp ubuntu/Makefile.config.Ubuntu14_cuda7.example Makefile.config
    # Change any custom flag from the resulting Makefile.config (e.g. OpenCV 3, Atlas/OpenBLAS/MKL, etc.)
    make all -j${number_of_cpus}
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
### Installation - Demo
1. Download and unzip the [portable OpenPose demo 1.0.1](http://posefs1.perception.cs.cmu.edu/OpenPose/OpenPose_demo_1.0.1.zip).



### Installation - Library
1. Install the pre-requisites:
    1. Microsoft Visual Studio (VS) 2015 Enterprise Update 3. If Visual Studio 2017 Community is desired, we do not support it, but it might be compiled by firstly [enabling CUDA 8.0 in VS2017](https://stackoverflow.com/questions/43745099/using-cuda-with-visual-studio-2017?answertab=active#tab-top). VS Enterprise Update 1 will give some compiler errors and VS 2015 Community has not been tested.
    2. [CUDA 8](https://developer.nvidia.com/cuda-80-ga2-download-archive): Install it on the default location, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0`. Otherwise, modify the Visual Studio project solution accordingly. Install CUDA 8.0 after Visual Studio 2015 is installed to assure that the CUDA installation will generate all necessary files for VS. If CUDA was already installed, re-install it after installing VS!
    3. [cuDNN 5.1](https://developer.nvidia.com/cudnn): Once you have downloaded it, just unzip it and copy (merge) the contents on the CUDA folder, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0`.
2. Download the OpenPose dependencies and models (body, face and hand models) by double-clicking on `{openpose_path}\windows\download_3rdparty_and_models.bat`. Alternatively, you might prefer to download them manually:
    - Models:
        - [COCO model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel): download in `models/pose/coco/`.
        - [MPI model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel): download in `models/pose/mpi/`.
        - [Face model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/face/pose_iter_116000.caffemodel): download in `models/face/`.
        - [Hands model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel): download in `models/hand/`.
    - Dependencies:
        - [Caffe](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe_2017_07_11.zip): Unzip as `3rdparty/windows/caffe/`.
        - [Caffe dependencies](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe3rdparty_2017_07_14.zip): Unzip as `3rdparty/windows/caffe3rdparty/`.
        - [OpenCV 3.1](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/opencv_310.zip): Unzip as `3rdparty/windows/opencv/`.
3. Open the Visual Studio project sln file by double-cliking on `{openpose_path}\windows\OpenPose.sln`.
4. In order to verify OpenPose is working, try compiling and executing the demo:
    1. Right click on `OpenPoseDemo` --> `Set as StartUp Project`.
    2. Change `Debug` by `Release` mode.
    3. Compile it and run it with F5 or the green play icon.
5. If you have a webcam connected, OpenPose will automatically start after being compiled.
6. In order to use the created exe file from the command line (i.e. outside Visual Studio), you have to:
    1. Copy all the DLLs located on `{openpose_folder}\3rdparty\windows\caffe\bin\` on the exe folder: `{openpose_folder}\windows\x64\Release`.
    2. Copy all the DLLs located on `{openpose_folder}\3rdparty\windows\opencv\x64\vc14\bin\` on the exe folder: `{openpose_folder}\windows\x64\Release`.
    3. Open the Windows cmd (Windows button + X, then A).
    4. Go to the OpenPose directory, assuming OpenPose has been downloaded on `C:\openpose`: `cd C:\openpose\`.
    5. Run the tutorial commands.
7. Check OpenPose was properly installed by running it on the default images, video or webcam: [doc/quick_start.md#quick-start](./quick_start.md#quick-start).



### Uninstallation
You just need to remove the OpenPose or portable demo folder.



### Reinstallation
If you updated some software that our library or 3rdparty use, or you simply want to reinstall it:
1. Open the Visual Studio project sln file by double-cliking on `{openpose_path}\windows\OpenPose.sln`.
2. Clean the OpenPose project by right-click on `Solution 'OpenPose'` and `Clean Solution`.
3. Compile it and run it with F5 or the green play icon.






## OpenPose 3D Demo
If you want to try our OpenPose 3-D reconstruction demo, see [doc/openpose_3d_reconstruction_demo.md](./openpose_3d_reconstruction_demo.md).





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

Then, you would have to reduce the `--net_resolution` flag to fit the model into the GPU memory. You can try values like "640x320", "320x240", "320x160", or "160x80" to see your GPU memory capabilities. After finding the maximum approximate resolution that your GPU can handle without throwing an out-of-memory error, adjust the `net_resolution` ratio to your image or video to be processed (see the `--net_resolution` explanation from [doc/demo_overview.md](./demo_overview.md)).





## FAQ
**Q: Out of memory error** - I get an error similar to: `Check failed: error == cudaSuccess (2 vs. 0)  out of memory`.

**A**: Most probably cuDNN is not installed/enabled, the default Caffe model uses >12 GB of GPU memory, cuDNN reduces it to ~1.5 GB.



**Q: Low speed** - OpenPose is quite slow, is it normal? How can I speed it up?

**A**: Check the [OpenPose Benchmark](https://docs.google.com/spreadsheets/d/1-DynFGvoScvfWDA1P4jDInCkbD4lg0IKOYbXgEq0sK0/edit#gid=0) to discover the approximate speed of your graphics card. Some speed tips:

    1. Use cuDNN 5.1 (cuDNN 6 is ~10% slower).
    2. Reduce the `--net_resolution` (e.g. to 320x176) (lower accuracy).
    3. For face, reduce the `--face_net_resolution`. The resolution 320x320 usually works pretty decently.
    4. Use the `MPI_4_layers` model (lower accuracy and lower number of parts).
    5. Change GPU rendering by CPU rendering to get approximately +0.5 FPS (`--render_pose 1`).



**Q: Webcam is slow** - Using a folder with images matches the speed FPS benchmarks, but the webcam has lower FPS. Note: often on Windows.

**A**: OpenCV has some issues with some camera drivers (specially on Windows). The first step should be to compile OpenCV by your own and re-compile OpenPose after that (following the `Reinstallation` section in Ubuntu or cleaning the project on Windows). If the speed is still slower, you can better debug it by running a webcam OpenCV example (e.g. [this C++ example](http://answers.opencv.org/question/1/how-can-i-get-frames-from-my-webcam/)). If you are able to get the proper FPS with the OpenCV demo but OpenPose is still low, then let us know!



**Q: Video and/or webcam are not working** - Using a folder with images does work, but the video and/or the webcam do not. Note: often on Windows.

**A**: OpenCV has some issues with some camera drivers and video codecs (specially on Windows). Follow the same steps as the `Webcam is slow` question to test the webcam is working. After re-compiling OpenCV, you can also try this [OpenCV example for video](http://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html).
