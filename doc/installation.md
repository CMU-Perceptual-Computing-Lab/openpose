OpenPose Library - Compilation and Installation
====================================

## Contents
1. [Requirements](#requirements)
2. [Ubuntu](#ubuntu)
3. [Windows](#windows)
4. [Quick Start](#quick-start)
5. [FAQ](#faq)



## Requirements
- Ubuntu (tested on 14 and 16) or Windows (tested on 10). We do not support any other OS but the community has been able to install it on: CentOS, Windows 7, and Windows 8.
- GPU with at least 1.5 GB available (the `nvidia-smi` command checks the available GPU memory in Ubuntu).
- CUDA and cuDNN installed.
- At least 2 GB of free RAM memory.
- Highly recommended: A CPU with at least 8 cores.

Note: These requirements assume the default configuration (i.e. `--net_resolution "656x368"` and `num_scales 1`). You might need more (with a greater net resolution and/or number of scales) or less resources (with smaller net resolution and/or using the MPI and MPI_4 models).





## Ubuntu
### Installation - Script Compilation
**Highly important**: This script only works with CUDA 8 and Ubuntu 14 or 16. Otherwise, check [Manual Compilation](#manual-compilation).
1. Required: CUDA, cuDNN, OpenCV and Atlas must be already installed on your machine.
    1. OpenCV can be installed with `apt-get install libopencv-dev`. If you have compiled OpenCV 3 by your own, follow [Manual Compilation](#manual-compilation). After both Makefile.config files have been generated, edit them and uncomment the line `# OPENCV_VERSION := 3`. You might alternatively modify all `Makefile.config.UbuntuXX` files and then run the scripts in step 2.
    2. In addition, OpenCV 3 does not incorporate the `opencv_contrib` module by default. Assuming you have OpenCV 3 compiled with the contrib module and you want to use it, append `opencv_contrib` at the end of the line `LIBRARIES += opencv_core opencv_highgui opencv_imgproc` in the `Makefile` file.
    3. Atlas can be installed with `sudo apt-get install libatlas-base-dev`. Instead of Atlas, you can use OpenBLAS or Intel MKL by modifying the line `BLAS := atlas` in the same way as previosuly mentioned for the OpenCV version selection.
2. Build Caffe & the OpenPose library + download the required Caffe models for Ubuntu 14.04 or 16.04 (auto-detected for the script) and CUDA 8:
```
chmod u+x install_caffe_and_openpose.sh
./install_caffe_and_openpose.sh
```



### Installation - Manual Compilation
Alternatively to the script installation, if you want to use CUDA 7, avoid using sh scripts, change some configuration labels (e.g. OpenCV version), etc., then:
1. Install the [Caffe prerequisites](http://caffe.berkeleyvision.org/installation.html).
2. Compile Caffe and OpenPose by running these lines:
    ```
    ### Install Caffe ###
    cd 3rdparty/caffe/
    # Select your desired Makefile file (run only one of the next 4 commands)
    cp Makefile.config.Ubuntu14_cuda_7.example Makefile.config # Ubuntu 14, cuda 7
    cp Makefile.config.Ubuntu14.example Makefile.config # Ubuntu 14, cuda 8
    cp Makefile.config.Ubuntu16_cuda_7.example Makefile.config # Ubuntu 16, cuda 7
    cp Makefile.config.Ubuntu16.example Makefile.config # Ubuntu 16, cuda 8
    # Change any custom flag from the resulting Makefile.config (e.g. OpenCV 3, Atlas/OpenBLAS/MKL, etc.)
    # Compile Caffe
    make all -j${number_of_cpus} && make distribute -j${number_of_cpus}

    ### Install OpenPose ###
    cd ../../models/
    ./getModels.sh # It just downloads the Caffe trained models
    cd ..
    # Same file cp command as the one used for Caffe
    cp Makefile.config.Ubuntu14_cuda_7.example Makefile.config
    # Change any custom flag from the resulting Makefile.config (e.g. OpenCV 3, Atlas/OpenBLAS/MKL, etc.)
    make all -j${number_of_cpus}
    ```

    NOTE: If you want to use your own Caffe distribution, follow the steps on `Custom Caffe` section and later re-compile the OpenPose library:
    ```
    chmod u+x install_openpose.sh
    ./install_openpose.sh
    ```
    Note: These steps only need to be performed once. If you are interested in making changes to the OpenPose library, you can simply recompile it with:
    ```
    make clean
    make all -j$(NUM_CORES)
    ```
**Highly important**: There are 2 `Makefile.config.Ubuntu##.example` analogous files, one in the main folder and one in [3rdparty/caffe/](../3rdparty/caffe/), corresponding to OpenPose and Caffe configuration files respectively. Any change must be done to both files (e.g. OpenCV 3 flag, Atlab/OpenBLAS/MKL flag, etc.). E.g. for CUDA 8 and Ubuntu16: [3rdparty/caffe/Makefile.config.Ubuntu16.example](../3rdparty/caffe/Makefile.config.Ubuntu16.example) and [Makefile.config.Ubuntu16.example](../Makefile.config.Ubuntu16.example).



### Reinstallation
If you updated some software that our library or 3rdparty use, or you simply want to reinstall it:
1. Clean the OpenPose and Caffe compilation folders:
```
make clean && cd 3rdparty/caffe && make clean
```
2. Repeat the [Installation](#installation) steps.



### Uninstallation
You just need to remove the OpenPose folder, by default called `openpose/`. E.g. `rm -rf openpose/`.





## Windows
### Installation - Demo
1. Install the pre-requisites:
    1. [CUDA 8](https://developer.nvidia.com/cuda-downloads): Install it on the default location, C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0. Otherwise, modify the Visual Studio project solution accordingly.
    2. [cuDNN 5.1](https://developer.nvidia.com/cudnn): Once you have downloaded it, just unzip it and copy (merge) the contents on the CUDA folder, C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0.
    3. [Microsoft Visual C++ 2015 Redistributable](https://www.microsoft.com/en-us/download/details.aspx?id=53587) (lighter) or Microsoft Visual Studio 2015 (only if you intend to use the library).
2. Download the portable demo from: [posefs1.perception.cs.cmu.edu/OpenPose/openpose_1.0.0rc2.zip](http://posefs1.perception.cs.cmu.edu/OpenPose/openpose_1.0.0rc2.zip).

### Installation - Library
1. Install the pre-requisites:
    1. Install all the demo pre-requisites.
    2. [Python 2.4.13 64 bits - Windows x86-64 MSI installer](https://www.python.org/downloads/release/python-2713/).
        - Install it on C:\Python27 (default) or D:\Programs\Python27. Otherwise, modify the VS solution accordingly.
        - In addition, open the Windows cmd (Windows button + X, then A), and install some Python libraries with this command: `pip install numpy protobuf hypothesis`.
    3. [Cmake](https://cmake.org/download/): Select the option to add it to the Windows PATH.
    4. [Ninja](https://ninja-build.org/): Select the option to add it to the Windows PATH.
    5. Microsoft Visual Studio 2015.
2. Download the `Windows` branch of Openpose by either cliking on `Download ZIP` on [openpose/tree/windows](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/windows) or cloning the repository: `git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose/ && git checkout windows`.
3. Install Caffe on Windows:
    1. Open the Windows cmd (Windows button + X, then A).
    2. Go to the Caffe directory, assuming OpenPose has been downloaded on `C:\openpose`: `cd C:\openpose\3rdparty\caffe\caffe-windows`.
    3. Compile Caffe by running: `scripts\build_win.cmd`. It will take several minutes.
        - Note: If Caffe asks you: `Does D:\openpose\3rdparty\caffe\caffe-windows\build\..\..\include\caffe specify a file name or directory name on the target (F = file, D = directory)?`, select `D`.
    4. If you find any problem installing Caffe, check [http://caffe.berkeleyvision.org/](http://caffe.berkeleyvision.org/).
4. You can now open the Visual Studio sln file located on `{openpose_path}\windows_project\OpenPose.sln`.
5. In order to verify OpenPose is working, try compiling and executing the demo:
    1. Right click on `OpenPoseDemo` --> `Set as StartUp Project`.
    2. Change `Debug` by `Release` mode.
    3. You can now compile it.
6. Download the body pose models:
    1. Download the [COCO model (18 key-points)](http://posefs1.perception.cs.cmu.edu/Users/tsimon/Projects/coco/data/models/coco/pose_iter_440000.caffemodel) as `{openpose_folder}\models\pose\coco\pose_iter_440000.caffemodel`.
    2. (Optionally) download the [MPI model (15 key-points, faster and less memory than COCO)](http://posefs1.perception.cs.cmu.edu/Users/tsimon/Projects/coco/data/models/mpi/pose_iter_160000.caffemodel) as `{openpose_folder}\models\pose\mpi\pose_iter_160000.caffemodel`.
7. If you have a webcam connected, you can test it by pressing the F5 key or the green play icon.
8. Otherwise, check [Quick Start](#quick-start) to verify OpenPose was properly compiled. In order to use the created exe from the command line, you have to:
    1. Copy all the DLLs located on `{openpose_folder}\3rdparty\caffe\caffe-windows\build\install\bin\` on the exe folder: `{openpose_folder}\windows_project\x64\Release`.
    2. Copy `opencv_ffmpeg310_64.dll`, `opencv_video310.dll` and `opencv_videoio310.dll` from `{openpose_folder}\3rdparty\caffe\dependencies\libraries_v140_x64_py27_1.1.0\libraries\x64\vc14\bin\` on the exe folder: `{openpose_folder}\windows_project\x64\Release`.





## Quick Start
Check that the library is working properly by using any of the following commands. Note that `examples/media/video.avi` and `examples/media` exist, so you do not need to change the paths. In addition, the following commands assume your terminal (Ubuntu) or cmd (Windows) are located in the OpenPose folder.

**1. Running on Video**
```
# Ubuntu
./build/examples/openpose/openpose.bin --video examples/media/video.avi
```
```
:: Windows - Demo
bin\OpenPoseDemo.exe --video examples/media/video.avi
```
```
:: Windows - Library
windows_project\x64\Release\OpenPoseDemo.exe --video examples/media/video.avi
```

**2. Running on Webcam**
```
# Ubuntu
./build/examples/openpose/openpose.bin
```
```
:: Windows - Demo
bin\OpenPoseDemo.exe
```
```
:: Windows - Library
windows_project\x64\Release\OpenPoseDemo.exe
```

**3. Running on Images**
```
# Ubuntu
./build/examples/openpose/openpose.bin --image_dir examples/media/
```
```
:: Windows - Demo
bin\OpenPoseDemo.exe --image_dir examples/media/
```
```
:: Windows - Library
windows_project\x64\Release\OpenPoseDemo.exe --image_dir examples/media/
```

The visual GUI should show the original image with the poses blended on it, similarly to the pose of this gif:
<p align="center">
    <img src="media/shake.gif", width="720">
</p>

If you choose to visualize a body part or a PAF (Part Affinity Field) heat map with the command option `--part_to_show`, the result should be similar to one of the following images:
<p align="center">
    <img src="media/body_heat_maps.png", width="720">
</p>

<p align="center">
    <img src="media/paf_heat_maps.png", width="720">
</p>





## FAQ
**Q: Out of memory error** - I get an error similar to: `Check failed: error == cudaSuccess (2 vs. 0)  out of memory`.

**A**: Most probably cuDNN is not installed/enabled, the default Caffe model uses >12 GB of GPU memory, cuDNN reduces it to ~1.5 GB.


**Q: Low speed** - OpenPose is quite slow, is it normal? How can I speed it up?

**A**: Check the Benchmark to discover the approximate speed of your graphics card: [https://github.com/CMU-Perceptual-Computing-Lab/openpose#openpose-benchmark](https://github.com/CMU-Perceptual-Computing-Lab/openpose#openpose-benchmark). Some speed tips:

    1. Use cuDNN 5.1 (cuDNN 6 is ~10% slower).
    2. If you have more than 1 GPU, set `--num_gpu`.
    3. Reduce the `--net_resolution` (e.g. to 320x176) (lower accuracy).
    4. Use the `MPI_4_layers` model (lower accuracy and lower number of parts).
