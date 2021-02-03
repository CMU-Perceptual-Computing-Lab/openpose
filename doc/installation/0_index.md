OpenPose Doc - Installation
==========================

## Contents
1. [Operating Systems, Requirements, and Dependencies](#operating-systems-requirements-and-dependencies)
2. [Windows Portable Demo](#windows-portable-demo)
3. [Compiling and Running OpenPose from Source](#compiling-and-running-openpose-from-source)
    1. [Problems and Errors Installing OpenPose](#problems-and-errors-installing-openpose)
    2. [Prerequisites](#prerequisites)
    3. [Clone OpenPose](#clone-openpose)
    4. [CMake Configuration](#cmake-configuration)
    5. [Compilation](#compilation)
    6. [Running OpenPose](#running-openpose)
    7. [Custom User Code](#custom-user-code)
4. [Compiling and Running OpenPose from Source on ROS, Docker, and Google Colab - Community-Based Work](#compiling-and-running-openpose-from-source-on-ros-docker-and-google-colab-community-based-work)
5. [Uninstalling, Reinstalling, or Updating OpenPose](#Uninstalling-reinstalling-or-updating-openpose)
6. [Advanced Additional Settings (Optional)](#advanced-additional-settings-optional)
    1. [Deploying or Exporting OpenPose to Other Projects](#deploying-or-exporting-openpose-to-other-projects)
    2. [Maximum Speed](#maximum-speed)
    3. [Faster CPU Version (Ubuntu Only)](#faster-cpu-version-ubuntu-only)
    4. [OpenCL Version](#opencl-version)
    5. [COCO and MPI Models](#coco-and-mpi-models)
    6. [3D Reconstruction Module](#3d-reconstruction-module)
    7. [Calibration Module](#calibration-module)
    8. [Unity Compatible Version](#unity-compatible-version)
    9. [Compiling without cuDNN](#compiling-without-cudnn)
    10. [Custom Caffe](#custom-caffe)
    11. [Custom NVIDIA NVCaffe](#custom-nvidia-nvcaffe)
    12. [Custom OpenCV](#custom-opencv)
    13. [Doxygen Documentation Autogeneration (Ubuntu Only)](#doxygen-documentation-autogeneration-ubuntu-only)
    14. [CMake Command Line Configuration (Ubuntu Only)](#cmake-command-line-configuration-ubuntu-only)





## Operating Systems, Requirements, and Dependencies
- Operating Systems
    - **Windows 10**.
    - **Ubuntu 20**.
    - **Mac OSX** Mavericks and above.
    - **Ubuntu 14, 16 and 18** as well as **Windows 7 and 8** are no longer officially maintained. However, they should still work (but might require minor changes).
    - **Nvidia Jetson TX1** (for JetPack 3.1), installation instructions in [doc/installation/jetson_tx/installation_jetson_tx1.md](jetson_tx/installation_jetson_tx1.md).
    - **Nvidia Jetson TX2** (for JetPack 3.1 or 3.3), installation instructions in [doc/installation/jetson_tx/installation_jetson_tx2_jetpack3.1.md](jetson_tx/installation_jetson_tx2_jetpack3.1.md) and [doc/installation/jetson_tx/installation_jetson_tx2_jetpack3.3.md](jetson_tx/installation_jetson_tx2_jetpack3.3.md) respectively.
    - OpenPose has also been used on **CentOS** and other **Nvidia Jetson (TK1)** embedded systems. However, we do not officially support them at the moment.
- **Requirements** for the default configuration
    - CUDA (Nvidia GPU) version:
        - NVIDIA graphics card with at least 1.6 GB available (the `nvidia-smi` command checks the available GPU memory in Ubuntu).
        - At least 2.5 GB of free RAM memory for BODY_25 model or 2 GB for COCO model (assuming cuDNN installed).
        - Highly recommended: cuDNN.
    - OpenCL (AMD GPU) version:
        - Vega series graphics card
        - At least 2 GB of free RAM memory.
    - CPU-only (no GPU) version:
        - Around 8GB of free RAM memory.
    - Highly recommended: a CPU with at least 8 cores.
- Advanced tip: You might need more resources with a greater `--net_resolution` and/or `scale_number` or less resources by reducing the net resolution and/or using the MPI and MPI_4 models.
- **Dependencies**:
    - OpenCV (all 2.X and 3.X versions are compatible).
    - Caffe and all its dependencies. Have you ported OpenPose into another DL framework (Tensorflow, Caffe2, Pytorch, ...)?. Email us (gines@alumni.cmu.edu) or feel free to make a pull request if you implemented any of those!
    - The demo and tutorials additionally use GFlags.





## Windows Portable Demo
**If you just want to use OpenPose** without compiling or writing any code, simply use the latest portable version of OpenPose for Windows.
1. For maximum speed, you should use OpenPose in a machine with a Nvidia GPU version. If so, you must upgrade your Nvidia drivers to the latest version (in the Nvidia "GeForce Experience" software or its [website](https://www.nvidia.com/Download/index.aspx)).
2. Download the latest OpenPose version from the [Releases](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases) section.
3. Follow the `Instructions.txt` file inside the downloaded zip file to download the models required by OpenPose (about 500 Mb).
4. Then, you can run OpenPose from the PowerShell command-line by following [doc/01_demo.md](../01_demo.md).

Note: If you are using the GPU-accelerated version and are seeing `Cuda check failed (3 vs. 0): initialization error` when running OpenPose, you can fix it by doing one of these:
- Upgrade your Nvidia drivers. If the error persists, make sure your machine does not contain any CUDA version (or if so, that it's the same than the OpenPose portable demo files). Otherwise, uninstall that CUDA version. If you need to keep that CUDA version installed, follow [Compiling and Running OpenPose from Source](#compiling-and-running-openpose-from-source) for that particular CUDA version instead.
- Download an older OpenPose version (v1.6.0 does not show this error).





## Compiling and Running OpenPose from Source
The instructions in the following subsections describe the steps to build OpenPose using CMake-GUI. These instructions are only recommended if you plan to modify the OpenPose code or integrate it with another library or project. You can stop reading this document if you just wanted to run OpenPose on Windows without compiling or modifying any code.



### Problems and Errors Installing OpenPose
Any problem installing OpenPose while following this guidelines? Check [doc/05_faq.md](../05_faq.md) and/or check existing GitHub issues. If you do you find your issue, post a new one. We will not respond to duplicated issues, as well as GitHub issues about Caffe, OpenCV or CUDA installation errors, as well as issues that do not fill all the information that the GitHub template asks for.



### Prerequisites
Make sure to download and install the [prerequisites for your particular operating system](1_prerequisites.md).



### Clone OpenPose
The first step is to clone the OpenPose repository.

1. Windows: You might use [GitHub Desktop](https://desktop.github.com/) or clone it from Powershell.
2. Ubuntu, Mac, or Windows Powershell:
```bash
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
cd openpose/
git submodule update --init --recursive --remote
```



### CMake Configuration
1. Go to the OpenPose folder and open CMake-GUI from it. On Windows, double click on CMake-gui. On Ubuntu, Mac, or Windows Powershell:
```
cd {OpenPose_folder}
mkdir build/
cd build/
cmake-gui ..
```
2. Select the OpenPose directory as project source directory, and a non-existing or empty sub-directory (e.g., `build`) where the Makefile files (Ubuntu) or Visual Studio solution (Windows) will be generated. If `build` does not exist, it will ask you whether to create it. Press `Yes`.
<p align="center">
    <img src="../../.github/media/installation/cmake_im_1.png" width="480">
    <img src="../../.github/media/installation/cmake_im_1_windows.png" width="480">
</p>

3. Press the `Configure` button, keep the generator in `Unix Makefiles` (Ubuntu) or set it to your 64-bit Visual Studio version (Windows), and press `Finish`. Note for Windows users: CMake-GUI has changed their design after version 14. For versions older than 14, you usually select `Visual Studio XX 20XX Win64` as the generator (`X` depends on your VS version), while the `Optional toolset to use` must be empty. However, new CMake versions require you to select only the VS version as the generator, e.g., `Visual Studio 16 2019`, and then you must manually choose `x64` for the `Optional platform for generator`. See the following images as example.
<p align="center">
    <img src="../../.github/media/installation/cmake_im_2.png" width="240">
    <img src="../../.github/media/installation/cmake_im_2_windows.png" width="240">
    <img src="../../.github/media/installation/cmake_im_2_windows_new.png" width="240">
</p>

4. Enabling Python (optional step, only apply it if you plan to use the Python API): Enable the `BUILD_PYTHON` flag and click `Configure` again.

5. Set the `GPU_MODE` flag to the proper value and click `Configure` again:
    1. If your machine has an Nvidia GPU, you should most probably not modify this flag and skip this step. Cases in which you might have to change it:
        - If you have a Nvidia GPU with 2GB of memory or less: Then you will have to follow some of the tricks in [doc/06_maximizing_openpose_speed.md](../06_maximizing_openpose_speed.md) or change `GPU_MODE` back to `CPU_ONLY`.
        - If you cannot install CUDA, then you can also set `GPU_MODE` to `CPU_ONLY`.
    2. Mac OSX and machines with a non-Nvidia GPU (Intel or AMD GPUs): Set the `GPU_MODE` flag to `CPU_ONLY` (easier to install but slower runtime) or `OPENCL` (GPU-accelerated, it is harder to install but provides a faster runtime speed). For more details on OpenCV support, see [doc/1_prerequisites.md](1_prerequisites.md) and [OpenCL Version](#opencl-version).
    3. If your machine does not have any GPU, set the `GPU_MODE` flag to `CPU_ONLY`.

6. If this step is successful, the `Configuring done` text will appear in the bottom box in the last line. Otherwise, some red text will appear in that same bottom box.
<p align="center">
    <img src="../../.github/media/installation/cmake_im_3.png" width="480">
    <img src="../../.github/media/installation/cmake_im_3_windows.png" width="480">
</p>

7. Press the `Generate` button and proceed to [Compilation](#compilation). You can now close CMake.

Note: If you prefer to use your own custom Caffe or OpenCV versions, see [Custom Caffe](#custom-caffe) or [Custom OpenCV](#custom-opencv) respectively.



### Compilation
#### Ubuntu
Run the following commands in your terminal.
```bash
cd build/
make -j`nproc`
```

#### Mac
Run the following commands in your terminal:
```bash
cd build/
make -j`sysctl -n hw.logicalcpu`
```
Advanced tip: Mac provides both `logicalcpu` and `physicalcpu`, but we want the logical number for maximum speed.

If the default compilation fails with Caffe errors, install Caffe separately and set `BUILD_CAFFE` to false in the CMake config. Steps:
- Re-create the build folder: `rm -rf build; mkdir build; cd build`.
- `brew uninstall caffe` to remove the version of Caffe previously installed via cmake.
- `brew install caffe` to install Caffe separately.
- Run `cmake-gui` and make the following adjustments to the cmake config:
    1. `BUILD_CAFFE` set to false.
    2. `Caffe_INCLUDE_DIRS` set to `/usr/local/include/caffe`.
    3. `Caffe_LIBS` set to `/usr/local/lib/libcaffe.dylib`.
    4. Run `Configure` and `Generate` from CMake GUI.

If you face an OpenCV error during compiling time similar to `fatal error: 'opencv2/highgui/highgui.hpp' file not found`, please apply the following patch (this error has been reported in the latest OSX 10.14):
```bash
cd 3rdparty/caffe; git apply ../../scripts/osx/mac_opencl_patch.txt
```

#### Windows
In order to build the project, select and run only one of the 2 following alternatives.

- **CMake-GUI alternative (recommended)**:
    1. Open the Visual Studio solution (Windows) by clicking in `Open Project` in CMake (or alternatively `build/OpenPose.sln`). Then, set the configuration from `Debug` to `Release`.
    2. Press <kbd>F7</kbd> (or `Build` menu and click on `Build Solution`).
    3. **Important for Python version**: Make sure not to skip step 2, it is not enough to click on <kbd>F5</kbd> (Run), you must also `Build Solution` for the Python bindings to be generated.
    4. After it has compiled, and if you have a webcam, you can press the green triangle icon (alternatively <kbd>F5</kbd>) to run the OpenPose demo with the default settings on the webcam.

- Command-line build alternative (not recommended). NOTE: The command line alternative is not officially supported, but it was added in [GitHub issue #1198](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1198). For any questions or bug report about this command-line version, comment in that GitHub issue.
    1. Run "MSVS 2019 Developer Command Console"
    ```batch
    openpose\mkdir  build
    cd build
    cmake .. -G "Visual Studio 16 2019" -A x64 -T v142
    cmake --build . --config Release
    copy x64\Release\*  bin\
    ```
    2. If you want to clean build
    ```batch
    cmake --clean-first .
    cmake --build . --config Release
    copy x64\Release\*  bin\
    ```

**VERY IMPORTANT NOTE**: In order to use OpenPose outside Visual Studio, and assuming you have not unchecked the `BUILD_BIN_FOLDER` flag in CMake, copy all DLLs from `{build_directory}/bin` into the folder where the generated `openpose.dll` and `*.exe` demos are, e.g., `{build_directory}x64/Release` for the 64-bit release version.

If you are facing errors with these instructions, these are a set of alternative instructions created by the community:
- OpenPose for Windows 10, Visual Studio 2019, CMake, and Nvidia GPU: [https://github.com/quickgrid/Build-Guide/blob/master/README.md#windows-10-cmu-openpose-setup-visual-studio-2019-cmake-nvidia-gpu](https://github.com/quickgrid/Build-Guide/blob/master/README.md#windows-10-cmu-openpose-setup-visual-studio-2019-cmake-nvidia-gpu).
- Video-tutorial: OpenPose + Visual Studio 2017 + CUDA 10.0 + cuDNN 7.5 (no portable demo): [https://youtu.be/QC9GTb6Wsb4](https://youtu.be/QC9GTb6Wsb4). For questions, post in GitHub issue #1426.

We welcome users to send us their installation videos (e.g., sharing them as GitHub issue or doing a pull request) and we will post them here.



### Running OpenPose
Check OpenPose was properly installed by running any demo example: [doc/01_demo.md](../01_demo.md).



### Custom User Code
You can quickly add your custom code so that quick prototypes can be easily tested without having to create a whole new project just for it. See [examples/user_code/README.md](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/examples/user_code/README.md) for more details.



## Compiling and Running OpenPose from Source on ROS, Docker, and Google Colab - Community-Based Work
If you do not want to use the Windows portable binaries nor compile it from source code, we add links to some community-based work based on OpenPose. Note: We do not support them, and we will remove new GitHub issues opened asking about them as well as block those users from posting again. If you face any issue, comment only in the GitHub issues links especified below, or ask the owner of them.

- ROS examples:
    - [ROS example 1](https://github.com/ravijo/ros_openpose). For questions and more details, read and post ONLY on [issue thread #891](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/891).
    - [ROS example 2](https://github.com/firephinx/openpose_ros) (based on a very old OpenPose version). For questions and more details, read and post ONLY on [issue thread #51](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/51).

- Docker Images. For questions and more details, read and post ONLY on [issue thread #347](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/347).
    - Dockerfile working also with CUDA 10:
        - Option 1:
            - 1. If necessary, install the latest version of docker. There are extra steps, but if you are on Ubuntu, the main one is `sudo apt-get install docker-ce`. The other steps can be found [here](https://phoenixnap.com/kb/how-to-install-docker-on-ubuntu-18-04).
            - 2. `docker pull exsidius/openpose` - [Guide](https://github.com/gormonn/openpose-docker/blob/master/README.md).
            - 3. [More details here](https://cloud.docker.com/repository/docker/exsidius/openpose/general).
        - [Link 2](https://github.com/esemeniuc/openpose-docker), it claims to also include Python support. Read and post ONLY on [issue thread #1102](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1102).
        - [Link 3](https://github.com/ExSidius/openpose-docker/blob/master/Dockerfile).
        - [Link 4](https://cloud.docker.com/repository/docker/exsidius/openpose/general).
    - Dockerfile working only with CUDA 8:
        - [Dockerfile - OpenPose v1.4.0, OpenCV, CUDA 8, CuDNN 5, Python2.7](https://github.com/tlkh/openpose). Read and post ONLY on [issue thread #1102](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1102).
        - [Dockerfile - OpenPose v1.4.0, OpenCV, CUDA 8, CuDNN 6, Python2.7](https://gist.github.com/moiseevigor/11c02c694fc0c22fccd59521793aeaa6).
        - [Dockerfile - OpenPose v1.2.1](https://gist.github.com/sberryman/6770363f02336af82cb175a83b79de33).

- Google Colab helper scripts: Script to install OpenPose on Google Colab. Really useful when access to a computer powerful enough to run OpenPose is not possible, so one possible way to use OpenPose is to build it on a GPU-enabled Colab runtime and then run the programs there.
    - [Google Colab 1/2](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1736#issuecomment-736846794): For questions and more details, read and post ONLY on [issue thread #1736](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1736).
    - [Google Colab 2/2](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/949#issue-387855863): For questions and more details, read and post ONLY on [issue thread #949](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/949).





## Uninstalling, Reinstalling, or Updating OpenPose
OpenPose can be easily uninstalled:
1. (Ubuntu and Mac) If you ran `sudo make install` (which we do not recommend), then run `sudo make uninstall` in `build/`.
2. Remove the OpenPose folder.

In order to update it or reinstall it:
1. Follow the above steps to uninstall it.
2. Follow the [Compiling and Running OpenPose from Source](#compiling-and-running-openpose-from-source) steps again.





## Advanced Additional Settings (Optional)
### Deploying or Exporting OpenPose to Other Projects
See [doc/advanced/deployment.md](../advanced/deployment.md).





### Maximum Speed
Check the OpenPose Benchmark as well as some hints to speed up and/or reduce the memory requirements to run OpenPose on [doc/06_maximizing_openpose_speed.md](../06_maximizing_openpose_speed.md).



### Faster CPU Version (Ubuntu Only)
**NOTE**: The accuracy of the CPU/OpenCL versions is a bit lower than CUDA version, so the results will very slightly vary. In practice, the different is barely noticeable, so you are safe using these.

This step is only supported for Intel CPUs on Ubuntu versions 16 and 14. It does not compile on Ubuntu 20, and we have not tested it on Ubuntu 18.

After setting the `GPU_MODE` flag to `CPU_ONLY` and clicking `Configured`, search for `USE_MKL` and set it to true. Then, click `Configure` again. This way, OpenPose will link against the Intel MKL version (Math Kernel Library) of Caffe. This speeds up CPU version on Ubuntu roughly about 2-3x, making it as fast as the Windows CPU-only version.

The default CPU version takes about 0.2 images per second on Ubuntu (~50x slower than GPU) while the MKL version provides a roughly 2x speedup at ~0.4 images per second. As of now OpenPose does not support MKL on Windows but will at a later date. Also, MKL version does not support unfixed resolution. So a folder of images of different resolutions requires a fixed net resolution (e.g., `--net_resolution 656x368`).

For MKL, the user can configure the environmental variables `MKL_NUM_THREADS` and `OMP_NUM_THREADS`. They are set at an optimum parameter level by default (i.e., to the number of threads of the machine). However, they can be tweak by running the following commands into the terminal window, right before running any OpenPose application. Eg:

```bash
# Optimal number = Number of threads (used by default)
export MKL_NUM_THREADS="8"
export OMP_NUM_THREADS="8"
```

Increasing the number of threads results in a higher RAM memory usage. You can check the [doc/06_maximizing_openpose_speed.md](../06_maximizing_openpose_speed.md) for more information about speed and memory requirements in several CPUs and GPUs.



### OpenCL Version
**NOTE**: The accuracy of the CPU/OpenCL versions is a bit lower than CUDA version, so the results will very slightly vary. In practice, the different is not barely noticeable, so you are safe using these.

If you have an AMD graphics card, you can compile OpenPose with the OpenCL option. To manually select the OpenCL Version, open CMake GUI mentioned above, and set the `GPU_MODE` flag to `OPENCL` (or non-UI CMake with `GPU_MODE=OPENCL`). **Very important:** If you compiled previously the CPU-only or CUDA versions on that same OpenPose folder, you will have to manually delete the `build` directory and run the installation steps from scratch. Otherwise, many weird errors will appear.

The OpenCL version has been tested on Ubuntu, Windows and OSX. This has been tested only on AMD Vega series and NVIDIA 10 series graphics cards. Please email us if you have issues with other operating systems or graphics cards. Running on OSX on a Mac with an AMD graphics card requires special instructions which can be seen in the section below.

Lastly, OpenCL version does not support unfixed `--net_resolution`. So a folder of images of different resolutions with OpenPose, requires the `--net_resolution 656x368` flag for example. This should be fixed by the Caffe author in a future patch.



### COCO and MPI Models
By default, the body `COCO` and `MPI` models are not downloaded (they are slower and less accurate than `BODY_25`, so not useful in most cases!). But you can download them by turning on the `DOWNLOAD_BODY_COCO_MODEL` or `DOWNLOAD_BODY_MPI_MODEL` flags. Check the differences between these models in [doc/05_faq.md#difference-between-body_25-vs-coco-vs-mpi](../05_faq.md#difference-between-body_25-vs-coco-vs-mpi).



### 3D Reconstruction Module
You can include the 3D reconstruction module by:

1. Install the FLIR camera software, Spinnaker SDK. It is a propietary software, so we cannot provide direct download link. Note: You might skip this step if you intend to use the 3-D OpenPose module with a different camera brand.
    1. Ubuntu: Get and install the latest Spinnaker SKD version in their default path. OpenPose will automatically find it. Otherwise, set the right path with CMake.
    2. Windows: Donwload the latest Spinnaker SKD version from [https://www.ptgrey.com/support/downloads](https://www.ptgrey.com/support/downloads).
        - Copy `{PointGreyParentDirectory}\Point Grey Research\Spinnaker\bin64\vs2015\` as `{OpenPoseDirectory}\3rdparty\windows\spinnaker\bin\`. You can remove all the *.exe files.
        - Copy `{PointGreyParentDirectory}\Point Grey Research\Spinnaker\include\` as `{OpenPoseDirectory}\3rdparty\windows\spinnaker\include\`.
        - Copy `Spinnaker_v140.lib` and `Spinnakerd_v140.lib` from `{PointGreyParentDirectory}\Point Grey Research\Spinnaker\lib64\vs2015\` into `{OpenPoseDirectory}\3rdparty\windows\spinnaker\lib\`.
        - (Optional) Spinnaker SDK overview: [https://www.ptgrey.com/spinnaker-sdk](https://www.ptgrey.com/spinnaker-sdk).
2. Install the 3D visualizer, FreeGLUT:
    1. Ubuntu: run `sudo apt-get update && sudo apt-get install build-essential freeglut3 freeglut3-dev libxmu-dev libxi-dev` and reboot your PC.
    2. Windows:
        1. It is automatically downloaded by the CMake installer.
        2. Alternatively, if you prefer to download it yourself, you could either:
            1. Double click on `3rdparty\windows\getFreeglut.bat`.
            2. Download [this version from our server](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/freeglut_2018_01_14.zip) and unzip it in `{OpenPoseDirectory}\3rdparty\windows\freeglut\`.
            3. Download the latest `MSVC Package` from [http://www.transmissionzero.co.uk/software/freeglut-devel/](http://www.transmissionzero.co.uk/software/freeglut-devel/).
                - Copy `{freeglutParentDirectory}\freeglut\bin\x64\` as `{OpenPoseDirectory}\3rdparty\windows\freeglut\bin\`.
                - Copy `{freeglutParentDirectory}\freeglut\include\` as `{OpenPoseDirectory}\3rdparty\windows\freeglut\include\`.
                - Copy `{freeglutParentDirectory}\freeglut\lib\x64\` as `{OpenPoseDirectory}\3rdparty\windows\freeglut\lib\`.
3. Follow the CMake installation steps. In addition, set the `WITH_FLIR_CAMERA` (only if Spinnaker was installed) and `WITH_3D_RENDERER` options.
4. Increased accuracy with Ceres solver (Ubuntu only): For extra 3-D reconstruction accuracy, run `sudo apt-get install libeigen3-dev`, install [Ceres solver](http://ceres-solver.org/installation.html), and enable `WITH_CERES` in CMake when installing OpenPose. Ceres is harder to install in Windows, so we have not tested it so far in there. Feel free to make a pull request if you do.

After installation, check the [doc/advanced/3d_reconstruction_module.md](../advanced/3d_reconstruction_module.md) instructions.



### Calibration Module
The instrinsic camera calibration toolbox is included by default.

To enable the extrinsic camera parameter estimation toolbox, you must also enable `WITH_EIGEN` in CMake during [CMake Configuration](#cmake-configuration). You can perform any of the 3 following options (but only 1 of them!)
- Recommended: Simply set the `WITH_EIGEN` flag to `AUTOBUILD`. CMake will automatically download Eigen and configure OpenPose to use it. If you prefer to download it manually (or if your firewall blocks CMake from downloading it):
    - [Eigen 3.3.8](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/eigen_2020_11_18.zip): Unzip it as `3rdparty/eigen/`.
- Advanced (not recommended): If you set `WITH_EIGEN` to `FIND`, you must have Eigen already installed in your system. Note that [Eigen <= 3.3.6 is not supported by CUDA >=9.1](https://bitbucket.org/eigen/eigen/commits/034b6c3e101792a3cc3ccabd9bfaddcabe85bb58?at=default). In order to install it (make sure that Eigen version is compatible with CUDA!):
    - Run `sudo apt-get install libeigen3-dev` and link CMake to the right CMake.
- Advanced (not recommended): Or you could also use your own version of Eigen by setting `WITH_EIGEN` to `AUTOBUILD`, click `Configure` to let CMake download the zip file, and replace `3rdparty/eigen/` by your own version.

After installation, check the [doc/advanced/calibration_module.md](../advanced/calibration_module.md) instructions.



### Unity Compatible Version
Check [**Unity Plugin**](https://github.com/CMU-Perceptual-Computing-Lab/openpose_unity_plugin).

However, the OpenPose Unity version will crash if if faces an error while it is not used inside Unity. Thus, do not use it without Unity. Although this version would work as long as no errors occur.



### Compiling without cuDNN
The [cuDNN](https://developer.nvidia.com/cudnn) library is not mandatory, but required for full keypoint detection accuracy. In case your graphics card is not compatible with cuDNN, you can disable it by unchecking `USE_CUDNN` in CMake.

Then, you would have to reduce the `--net_resolution` flag to fit the model into the GPU memory. You can try values like `640x320`, `320x240`, `320x160`, or `160x80` to see your GPU memory capabilities. After finding the maximum approximate resolution that your GPU can handle without throwing an out-of-memory error, adjust the `net_resolution` ratio to your image or video to be processed (see the `--net_resolution` explanation from [doc/advanced/demo_advanced.md](../advanced/demo_advanced.md)), or use `-1` (e.g., `--net_resolution -1x320`).



### Custom Caffe
OpenPose uses a [custom fork of Caffe](https://github.com/CMU-Perceptual-Computing-Lab/caffe) (rather than the official Caffe master). Our custom fork is only updated if it works on our machines, but we try to keep it updated with the latest Caffe version. This version works on a newly formatted machine (Ubuntu 16.04 LTS) and in all our machines (CUDA 8 and 10 tested). The default GPU version is the master branch, which it is also compatible with CUDA 10 without changes (official Caffe version might require some changes for it). We also use the OpenCL and CPU tags if their CMake flags are selected. We only modified some Caffe compilation flags and minor details.

Alternatively, you can use your own Caffe distribution on Ubuntu/Mac by 1) disabling `BUILD_CAFFE`, 2) setting `Caffe_INCLUDE_DIRS` to `{CAFFE_PATH}/include/caffe`, and 3) setting `Caffe_LIBS` to `{CAFFE_PATH}/build/lib/libcaffe.so`, as shown in the image below. Note that cuDNN-compatible Caffe version is required in order to get the maximum possible accuracy in OpenPose.
<p align="center">
    <img src="../../.github/media/installation/cmake_im_5.png" width="480">
</p>

For Windows, simply replace the OpenCV DLLs and include folder for your custom one.



### Custom NVIDIA NVCaffe
This functionality was added by the community, and we do not officially support it. New pull requests with additional functionality or fixing any bug are welcome!

It has been tested with the official Nvidia Docker image [nvcr.io/nvidia/caffe:18.12-py2](https://ngc.nvidia.com/catalog/containers/nvidia:caffe).

For questions and issues, please only post on the related [Pull Request #1169](https://github.com/CMU-Perceptual-Computing-Lab/openpose/pull/1169). New GitHub issues about this topic (i.e., outside PR #1169) will be automatically closed with no answer.

Windows support has not been added. Replace `set_property(CACHE DL_FRAMEWORK PROPERTY STRINGS CAFFE)` by `set_property(CACHE DL_FRAMEWORK PROPERTY STRINGS CAFFE NV_CAFFE)` in `CMakeLists.txt` if you intend to use it for Windows, and feel free to do a pull request of it working!

To use a NVIDIA's NVCaffe docker image instead of the standard Caffe, set the following CMake flags:

1. Set the `DL_FRAMEWORK` variable to `NV_CAFFE`.
2. Set the `BUILD_CAFFE` variable to `OFF`.
3. Set the correct `Caffe_INCLUDE_DIRS` and `Caffe_LIBS` paths following [Custom Caffe](#custom-caffe).

In addition, [peter-uhrig.de/openpose-with-nvcaffe-in-a-singularity-container-with-support-for-multiple-architectures/](http://peter-uhrig.de/openpose-with-nvcaffe-in-a-singularity-container-with-support-for-multiple-architectures/) contains a detailed step-by-step guide to install a portable container with NVCaffe and support for multiple NVidia cards as well as CPU.



### Custom OpenCV
If you have built OpenCV from source and OpenPose cannot find it automatically, you can set the `OPENCV_DIR` variable to the directory where you build OpenCV (Ubuntu and Mac). For Windows, simply replace the OpenCV DLLs and include folder for your custom one.



### Doxygen Documentation Autogeneration (Ubuntu Only)
You can generate the documentation by setting the `BUILD_DOCS` flag. The documentation will be generated in `doc/doxygen/html/index.html`. You can simply open it with double-click (your default browser should automatically display it).



### CMake Command Line Configuration (Ubuntu Only)
Note that this step is unnecessary if you already used the CMake GUI alternative.

Create a `build` folder in the root OpenPose folder, where you will build the library --
```bash
cd openpose
mkdir build
cd build
```

The next step is to generate the Makefiles. Now there can be multiple scenarios based on what the user already has e.x. Caffe might be already installed and the user might be interested in building OpenPose against that version of Caffe instead of requiring OpenPose to build Caffe from scratch.

#### Scenario 1 - Caffe not installed and OpenCV installed using `apt-get`
In the build directory, run the below command --
```bash
cmake ..
```

#### Scenario 2 - Caffe installed and OpenCV build from source
In this example, we assume that Caffe and OpenCV are already present. The user needs to supply the paths of the libraries and the include directories to CMake. For OpenCV, specify the include directories and the libraries directory using `OpenCV_INCLUDE_DIRS` and `OpenCV_LIBS_DIR` variables respectively. Alternatively, the user can also specify the path to the `OpenCVConfig.cmake` file by setting the `OpenCV_CONFIG_FILE` variable. For Caffe, specify the include directory and library using the `Caffe_INCLUDE_DIRS` and `Caffe_LIBS` variables. This will be where you installed Caffe. Below is an example of the same.
```bash
cmake -DOpenCV_INCLUDE_DIRS=/home/"${USER}"/softwares/opencv/build/install/include \
  -DOpenCV_LIBS_DIR=/home/"${USER}"/softwares/opencv/build/install/lib \
  -DCaffe_INCLUDE_DIRS=/home/"${USER}"/softwares/caffe/build/install/include \
  -DCaffe_LIBS=/home/"${USER}"/softwares/caffe/build/install/lib/libcaffe.so -DBUILD_CAFFE=OFF ..
```

```bash
cmake -DOpenCV_CONFIG_FILE=/home/"${USER}"/softwares/opencv/build/install/share/OpenCV/OpenCVConfig.cmake \
  -DCaffe_INCLUDE_DIRS=/home/"${USER}"/softwares/caffe/build/install/include \
  -DCaffe_LIBS=/home/"${USER}"/softwares/caffe/build/install/lib/libcaffe.so -DBUILD_CAFFE=OFF ..
```

#### Scenario 3 - OpenCV already installed
If Caffe is not already present but OpenCV is, then use the below command.
```bash
cmake -DOpenCV_INCLUDE_DIRS=/home/"${USER}"/softwares/opencv/build/install/include \
  -DOpenCV_LIBS_DIR=/home/"${USER}"/softwares/opencv/build/install/lib ..
```

```bash
cmake -DOpenCV_CONFIG_FILE=/home/"${USER}"/softwares/opencv/build/install/share/OpenCV/OpenCVConfig.cmake ..
```

#### Any Other Scenario
You can check the CMake online documentation to check all the options that CMake provides and its analogs to the CMake-gui ones that we show on this document.
