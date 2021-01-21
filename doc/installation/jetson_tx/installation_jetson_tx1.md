OpenPose Doc - Installation on Nvidia Jetson TX1
====================================
## Introduction
We do not officially support TX1, but thanks to @dreinsdo, we have these instructions about how he made it work in his TX1. We would like to thank @dreinsdo, who added this documentation in [this GitHub issue post](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1124#issuecomment-474090671). If you face any issue, feel free to post on that issue post.

## Purpose
This document describes the full procedure for installing openpose on the Jetson TX1. Other, and less involved, procedures may have been found successful by community members, and we encourage you to share your alternatives.


## Preliminary remarks
This procedure details moving the Jetson file system to a larger drive, building a custom kernel, building OpenCV from source, and customizing Openpose Makefiles because TX1 eMMC is limited, the onboard camera did not work with Openpose, stock Jetpack 3.1 OpenCV build lacks Openpose dependencies, and Openpose makefiles are not compatible with TX1 CUDA arch, respectively. We used a PS3 Eye camera in place of the onboard camera, and a 120Gb SSD, but most USB webcams and SATA drives should work fine.


## Contents
- [Prep the TX1](#prep-the-tx1)
- [Build custom kernel](#build-custom-kernel)
- [Build OpenCV from source](#build-opencv-from-source)
- [Install Openpose](#install-openpose)
- [Usage](#usage)


## Prep the TX1
1. Flash Jetson TX1 with [JetPack 3.1](https://developer.nvidia.com/embedded/jetpack) per Jetpack installation guide. Be sure to complete both OS flashing and CUDA / cuDNN installation parts before installation.
2. Move file system to SATA drive. Follow steps of JetsonHacks article [Install Samsung SSD on NVIDIA Jetson TX1](https://www.jetsonhacks.com/2017/01/28/install-samsung-ssd-on-nvidia-jetson-tx1/).


## Build custom kernel
This step is required because we were not able to use Openpose with the onboard TX1 camera. The steps are a combination of two JetsonHacks articles [Build Kernel and ttyACM Module – NVIDIA Jetson TX1](https://www.jetsonhacks.com/2017/08/07/build-kernel-ttyacm-module-nvidia-jetson-tx1/) and [Sony PlayStation Eye – NVIDIA Jetson TX1](https://www.jetsonhacks.com/2016/09/29/sony-playstation-eye-nvidia-jetson-tx1/). If you are using a different webcam then include the driver for that webcam in place of the PS3 eye driver in step 3.
1. Get the install scripts from [JetsonHacks Github](https://github.com/jetsonhacks/buildJetsonTX1Kernel/archive/v1.0-L4T28.1.zip). This link is to the zip of the 'JetPack 3.1' release. If you $git clone$ from the master branch then you will get the most recent kernel build files, which are not compatible with JetPack 3.1.
2. Unzip the downloaded files, enter the unzipped directory and run script to get kernel sources.
```
$ cd buildJetsonTX1Kernel
$ sudo ./getKernelSources.sh
```
3. The script will open the editor for the kernel configuration. Find the driver for your webcam and select with a checkbox (not a dot). Save the configuration and quit the config window.
4. Make the kernel.
```
$ sudo ./makeKernel.sh
```
5. Replace the current kernel the newly built kernel image.
```
$ sudo cp /usr/src/kernel/kernel-4.4/arch/arm64/boot/Image ($PATH_TO_EMMC)/boot/Image
```
Replace $PATH_TO_EMMC with the path to your eMMC. This is required because the Jetson initially boots to eMMC and loads the kernel from their, even with the SATA drive connected.


## Build OpenCV from source
Follow JK Jung's steps from [How to Install OpenCV (3.4.0) on Jetson TX2](https://jkjung-avt.github.io/opencv3-on-tx2/) verbatim, with the following exception: omit installation of Python3 dependencies, i.e. skip the following lines.
```
$ sudo apt-get install python3-dev python3-pip python3-tk
$ sudo pip3 install numpy
$ sudo pip3 install matplotlib
```


## Install Openpose
The following steps detail the modification of three files to install Openpose. Modified versions of the files are attached and may alternatively be used. To use, be sure to rename both makefile configs to `Makefile.config.Ubuntu16_cuda8_JetsonTX2`. 
1. Clone from the master branch.
```
$ git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
$ cd openpose
```
2. Modify makefile config. For the installation procedure we will use the TX2 files for JetPack 3.1. 
```
$ gedit scripts/ubuntu/Makefile.config.Ubuntu16_cuda8_JetsonTX2
```
Uncomment the opencv line
```
OPENCV_VERSION := 3
```
Replace all the 'CUDA_ARCH :=' lines with the following
```
CUDA_ARCH := -gencode arch=compute_53,code=[sm_53,compute_53]
```
Add CUDNN - not sure if this is necessary, have not retried the install without it.
```
USE_CUDNN := 1
```
3. Correct error in install script path.
```
$ gedit scripts/ubuntu/install_caffe_and_openpose_JetsonTX2_JetPack3.1.sh
```
Replace 
```
executeShInItsFolder "install_openpose_JetsonTX2_JetPack3.1.sh" "./scripts/ubuntu/" "./"
```
with
```
executeShInItsFolder "./scripts/ubuntu/install_openpose_JetsonTX2_JetPack3.1.sh" "./" "./"
```
4. Start the install process. When you initially call the install script the caffe repo will be cloned and associated files downloaded. As soon as Caffe starts compiling, halt the process and change the makefile config as in step 2.
```
bash ./scripts/ubuntu/install_caffe_and_openpose_JetsonTX2_JetPack3.1.sh
```
Once caffe begins to compile `CTRL+C`.
```
$ gedit 3rdparty/caffe/Makefile.config.Ubuntu16_cuda8_JetsonTX2
```
Make the same changes as in step 2. The CUDNN switch should already be on.

5. Restart the installation process.
```
bash ./scripts/ubuntu/install_caffe_and_openpose_JetsonTX2_JetPack3.1.sh
```

## Usage
To get to decent FPS you need to lower the net resolution:
```
./build/examples/openpose/openpose.bin -camera_resolution 640x480 -net_resolution 128x96
```

To activate hand or face resolution please complete this command with the following options (warning, both simultaneously will cause out of memory error):
```
# Body and face
./build/examples/openpose/openpose.bin --face -face_net_resolution 256x256
# Body and hands
./build/examples/openpose/openpose.bin --hand -hand_net_resolution 256x256
# All body, face, and hands
./build/examples/openpose/openpose.bin --face -face_net_resolution 256x256 --hand -hand_net_resolution 256x256
```
