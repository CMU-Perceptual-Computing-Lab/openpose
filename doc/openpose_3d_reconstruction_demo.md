# Running OpenPose 3-D Reconstruction Demo - Windows
This is a beta version that makes body pose 3-D reconstruction and rendering. We will not keep updating it nor solving questions/issues about it at the moment. It requires the user to be familiar with computer vision, in particular with camera calibration, i.e. extraction of intrinsic and extrinsic parameters.

The Windows steps were tested and worked in the OpenPose 1.0.1 version from the last GitHub commit on July 7th, 2017 in the [oficial repository](https://github.com/CMU-Perceptual-Computing-Lab/openpose).



### Hardware
This demo assumes 3 stereo cameras, FLIR company (former Point Grey). Ideally any USB-3 FLIR model should work, but we have only used the following specific specifications:

1. Camera details:
    - Blackfly S Color 1.3 MP USB3 Vision (ON Semi PYTHON 1300)
    - Model: BFS-U3-13Y3C-C
    - 1280x1024 resolution and 170 FPS
    - https://www.ptgrey.com/blackfly-s-13-mp-color-usb3-vision-on-semi-python1300
    - Hardware trigger synchronization required. For this camera model, see `Blackfly S` section in [https://www.ptgrey.com/tan/11052](https://www.ptgrey.com/tan/11052) or [https://www.ptgrey.com/KB/11052](https://www.ptgrey.com/KB/11052).
    - (Ubuntu-only) Open your USB ports following section `Configuring USBFS` in [http://www.ptgrey.com/KB/10685](http://www.ptgrey.com/KB/10685).
    - Install the Spinnaker SDK for your operating system: [https://www.ptgrey.com/support/downloads](https://www.ptgrey.com/support/downloads).
2. Fujinon 3 MP Varifocal Lens (3.8-13mm, 3.4x Zoom) for each camera.
    - E.g. [https://www.bhphotovideo.com/c/product/736855-REG/Fujinon_DV3_4X3_8SA_1_3_MP_Varifocal_Lens.html](https://www.bhphotovideo.com/c/product/736855-REG/Fujinon_DV3_4X3_8SA_1_3_MP_Varifocal_Lens.html).
3. 4-Port PCI Express (PCIe) USB 3.0 Card Adapter with 4 dedicated channels.
    - E.g. [https://www.startech.com/Cards-Adapters/USB-3.0/Cards/PCI-Express-USB-3-Card-4-Dedicated-Channels-4-Port~PEXUSB3S44V](https://www.startech.com/Cards-Adapters/USB-3.0/Cards/PCI-Express-USB-3-Card-4-Dedicated-Channels-4-Port~PEXUSB3S44V).
4. USB 3.0 cable for each FLIR camera.
    - From their official website: [https://www.ptgrey.com/5-meter-type-a-to-micro-b-locking-usb-30-cable](https://www.ptgrey.com/5-meter-type-a-to-micro-b-locking-usb-30-cable).



### Calibrate Cameras
You must manually get the intrinsic and extrinsic parameters of your cameras and introduce them on: `openpose3d\cameraParameters.hpp`. We used the 8-distortion-parameter version of OpenCV.



### Windows
1. Open the OpenPose visual studio solution `windows\openpose.sln`.
2. Right-click on `Solution 'OpenPose'` of the `Solution Explorer` window, usually placed at the top-right part of the VS screen.
3. Click on `Properties`. Go to `Configuration Properties` -> `Configuration` and check `Build` for the `OpenPose3DReconstruction` project.
4. Get the last Spinnaker SKD version, i.e. the FLIR camera driver and software:
    - Download last Spinnaker SDK: https://www.ptgrey.com/support/downloads
    - Copy `{PointGreyParentDirectory}\Point Grey Research\Spinnaker\bin64\vs2015\` as `{OpenPoseDirectory}\3rdparty\windows\spinnaker\bin\`. You can remove all the *.exe files.
    - Copy `{PointGreyParentDirectory}\Point Grey Research\Spinnaker\include\` as `{OpenPoseDirectory}\3rdparty\windows\spinnaker\include\`.
    - Copy `Spinnaker_v140.lib` and `Spinnakerd_v140.lib` from `{PointGreyParentDirectory}\Point Grey Research\Spinnaker\lib64\vs2015\` into `{OpenPoseDirectory}\3rdparty\windows\spinnaker\lib\`.
    - (Optional) Spinnaker SDK overview: https://www.ptgrey.com/spinnaker-sdk
5. Get the last OpenGL Glut library version for the rendering:
    - Download the latest `MSVC Package` from http://www.transmissionzero.co.uk/software/freeglut-devel/
    - Copy `{freeglutParentDirectory}\freeglut\bin\x64\` as `{OpenPoseDirectory}\3rdparty\windows\freeglut\bin\bin\`.
    - Copy `{freeglutParentDirectory}\freeglut\include\` as `{OpenPoseDirectory}\3rdparty\windows\freeglut\include\`.
    - Copy `{freeglutParentDirectory}\freeglut\lib\x64\` as `{OpenPoseDirectory}\3rdparty\windows\freeglut\lib\`.



### Ubuntu
We do not support Ubuntu at all for this demo. We did an original and very initial version long ago, but it was highly changed later. In case you need Ubuntu, these are the steps we used for our that original version in Ubuntu 16. Note that they might be several differences to make it work in the current version. Feel free to send us or make a pull request with the updated steps and we will update them, but we will not answer any kind of questions about it.

1. `sudo apt-get install freeglut3-dev`.
2. Compile OpenPose by your own [from https://github.com/CMU-Perceptual-Computing-Lab/openpose](from https://github.com/CMU-Perceptual-Computing-Lab/openpose).
3. Perform `make distribute` on OpenPose, and copy the `include` and `lib` files in [3rdparty/openpose/](3rdparty/openpose/).
4. Copy the `include` and `lib` folders from {OpenPose path}/3rdparty/caffe/distribute/ to [3rdparty/caffe/](3rdparty/caffe/).
5. Copy your Spinnaker desired version `include` and `lib` folders on [3rdparty/spinnaker/](3rdparty/spinnaker/).
6. Open the [rtstereo.pro](rtstereo.pro) file with Qt to have the project ready-to-compile-and-go. If you prefer using your own Makefile file, you can take a look to this Qt file to know which files (basically [src/](src/) and [include/](include/)) and compiler flags used.
	1. If using Qt, you will have to manually copy the {OpenPose path}/models folder inside the generated build folder.
7. You must copy the contents of [add_to_bin_file/](add_to_bin_file/) where the final binary file is generated.
