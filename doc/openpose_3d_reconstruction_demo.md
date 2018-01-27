OpenPose 3-D Reconstruction Module and Demo
=============================================

This experimental module performs 3-D keypoint (body, face, and hand) reconstruction and rendering for 1 person. We will not keep updating it nor solving questions/issues about it at the moment. It requires the user to be familiar with computer vision and camera calibration, including extraction of intrinsic and extrinsic parameters.



## Features
- Auto detection of all FLIR cameras connected to your machine, and image streaming from all of them.
- Hardware trigger and buffer `NewestFirstOverwrite` modes enabled. Hence, the algorithm will always get the last synchronized frame from each camera, deleting the rest.
- 3-D reconstruction of body, face, and hands for 1 person.
- If more than 1 person is detected per camera, the algorithm will just try to match person 0 on each camera, which will potentially correspond to different people in the scene. Thus, the 3-D reconstruction will completely fail.
- Only points with high threshold with respect to each one of the cameras are reprojected (and later rendered). An alternative for > 4 cameras could potentially do 3-D reprojection and render all points with good views in more than N different cameras (not implemented here).
- Only Direct linear transformation (DLT) is applied for reconstruction. Non-linear optimization methods (e.g. from Ceres Solver) will potentially improve results (not implemented).
- Basic OpenGL rendering with the `freeglut` library.



## Required Hardware
This demo assumes n arbitrary stereo cameras from the FLIR company (formerly Point Grey). Ideally any USB-3 FLIR model should work, but we have only used the following specific specifications:

1. Camera details:
    - Blackfly S Color 1.3 MP USB3 Vision (ON Semi PYTHON 1300)
    - Model: BFS-U3-13Y3C-C
    - 1280x1024 resolution and 170 FPS
    - [https://www.ptgrey.com/blackfly-s-13-mp-color-usb3-vision-on-semi-python1300](https://www.ptgrey.com/blackfly-s-13-mp-color-usb3-vision-on-semi-python1300)
    - Hardware trigger synchronization required. For this camera model, see `Blackfly S` section in [https://www.ptgrey.com/tan/11052](https://www.ptgrey.com/tan/11052) or [https://www.ptgrey.com/KB/11052](https://www.ptgrey.com/KB/11052).
    - (Ubuntu-only) Open your USB ports following section `Configuring USBFS` in [http://www.ptgrey.com/KB/10685](http://www.ptgrey.com/KB/10685).
    - Install the Spinnaker SDK for your operating system: [https://www.ptgrey.com/support/downloads](https://www.ptgrey.com/support/downloads).
2. Fujinon 3 MP Varifocal Lens (3.8-13mm, 3.4x Zoom) for each camera.
    - E.g. [https://www.bhphotovideo.com/c/product/736855-REG/Fujinon_DV3_4X3_8SA_1_3_MP_Varifocal_Lens.html](https://www.bhphotovideo.com/c/product/736855-REG/Fujinon_DV3_4X3_8SA_1_3_MP_Varifocal_Lens.html).
3. 4-Port PCI Express (PCIe) USB 3.0 Card Adapter with 4 dedicated channels.
    - E.g. [https://www.startech.com/Cards-Adapters/USB-3.0/Cards/PCI-Express-USB-3-Card-4-Dedicated-Channels-4-Port~PEXUSB3S44V](https://www.startech.com/Cards-Adapters/USB-3.0/Cards/PCI-Express-USB-3-Card-4-Dedicated-Channels-4-Port~PEXUSB3S44V).
4. USB 3.0 cable for each FLIR camera.
    - From their official website: [https://www.ptgrey.com/5-meter-type-a-to-micro-b-locking-usb-30-cable](https://www.ptgrey.com/5-meter-type-a-to-micro-b-locking-usb-30-cable).



## Camera Calibration
The user must manually get the intrinsic and extrinsic parameters of the cameras, introduce them on: `src/experimental/3d/cameraParameters.cpp`, and recompile OpenPose.

The program uses 3 cameras by default, but cameras can be added or removed from `cameraParameters.hpp` by adding or removing elements to `INTRINSICS`, `DISTORTIONS` and `M_EACH_CAMERA`. `INTRINSICS` corresponds to the intrinsic parameters, `DISTORTIONS` to the distortion coefficients, and `M` corresponds to the extrinsic parameters of the cameras with respect to camera 1, i.e. camera 1 is considered the coordinates origin. However, it can be changed by the user by setting these parameters differently.

3D OpenPose uses the 8-distortion-parameter version of OpenCV by default. Internally, the distortion parameters are used by the OpenCV function [undistort()](http://docs.opencv.org/3.2.0/da/d54/group__imgproc__transform.html#ga69f2545a8b62a6b0fc2ee060dc30559d) to rectify the images. This function can take either 4-, 5- or 8-parameter distortion coefficients (OpenCV 3.X also adds a 12- and 14-parameter alternatives). Therefore, either version (4, 5, 8, 12 or 14) will work in 3D OpenPose, the user just needs to modify the `DISTORTION_i` variables in `cameraParameters.hpp` with the desired number of distortion parameters for each camera.



## Camera Ordering
In order to verify that the camera parameters introduced by the user are sorted in the same way that OpenPose reads the cameras, make sure of the following points:

1. Initially, introduce the camera parameters sorted by serial number. By default (in Spinnaker 1.8), they are sorted by serial number.
2. When the program is run, OpenPose displays the camera serial number associated to each index of each detected camera. If the number of cameras detected is different to the number of actual cameras, make sure the hardware is properly connected and the camera leds are on.
3. Make sure that the order in which you introduced your camera parameters matches this index ordering displayed by OpenPose. Again, it should be sorted by serial number, but different Spinnaker versions might work differently.




## Installing the OpenPose 3-D Reconstruction Module
Check the [doc/installation_cmake.md#openpose-3d-reconstruction-module-and-demo](./installation_cmake.md#openpose-3d-reconstruction-module-and-demo) instructions.



## Expected Visual Results
The visual GUI should show 3 screens.

1. The Windows command line or Ubuntu bash terminal.
2. The different cameras 2-D keypoint estimations.
3. The final 3-D reconstruction.

It should be similar to the following image.

<p align="center">
    <img src="media/openpose3d.png">
</p>
