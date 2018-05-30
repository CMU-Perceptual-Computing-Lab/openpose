OpenPose Calibration Module and Demo
=============================================

## Contents
1. [Introduction](#introduction)
2. [Installing the Calibration Module](#installing-the-calibration-module)
3. [Quick Start](#quick-start)
4. [Using a Different Camera Brand](#using-a-different-camera-brand)



## Introduction
This experimental module performs camera calibration (distortion, intrinsic, and extrinsic camera parameter extraction). It computes and saves the intrinsics parameters of the input images. It is built on top of OpenCV, but aimed to simplify the process for people with no calibration or computer vision background at all (or for lazy people like myself).

Note: We are not aiming to have the best calibration toolbox, but the simplest one. If very high quality calibration is required, I am sure there must exist many other toolboxs with better extrinsic parameter estimation tools.



## Installing the OpenPose 3-D Reconstruction Module
Check [doc/installation.md#calibration-module](./installation.md#calibration-module) for installation steps.



## Quick Start
Note: This example will assume that the target are 3 Flir/Point Grey cameras, but it can be generalized to any camera model.

1. Distortion and intrinsic parameter calibration:
    1. Run OpenPose and save images for your desired camera. Use a grid (chessboard) pattern and move around all the image area.
        1. Quality tips:
            1. Keep the same orientation of the chessboard, i.e., do not rotate it circularly more than ~15-30 degress. Our algorithm assumes that the origin is the corner at the top left, so rotating the chessboard circularly will change this origin across frames, resulting in many frames being rejected for the final calibration, i.e., lower calibration accuracy.
            2. You should get at least 100-200 images for a good calibration (and no more than 500).
            3. Cover several distances, and within each distance, cover all parts of the image view (all corners and center).
            4. Save the images in PNG format (default behavior) in order to improve calibration quality. 
            5. The calibration of a camera takes about 3 minutes with about 100 images, about 1.5h with 200 images, and about 9.5h with 450 images. Required RAM memory also grows exponentially.
        2. Changing image source:
            1. Webcam calibration: `./build/examples/openpose/openpose.bin --num_gpu 0 --frame_keep_distortion --write_images {intrinsic_images_folder_path}`.
            2. Flir camera calibration: Add the flags `--flir_camera --flir_camera_index 0` (or the desired flir camera index) to the webcam command.
            3. Calibration from video sequence: Add the flag `--video {video_path}` to the webcam command.
            4. Any other camera brand: Simply save your images in {intrinsic_images_folder_path}, file names are not relevant.
    2. Get familiar with the calibration parameters used in point 3 (i.e., `grid_square_size_mm`, `grid_number_inner_corners`, etc.) by running:
```
./build/examples/calibration/calibration.bin --help
```
    3. Extract and save the intrinsic parameters:
```
./build/examples/calibration/calibration.bin --mode 1 --grid_square_size_mm 40.0 --grid_number_inner_corners "9x5" --camera_serial_number 18079958 --intrinsics_image_dir {intrinsic_images_folder_path}
```
    4. In this case, the intrinsic parameters would have been generated as {intrinsic_images_folder_path}/18079958.xml.
    5. Run steps 1-4 for each one of your cameras.
    6. After you calibrate the camera intrinsics, when you run OpenPose with those cameras, you should see the lines in real-life to be (almost) perfect lines in the image. Otherwise, the calibration was not good. Try checking straight patterns such us wall corners or ceilings:
```
# With distortion (lines might seem with a more circular shape)
./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera --flir_camera_index 0 --frame_keep_distortion
# Without distortion (lines should look as lines)
./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera --flir_camera_index 0
```
    7. Full example for 4 flir cameras:
```
# Get images for calibration
./build/examples/openpose/openpose.bin --num_gpu 0 --frame_keep_distortion --flir_camera --flir_camera_index 0 --write_images ~/Desktop/intrinsics_0
./build/examples/openpose/openpose.bin --num_gpu 0 --frame_keep_distortion --flir_camera --flir_camera_index 1 --write_images ~/Desktop/intrinsics_1
./build/examples/openpose/openpose.bin --num_gpu 0 --frame_keep_distortion --flir_camera --flir_camera_index 2 --write_images ~/Desktop/intrinsics_2
./build/examples/openpose/openpose.bin --num_gpu 0 --frame_keep_distortion --flir_camera --flir_camera_index 3 --write_images ~/Desktop/intrinsics_3
# Run calibration
#     - Note: If your computer has enough RAM memory, you can run all of them at the same time in order to speed up the time (they are not internally multi-threaded).
./build/examples/calibration/calibration.bin --mode 1 --grid_square_size_mm 127.0 --grid_number_inner_corners "9x6" --camera_serial_number 17012332 --intrinsics_image_dir ~/Desktop/intrinsics_0
./build/examples/calibration/calibration.bin --mode 1 --grid_square_size_mm 127.0 --grid_number_inner_corners "9x6" --camera_serial_number 17092861 --intrinsics_image_dir ~/Desktop/intrinsics_1
./build/examples/calibration/calibration.bin --mode 1 --grid_square_size_mm 127.0 --grid_number_inner_corners "9x6" --camera_serial_number 17092865 --intrinsics_image_dir ~/Desktop/intrinsics_2
./build/examples/calibration/calibration.bin --mode 1 --grid_square_size_mm 127.0 --grid_number_inner_corners "9x6" --camera_serial_number 18079957 --intrinsics_image_dir ~/Desktop/intrinsics_3
# Visualize undistorted images
#     - Camera parameters will be saved on their respective serial number files, so OpenPose will automatically find them
./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera
```

2. Extrinsic parameter calibration:
    1. We are still implementing this part. Documentation will be available after completing it.



## Using a Different Camera Brand
You can use any camera brand, check [doc/3d_reconstruction_demo.md#using-a-different-camera-brand](./3d_reconstruction_demo.md#using-a-different-camera-brand).
