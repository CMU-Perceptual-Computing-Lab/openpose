OpenPose Calibration Module and Demo
=============================================

## Contents
1. [Introduction](#introduction)
2. [Installing the Calibration Module](#installing-the-calibration-module)
3. [Running Calibration](#running-calibration)
    1. [Step 1 - Distortion and Intrinsic Parameter Calibration](#step-1---distortion-and-intrinsic-parameter-calibration)
    2. [Step 2 - Extrinsic Parameter Calibration](#step-2---extrinsic-parameter-calibration)
4. [Using a Different Camera Brand](#using-a-different-camera-brand)



## Introduction
This experimental module performs camera calibration (distortion, intrinsic, and extrinsic camera parameter extraction). It computes and saves the intrinsics parameters of the input images. It is built on top of OpenCV, but aimed to simplify the process for people with no calibration or computer vision background at all (or for lazy people like myself).

Note: We are not aiming to have the best calibration toolbox, but the simplest one. If very high quality calibration is required, I am sure there must exist many other toolboxs with better extrinsic parameter estimation tools.



## Installing the Calibration Module
Check [doc/installation.md#calibration-module](./installation.md#calibration-module) for installation steps.



## Running Calibration
Note: In order to maximize calibration quality, **do not reuse the same video sequence for both intrinsic and extrinsic parameter estimation**. The intrinsic parameter calibration should be run camera by camera, where each recorded video sequence should be focused in covering all regions of the camera view and repeated from several distances. In the extrinsic sequence, this video sequence should be focused in making sure that the checkboard is visible from at least 2 cameras at the time. So for 3-camera calibration, you would need 1 video sequence per camera as well as a final sequence for the extrinsic parameter calibration.

### Step 1 - Distortion and Intrinsic Parameter Calibration
1. Run OpenPose and save images for your desired camera. Use a grid (chessboard) pattern and move around all the image area.
    1. Quality tips:
        1. Keep the same orientation of the chessboard, i.e., do not rotate it circularly more than ~15-30 degress. Our algorithm assumes that the origin is the corner at the top left, so rotating the chessboard circularly will change this origin across frames, resulting in many frames being rejected for the final calibration, i.e., lower calibration accuracy.
        2. Cover several distances, and within each distance, cover all parts of the image view (all corners and center).
        3. Save the images in PNG format (default behavior) in order to improve calibration quality. PNG images are bigger than JPG equivalent, but do not lose information by compression.
        4. Recommended about 400 image views for high quality calibration. You should get at least 150 images for a good calibration, while no more than 500. The calibration of a camera takes about 3 minutes with about 100 images, about 1.5h with 200 images, and about 9.5h with 450 images. Required RAM memory also grows exponentially.
    2. Changing image source:
        1. Webcam calibration: `./build/examples/openpose/openpose.bin --num_gpu 0 --frame_keep_distortion --write_images {intrinsic_images_folder_path}`.
        2. Flir camera calibration: Add the flags `--flir_camera --flir_camera_index 0` (or the desired flir camera index) to the webcam command.
        3. Calibration from video sequence: Add the flag `--video {video_path}` to the webcam command.
        4. Any other camera brand: Simply save your images in {intrinsic_images_folder_path}, file names are not relevant.
2. Get familiar with the calibration parameters used in point 3 (i.e., `grid_square_size_mm`, `grid_number_inner_corners`, etc.) by running the `--help` flag:
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
7. Full example for 4 Flir/Point Grey cameras:
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



### Step 2 - Extrinsic Parameter Calibration
1. We are still implementing this part. Documentation will be available after completing it.



## Using a Different Camera Brand
If you plan to use the calibration tool without using OpenPose, you can manually save a video sequence of your desired camera into each of the camera image folders (i.e., in the above example, the `~/Desktop/intrinsics_0`, `~/Desktop/intrinsics_1`, etc. folders).

If you wanna eventually run that camera with OpenPose, check [doc/3d_reconstruction_demo.md#using-a-different-camera-brand](./3d_reconstruction_demo.md#using-a-different-camera-brand).
