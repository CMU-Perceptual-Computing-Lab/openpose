OpenPose Calibration Module and Demo
=============================================

## Contents
1. [Introduction](#introduction)
2. [Installing the Calibration Module](#installing-the-calibration-module)
3. [Running Calibration](#running-calibration)
    1. [General Quality Tips](#general-quality-tips)
    2. [Step 1 - Distortion and Intrinsic Parameter Calibration](#step-1---distortion-and-intrinsic-parameter-calibration)
    3. [Step 2 - Extrinsic Parameter Calibration](#step-2---extrinsic-parameter-calibration)
4. [Using a Different Camera Brand](#using-a-different-camera-brand)



## Introduction
This experimental module performs camera calibration (distortion, intrinsic, and extrinsic camera parameter extraction). It computes and saves the intrinsics parameters of the input images. It is built on top of OpenCV, but aimed to simplify the process for people with no calibration or computer vision background at all (or for lazy people like myself).

Note: We are not aiming to have the best calibration toolbox, but the simplest one. If very high quality calibration is required, I am sure there must exist many other toolboxs with better extrinsic parameter estimation tools.



## Installing the Calibration Module
Check [doc/installation.md#calibration-module](./installation.md#calibration-module) for installation steps.



## Running Calibration
Note: In order to maximize calibration quality, **do not reuse the same video sequence for both intrinsic and extrinsic parameter estimation**. The intrinsic parameter calibration should be run camera by camera, where each recorded video sequence should be focused in covering all regions of the camera view and repeated from several distances. In the extrinsic sequence, this video sequence should be focused in making sure that the checkboard is visible from at least 2 cameras at the time. So for 3-camera calibration, you would need 1 video sequence per camera as well as a final sequence for the extrinsic parameter calibration.

### General Quality Tips
1. Keep the same orientation of the chessboard, i.e., do not rotate it circularly more than ~15-30 degress with respect to its center (i.e., going from a `w` x `h` number of squares to a `h` x `w` one). Our algorithm assumes that the origin is the corner at the top left, so rotating the chessboard circularly will change this origin across frames, resulting in many frames being rejected for the final calibration, i.e., lower calibration accuracy.
2. Cover several distances, and within each distance, cover all parts of the image view (all corners and center).
3. Save the images in PNG format (default behavior) in order to improve calibration quality. PNG images are bigger than JPG equivalent, but do not lose information by compression.
4. Use a chessboard as big as possible, ideally a chessboard with of at least 8x6 squares with a square size of at least 100 millimeters. It will specially affect the extrinsic calibration quality.
5. Intrinsics: Recommended about 400 image views for high quality calibration. You should get at least 150 images for a good calibration, while no more than 500. The calibration of a camera takes about 3 minutes with about 100 images, about 1.5h with 200 images, and about 9.5h with 450 images. Required RAM memory also grows exponentially.
6. Extrinsics: Recommended at least 250 images per camera for high quality calibration.

### Step 1 - Distortion and Intrinsic Parameter Calibration
1. Run OpenPose and save images for your desired camera. Use a grid (chessboard) pattern and move around all the image area. Depending on the images source:
    1. Webcam calibration: `./build/examples/openpose/openpose.bin --num_gpu 0 --write_images {intrinsic_images_folder_path}`.
    2. Flir camera calibration: Add the flags `--flir_camera --flir_camera_index 0` (or the desired flir camera index) to the webcam command.
    3. Calibration from video sequence: Add the flag `--video {video_path}` to the webcam command.
    4. Any other camera brand: Simply save your images in `{intrinsic_images_folder_path}`, file names are not relevant.
2. Get familiar with the calibration parameters used in point 3 (i.e., `grid_square_size_mm`, `grid_number_inner_corners`, etc.) by running the `--help` flag:
```sh
./build/examples/calibration/calibration.bin --help
```
3. Extract and save the intrinsic parameters:
```sh
./build/examples/calibration/calibration.bin --mode 1 --grid_square_size_mm 40.0 --grid_number_inner_corners "9x5" --camera_serial_number 18079958 --calibration_image_dir {intrinsic_images_folder_path}
```
4. In this case, the intrinsic parameters would have been generated as `{intrinsic_images_folder_path}/18079958.xml`.
5. Run steps 1-4 for each one of your cameras.
6. After you calibrate the camera intrinsics, when you run OpenPose with those cameras, you should see the lines in real-life to be (almost) perfect lines in the image. Otherwise, the calibration was not good. Try checking straight patterns such us wall or ceiling edges:
```sh
# With distortion (straight lines might not look as straight lines but rather with a more circular shape)
./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera --flir_camera_index 0
# Without distortion (straight lines should look as straight lines)
./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera --flir_camera_index 0 --frame_undistort
```

Examples:

1. Full example for a folder of images, a video, webcam streaming, etc.:
```sh
# Ubuntu and Mac
# Get images for calibration (only if target is not `--image_dir`)
    # If video
./build/examples/openpose/openpose.bin --num_gpu 0 --video examples/media/video_chessboard.avi --write_images ~/Desktop/Calib_intrinsics
    # If webcam
./build/examples/openpose/openpose.bin --num_gpu 0 --webcam --write_images ~/Desktop/Calib_intrinsics
# Run calibration
./build/examples/calibration/calibration.bin --mode 1 --grid_square_size_mm 30.0 --grid_number_inner_corners "8x6" --calibration_image_dir ~/Desktop/Calib_intrinsics/ --camera_parameter_folder models/cameraParameters/ --camera_serial_number frame_intrinsics
# Output: {OpenPose path}/models/cameraParameters/frame_intrinsics.xml
# Visualize undistorted images
./build/examples/openpose/openpose.bin --num_gpu 0 --image_dir ~/Desktop/Calib_intrinsics/ --frame_undistort --camera_parameter_path "models/cameraParameters/frame_intrinsics.xml"
# If video
./build/examples/openpose/openpose.bin --num_gpu 0 --video examples/media/video_chessboard.avi --frame_undistort --camera_parameter_path "models/cameraParameters/frame_intrinsics.xml"
# If webcam
./build/examples/openpose/openpose.bin --num_gpu 0 --webcam --frame_undistort --camera_parameter_path "models/cameraParameters/frame_intrinsics.xml"
```

2. Full example for 4-view Flir/Point Grey camera system:
```sh
# Ubuntu and Mac
# Get images for calibration
./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera --flir_camera_index 0 --write_images ~/Desktop/intrinsics_0
./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera --flir_camera_index 1 --write_images ~/Desktop/intrinsics_1
./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera --flir_camera_index 2 --write_images ~/Desktop/intrinsics_2
./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera --flir_camera_index 3 --write_images ~/Desktop/intrinsics_3
# Run calibration
#     - Note: If your computer has enough RAM memory, you can run all of them at the same time in order to speed up the time (they are not internally multi-threaded).
./build/examples/calibration/calibration.bin --mode 1 --grid_square_size_mm 127.0 --grid_number_inner_corners "9x6" --camera_serial_number 17012332 --calibration_image_dir ~/Desktop/intrinsics_0
./build/examples/calibration/calibration.bin --mode 1 --grid_square_size_mm 127.0 --grid_number_inner_corners "9x6" --camera_serial_number 17092861 --calibration_image_dir ~/Desktop/intrinsics_1
./build/examples/calibration/calibration.bin --mode 1 --grid_square_size_mm 127.0 --grid_number_inner_corners "9x6" --camera_serial_number 17092865 --calibration_image_dir ~/Desktop/intrinsics_2
./build/examples/calibration/calibration.bin --mode 1 --grid_square_size_mm 127.0 --grid_number_inner_corners "9x6" --camera_serial_number 18079957 --calibration_image_dir ~/Desktop/intrinsics_3
# Visualize undistorted images
#     - Camera parameters will be saved on their respective serial number files, so OpenPose will automatically find them
./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera --frame_undistort
```

3. For Windows, simply run `build\x64\Release\calibration.exe` (or the one from the binary portable demo) with the same flags as above.



### Step 2 - Extrinsic Parameter Calibration
1. **VERY IMPORTANT NOTE**: If you want to re-run the extrinsic parameter calibration over the same intrinsic XML files (e.g., if you move the camera location, but you know the instrinsics are the same), you must manually re-set to `1 0 0 0  0 1 0 0  0 0 1 0` the camera matrix of each XML file.
2. After intrinsics calibration, save undirtoted images for all the camera views:
```sh
./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera --write_images ~/Desktop/extrinsics
```
3. Run the extrinsic calibration tool between each pair of close cameras. In this example:
	- We assume camera 0 to the right, 1 in the middle-right, 2 in the middle-left, and 3 in the left.
	- We assume camera 1 as the coordinate origin.
```sh
# Ubuntu and Mac
./build/examples/calibration/calibration.bin --mode 2 --grid_square_size_mm 127.0 --grid_number_inner_corners 9x6 --omit_distortion --calibration_image_dir ~/Desktop/extrinsics/ --cam0 1 --cam1 0
./build/examples/calibration/calibration.bin --mode 2 --grid_square_size_mm 127.0 --grid_number_inner_corners 9x6 --omit_distortion --calibration_image_dir ~/Desktop/extrinsics/ --cam0 1 --cam1 2
./build/examples/calibration/calibration.bin --mode 2 --grid_square_size_mm 127.0 --grid_number_inner_corners 9x6 --omit_distortion --calibration_image_dir ~/Desktop/extrinsics/ --cam0 1 --cam1 3
# Potentially more accurate equivalent for the calibration between cameras 1 and 3: If camera 3 and 1 are too far from each other and the calibration chessboard is not visible from both cameras at the same time enough times, the calibration can be run between camera 3 and camera 2, which is closer to 3. In that case, the `combine_cam0_extrinsics` flag is required, which tells the calibration toolbox that cam0 is not the global origin (in this case is camera 1).
# Note: Wait until calibration of camera index 2 with respect to 1 is completed, as information from camera 2 XML calibration file will be used:
./build/examples/calibration/calibration.bin --mode 2 --grid_square_size_mm 127.0 --grid_number_inner_corners 9x6 --omit_distortion --calibration_image_dir ~/Desktop/extrinsics/ --cam0 2 --cam1 3 --combine_cam0_extrinsics
```
```
:: Windows
:: build\x64\Release\calibration.exe with the same flags as above
```
4. Hint to verify extrinsic calibration is successful:
    1. Translation vector - Global distance:
        1. Manually open each one of the generated XML files from the folder indicated by the flag `--camera_parameter_path` (or the default one indicated by the `--help` flag if the former was not used).
        2. The field `CameraMatrix` is a 3 x 4 matrix (you can see that the subfield `rows` in that file is 3 and `cols` is 4).
        3. Order the matrix in that 3 x 4 shape (e.g., by copying in a different text file with the shape of 3 rows and 4 columns).
        4. The 3 first components of the last column of the `CameraMatrix` field define the global `translation` (in meters) with respect to the global origin (in our case camera 1).
        5. Thus, the distance between that camera and the origin camera 1 should be (approximately) equal to the L2-norm of the `translation` vector.
    2. Translation vector - Relative x-y-z distances:
        1. The 3x1 `translation` vector represents the `x`, `y`, and `z` distances to the origin camera, respectively. The camera is looking along the positive `z` axis, the `y` axis is down, and the `x` axis is right. This should match the real distance between both cameras.



## Using a Different Camera Brand
If you plan to use the calibration tool without using OpenPose, you can manually save a video sequence of your desired camera into each of the camera image folders (i.e., in the above example, the `~/Desktop/intrinsics_0`, `~/Desktop/intrinsics_1`, etc. folders).

If you wanna eventually run that camera with OpenPose, check [doc/modules/3d_reconstruction_module.md#using-a-different-camera-brand](./modules/3d_reconstruction_module.md#using-a-different-camera-brand).
