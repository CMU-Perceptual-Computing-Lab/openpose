OpenPose Demo - Quick Start
====================================

Forget about the OpenPose code, just download the portable Windows binaries (or compile the code from source) and use the demo by following this tutorial!

## Contents
1. [Mac OSX Additional Step](#mac-osx-additional-step)
2. [Quick Start](#quick-start)
    1. [Improving Memory and Speed but Decreasing Accuracy](#improving-memory-and-speed-but-decreasing-accuracy)
    2. [Running on Images, Video, or Webcam](#running-on-images-video-or-webcam)
    3. [Face and Hands](#face-and-hands)
    4. [Different Outputs (JSON, Images, Video, UI)](#different-outputs-json-images-video-ui)
    5. [Only Skeleton without Background Image](#only-skeleton-without-background-image)
    6. [Not Running All GPUs](#not-running-all-gpus)
    7. [Maximum Accuracy Configuration](#maximum-accuracy-configuration)
    8. [3-D Reconstruction](#3-d-reconstruction)
    9. [Tracking](#tracking)
    10. [Kinect 2.0 as Webcam on Windows 10](#kinect-20-as-webcam-on-windows-10)
    11. [Main Flags](#main-flags)
3. [Advanced Quick Start](#advanced-quick-start)





## Mac OSX Version Additional Step
If you are using a Mac and selected `CPU_ONLY`, you can skip this section.

If you are using a Mac and selected `OPENCL` support, and it has an inbuilt AMD graphics card, you have to manually select your AMD GPU. To do that, first note which device your Graphics card is set under. Most likely, your AMD device will be device 2.
```bash
clinfo
```

For any OpenPose command you run, add the following 2 flags to use your AMD card for acceleration (there `num_gpu_start` should be the number given above).
```bash
./build/examples/openpose/openpose.bin --num_gpu 1 --num_gpu_start 2
```

If you only have an integrated Intel Graphics card, then it will most probably be the device 1. Then, always add the following 2 flags to use your AMD card for acceleration.
```bash
./build/examples/openpose/openpose.bin --num_gpu 1 --num_gpu_start 1
```



## Quick Start
Check that OpenPose was properly installed by running any of the following 3 examples (image folder, video, or webcam). The expected visual result should look like [doc/output.md#expected-visual-results](output.md#expected-visual-results).

In Ubuntu, Mac, and other Unix systems, use any command-line interface, such as `Terminal` or `Terminator`. In Windows, open the `PowerShell`. You can do so with right-click on the Windows button, and `Windows PowerShell` (or pressing the Windows button + X, and then A). Feel free to watch any Youtube video tutorial if you are not familiar with these non-GUI tools.

Make sure that you are in the **root directory of the project** when running any command (i.e., in the OpenPose folder, not inside `build/` nor `windows/` nor `bin/`). In addition, `examples/media/video.avi` and `examples/media` already exist, so there is no need to change any lines of code on this tutorial. You can test OpenPose by running:
```
# Ubuntu and Mac
./build/examples/openpose/openpose.bin --video examples/media/video.avi
```
```
:: Windows - Portable Demo
bin\OpenPoseDemo.exe --video examples\media\video.avi
```

If these fail with an out of memory error, do not worry, the next example will fix this issue.



### Improving Memory and Speed but Decreasing Accuracy
**If you have a Nvidia GPU that does not goes out of memory when running, you should skip this step!**

**Use `net_resolution` at your own risk**: If your GPU runs out of memory or you do not have a Nvidia GPU, you can reduce `--net_resolution` to improve the speed and reduce the memory requirements, but it will also highly reduce accuracy! The lower the resolution, the lower accuracy but better speed/memory.
```
# Ubuntu and Mac
./build/examples/openpose/openpose.bin --video examples/media/video.avi --net_resolution -1x320
./build/examples/openpose/openpose.bin --video examples/media/video.avi --net_resolution -1x256
./build/examples/openpose/openpose.bin --video examples/media/video.avi --net_resolution -1x196
./build/examples/openpose/openpose.bin --video examples/media/video.avi --net_resolution -1x128
```
```
:: Windows - Portable Demo
bin\OpenPoseDemo.exe --video examples\media\video.avi --net_resolution -1x320
bin\OpenPoseDemo.exe --video examples\media\video.avi --net_resolution -1x256
bin\OpenPoseDemo.exe --video examples\media\video.avi --net_resolution -1x196
bin\OpenPoseDemo.exe --video examples\media\video.avi --net_resolution -1x128
```
```
:: Windows - Library - Assuming you copied the DLLs following doc/installation/README.md#windows
build\x64\Release\OpenPoseDemo.exe --video examples\media\video.avi --net_resolution -1x320
build\x64\Release\OpenPoseDemo.exe --video examples\media\video.avi --net_resolution -1x256
build\x64\Release\OpenPoseDemo.exe --video examples\media\video.avi --net_resolution -1x196
build\x64\Release\OpenPoseDemo.exe --video examples\media\video.avi --net_resolution -1x128
```

Additional notes:
- The default resolution is `-1x368`, any resolution smaller will improve speed.
- The `-1` means that that the resolution will be adapted to maintain the aspect ratio of the input source. E.g., `-1x368`, `656x-1`, and `656x368` will result in the same exact resolution for 720p and 1080p input images.
- For videos, using `-1` is recommended to let OpenPose find the ideal resolution. For a folder of images of different sizes, not adding `-1` and using images with completely different aspect ratios might result in out of memory issues. E.g., if a folder contains 2 images of resolution `100x11040` and `10000x368`. Then, using the default `-1x368` will result in the network output resolutions of `3x368` and `10000x368`, resulting in an obvious out of memory for the `10000x368` image.



### Running on Images, Video, or Webcam
- Directory with images (`--image_dir {DIRECTORY_PATH}`):
```
# Ubuntu and Mac
./build/examples/openpose/openpose.bin --image_dir examples/media/
# With face and hands
./build/examples/openpose/openpose.bin --image_dir examples/media/ --face --hand
```
```
:: Windows - Portable Demo
bin\OpenPoseDemo.exe --image_dir examples\media\
:: With face and hands
bin\OpenPoseDemo.exe --image_dir examples\media\ --face --hand
```
```
:: Windows - Library - Assuming you copied the DLLs following doc/installation/README.md#windows
build\x64\Release\OpenPoseDemo.exe --image_dir examples\media\
:: With face and hands
build\x64\Release\OpenPoseDemo.exe --image_dir examples\media\ --face --hand
```
- Video (`--video {VIDEO_PATH}`):
```
# Ubuntu and Mac
./build/examples/openpose/openpose.bin --video examples/media/video.avi
# With face and hands
./build/examples/openpose/openpose.bin --video examples/media/video.avi --face --hand
```
```
:: Windows - Portable Demo
bin\OpenPoseDemo.exe --video examples\media\video.avi
:: With face and hands
bin\OpenPoseDemo.exe --video examples\media\video.avi --face --hand
```
```
:: Windows - Library - Assuming you copied the DLLs following doc/installation/README.md#windows
build\x64\Release\OpenPoseDemo.exe --video examples\media\video.avi
:: With face and hands
build\x64\Release\OpenPoseDemo.exe --video examples\media\video.avi --face --hand
```
- Webcam is applied by default (i.e., if no `--image_dir` or `--video` flags used). Optionally, if you have more than 1 camera, you could use `--camera {CAMERA_NUMBER}` to select the right one:
```
# Ubuntu and Mac
./build/examples/openpose/openpose.bin
./build/examples/openpose/openpose.bin --camera 0
./build/examples/openpose/openpose.bin --camera 1
# With face and hands
./build/examples/openpose/openpose.bin --face --hand
```
```
:: Windows - Portable Demo
bin\OpenPoseDemo.exe
bin\OpenPoseDemo.exe --camera 0
bin\OpenPoseDemo.exe --camera 1
:: With face and hands
bin\OpenPoseDemo.exe --face --hand
```
```
:: Windows - Library - Assuming you copied the DLLs following doc/installation/README.md#windows
build\x64\Release\OpenPoseDemo.exe
build\x64\Release\OpenPoseDemo.exe --camera 0
build\x64\Release\OpenPoseDemo.exe --camera 1
:: With face and hands
build\x64\Release\OpenPoseDemo.exe --face --hand
```



### Face and Hands
Simply add `--face` and/or `--hand` to any command:
```
# Ubuntu and Mac
./build/examples/openpose/openpose.bin --image_dir examples\media\
./build/examples/openpose/openpose.bin --video examples\media\video.avi
./build/examples/openpose/openpose.bin
# With face and hands
./build/examples/openpose/openpose.bin --image_dir examples\media\ --face --hand
./build/examples/openpose/openpose.bin --video examples\media\video.avi --face --hand
./build/examples/openpose/openpose.bin --face --hand
```
```
:: Windows - Portable Demo
bin\OpenPoseDemo.exe --image_dir examples\media\
bin\OpenPoseDemo.exe --video examples\media\video.avi
bin\OpenPoseDemo.exe
:: With face and hands
bin\OpenPoseDemo.exe --image_dir examples\media\ --face --hand
bin\OpenPoseDemo.exe --video examples\media\video.avi --face --hand
bin\OpenPoseDemo.exe --face --hand
```



## Different Outputs (JSON, Images, Video, UI)
All the output options are complementary to each other. E.g., whether you display the images with the skeletons on the UI (or not) is independent on whether you save them on disk (or not).

- Save the skeletons in a set of JSON files with `--write_json {OUTPUT_VIDEO_PATH}`. Omitting the flag (default) means no JSON saving. See [doc/output.md](output.md) to understand the output format of the JSON files.
```
# Ubuntu and Mac (same flags for Windows)
./build/examples/openpose/openpose.bin --image_dir examples\media\ --write_json output_jsons/
./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_json output_jsons/
./build/examples/openpose/openpose.bin --write_json output_jsons/
```
- Save on disk the visual output of OpenPose (the images with the skeletons overlaid) as an output video (`--write_video {OUTPUT_VIDEO_PATH}`) or a set of images (`--write_images {OUTPUT_IMAGE_DIRECTORY_PATH}`, where `--write_images_format {FORMAT}` could also come handy):
```
# Ubuntu and Mac (same flags for Windows)
./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_video output/result.avi
./build/examples/openpose/openpose.bin --image_dir examples\media\ --write_video output/result.avi
./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_images output_images/ --write_images_format png
./build/examples/openpose/openpose.bin --image_dir examples\media\ --write_images output_images/ --write_images_format jpg
```
- You can also disable the UI visualization with `--display 0`. However, OpenPose will check and make sure your application is generating some kind of output. I.e., one out of `--write_json`, `--write_video`, or `--write_images` must be set if `--display 0`).
```
# Ubuntu and Mac (same flags for Windows)
./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_images output_images/ --display 0
```
- To speed up OpenPose even further when using `--display 0`, also add `--render_pose 0` if you are not using `--write_video` or `--write_images`. This way, OpenPose will not waste time overlaying skeletons with the input images.
```
# Ubuntu and Mac (same flags for Windows)
./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_json output_jsons/ --display 0 --render_pose 0
```



## Only Skeleton without Background Image
You can also visualize/save the skeleton without the original image overlaid or blended by adding `--disable_blending`:
```
# Ubuntu and Mac (same flags for Windows)
# Only body
./build/examples/openpose/openpose.bin --video examples/media/video.avi --disable_blending
```



## Not Running All GPUs
By default, OpenPose will use all the GPUs available in your machine. The following example runs the demo video `video.avi`, parallelizes it over 2 GPUs, GPUs 1 and 2 (note that it will skip GPU 0):
```
:: Windows - Portable Demo (same flags for Ubuntu and Mac)
bin\OpenPoseDemo.exe --video examples/media/video.avi --num_gpu 2 --num_gpu_start 1
```



### Maximum Accuracy Configuration
This command provides the most accurate results we have been able to achieve for body, hand and face keypoint detection.

However:
- This will only work on Nvidia GPUs with at least 8 GB of memory. It won't work on CPU or OpenCL settings. Your only option to maximize accuracy with those is to manually crop the people to fit the whole area of the image that is fed into OpenPose.
- It will need ~10.5 GB of GPU memory for the body-foot model (`BODY_25`) or ~6.7 GB for the `COCO` model.
- This requires GPUs like Titan X, Titan XP, some Quadro models, P100, V100, etc.
- Including hands and face will require >= 16GB GPUs (so the 12 GB GPUs like Titan X and XPs will no longer work).
- This command runs at ~2 FPS on a Titan X for the body-foot model (~1 FPS for COCO).
- Increasing `--net_resolution` will highly reduce the frame rate and increase latency, while it might increase the accuracy. However, this accuracy increase is not guaranteed in all scenarios, required a more detailed analysis for each particular scenario. E.g., it will work better for images with very small people, but usually worse for people taking a big ratio of the image. Thus, we recommend to follow the commands below for maximum accuracy in most cases for both big and small-size people.
- **Do not use this configuration for MPII model**, its accuracy might be harmed by this multi-scale setting. This configuration is optimal only for COCO and COCO-extended (e.g., the default BODY_25) models.

**Method overview:**
```
# Ubuntu and Mac: Body
./build/examples/openpose/openpose.bin --net_resolution "1312x736" --scale_number 4 --scale_gap 0.25
# Ubuntu and Mac: Body + Hand + Face
./build/examples/openpose/openpose.bin --net_resolution "1312x736" --scale_number 4 --scale_gap 0.25 --hand --hand_scale_number 6 --hand_scale_range 0.4 --face
```
```
:: Windows - Portable Demo: Body
bin\OpenPoseDemo.exe --net_resolution "1312x736" --scale_number 4 --scale_gap 0.25
:: Windows - Portable Demo: Body + Hand + Face
bin\OpenPoseDemo.exe --net_resolution "1312x736" --scale_number 4 --scale_gap 0.25 --hand --hand_scale_number 6 --hand_scale_range 0.4 --face
```
```
:: Windows - Library - Assuming you copied the DLLs following doc/installation/README.md#windows: Body
build\x64\Release\OpenPoseDemo.exe --net_resolution "1312x736" --scale_number 4 --scale_gap 0.25
:: Windows - Library - Assuming you copied the DLLs following doc/installation/README.md#windows: Body + Hand + Face
build\x64\Release\OpenPoseDemo.exe --net_resolution "1312x736" --scale_number 4 --scale_gap 0.25 --hand --hand_scale_number 6 --hand_scale_range 0.4 --face
```

If you want to increase the accuracy value metric on COCO, while harming the qualitative accuracy, add the flag `--maximize_positives`. It reduces the thresholds to accept a person candidate. It highly increases both false and true positives. I.e., it maximizes average recall but could harm average precision. Our experience is that it looks much worse visually, but it improves the COCO accuracy numbers, so use it at your own risk.

In addition, our paper numbers are not based on the current models that have been released. We released our best model at the time but later found a better one. But given that the accuracy difference is less than 2%, we did not want to release yet another model to avoid confusion for the users (otherwise there would be more than 10 models released at this point). We will release a new one every time a major improvement is achieved.

If you are operating on Ubuntu, you can check the experimental scripts that we use to test our accuracy (we do not officially support it, i.e., we will not answer questions about it, as well as it might change it continuously), they are placed in `openpose/scripts/tests/`, called `pose_accuracy_coco_test_dev.sh` and `pose_accuracy_coco_val.sh`.



### 3-D Reconstruction
1. Real-time demo
```
# Ubuntu and Mac
./build/examples/openpose/openpose.bin --flir_camera --3d --number_people_max 1
# With face and hands
./build/examples/openpose/openpose.bin --flir_camera --3d --number_people_max 1 --face --hand
```
```
:: Windows - Portable Demo
bin\OpenPoseDemo.exe --flir_camera --3d --number_people_max 1
:: With face and hands
bin\OpenPoseDemo.exe --flir_camera --3d --number_people_max 1 --face --hand
```
```
:: Windows - Library - Assuming you copied the DLLs following doc/installation/README.md#windows
build\x64\Release\OpenPoseDemo.exe --flir_camera --3d --number_people_max 1
:: With face and hands
build\x64\Release\OpenPoseDemo.exe --flir_camera --3d --number_people_max 1 --face --hand
```

2. Saving 3-D keypoints and video
```
# Ubuntu and Mac (same flags for Windows)
./build/examples/openpose/openpose.bin --flir_camera --3d --number_people_max 1 --write_json output_folder_path/ --write_video_3d output_folder_path/video_3d.avi
```

3. Fast stereo camera image saving (without keypoint detection) for later post-processing
```
# Ubuntu and Mac (same flags for Windows)
# Saving video
# Note: saving in PNG rather than JPG will improve image quality, but slow down FPS (depending on hard disk writing speed and camera number)
./build/examples/openpose/openpose.bin --flir_camera --num_gpu 0 --write_video output_folder_path/video.avi --write_video_fps 5
# Saving images
# Note: saving in PNG rather than JPG will improve image quality, but slow down FPS (depending on hard disk writing speed and camera number)
./build/examples/openpose/openpose.bin --flir_camera --num_gpu 0 --write_images output_folder_path/ --write_images_format jpg
```

4. Reading and processing previouly saved stereo camera images
```
# Ubuntu and Mac (same flags for Windows)
# Optionally add `--face` and/or `--hand` to include face and/or hands
# Assuming 3 cameras
# Note: We highly recommend to reduce `--output_resolution`. E.g., for 3 cameras recording at 1920x1080, the resulting image is (3x1920)x1080, so we recommend e.g. 640x360 (x3 reduction).
# Video
./build/examples/openpose/openpose.bin --video output_folder_path/video.avi --3d_views 3 --3d --number_people_max 1 --output_resolution {desired_output_resolution}
# Images
./build/examples/openpose/openpose.bin --image_dir output_folder_path/ --3d_views 3 --3d --number_people_max 1 --output_resolution {desired_output_resolution}
```

5. Reconstruction when the keypoint is visible in at least `x` camera views out of the total `n` cameras
```
# Ubuntu and Mac (same flags for Windows)
# Reconstruction when a keypoint is visible in at least 2 camera views (assuming `n` >= 2)
./build/examples/openpose/openpose.bin --flir_camera --3d --number_people_max 1 --3d_min_views 2 --output_resolution {desired_output_resolution}
# Reconstruction when a keypoint is visible in at least max(2, min(4, n-1)) camera views
./build/examples/openpose/openpose.bin --flir_camera --3d --number_people_max 1 --output_resolution {desired_output_resolution}
```



### Tracking
1. Runtime huge speed up by reducing the accuracy:
```
:: Windows - Portable Demo (same flags for Ubuntu and Mac)
# Using OpenPose 1 frame, tracking the following e.g., 5 frames
bin\OpenPoseDemo.exe --tracking 5 --number_people_max 1
```

2. Runtime speed up while keeping most of the accuracy:
```
:: Windows - Portable Demo (same flags for Ubuntu and Mac)
# Using OpenPose 1 frame and tracking another frame
bin\OpenPoseDemo.exe --tracking 1 --number_people_max 1
```

3. Visual smoothness:
```
:: Windows - Portable Demo (same flags for Ubuntu and Mac)
# Running both OpenPose and tracking on each frame. Note: There is no speed up/slow down
bin\OpenPoseDemo.exe --tracking 0 --number_people_max 1
```



## Kinect 2.0 as Webcam on Windows 10
Since the Windows 10 Anniversary, Kinect 2.0 can be read as a normal webcam. All you need to do is go to `device manager`, expand the `kinect sensor devices` tab, right click and update driver of `WDF kinectSensor Interface`. If you already have another webcam, disconnect it or use `--camera 2`.





### Main Flags
These are the most common flags, but check [doc/demo_not_quick_start.md](demo_not_quick_start.md) for a full list and description of all of them.

- `--face`: Enables face keypoint detection.
- `--hand`: Enables hand keypoint detection.
- `--video input.mp4`: Read video `input.mp4`.
- `--camera 3`: Read webcam number 3.
- `--image_dir path_with_images/`: Run on the directory `path_with_images/` with images.
- `--ip_camera http://iris.not.iac.es/axis-cgi/mjpg/video.cgi?resolution=320x240?x.mjpeg`: Run on a streamed IP camera. See examples public IP cameras [here](http://www.webcamxp.com/publicipcams.aspx).
- `--write_video path.avi`: Save processed images as video.
- `--write_images folder_path`: Save processed images on a folder.
- `--write_keypoint path/`: Output JSON, XML or YML files with the people pose data on a folder.
- `--process_real_time`: For video, it might skip frames to display at real time.
- `--disable_blending`: If enabled, it will render the results (keypoint skeletons or heatmaps) on a black background, not showing the original image. Related: `part_to_show`, `alpha_pose`, and `alpha_pose`.
- `--part_to_show`: Prediction channel to visualize.
- `--display 0`: Display window not opened. Useful for servers and/or to slightly speed up OpenPose.
- `--num_gpu 2 --num_gpu_start 1`: Parallelize over this number of GPUs starting by the desired device id. By default it uses all the available GPUs.
- `--model_pose MPI`: Model to use, affects number keypoints, speed and accuracy.
- `--logging_level 3`: Logging messages threshold, range [0,255]: 0 will output any message & 255 will output none. Current messages in the range [1-4], 1 for low priority messages and 4 for important ones.





## Advanced Quick Start
In order to learn about many more flags, check [doc/demo_not_quick_start.md](demo_not_quick_start.md).
