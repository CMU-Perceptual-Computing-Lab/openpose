OpenPose - Quick Start
====================================

## Contents
1. [Quick Start](#quick-start)
    1. [Running on Video](#running-on-video)
    2. [Running on Webcam](#running-on-webcam)
    3. [Running on Images](#running-on-images)
    4. [Maximum Accuracy Configuration](#Maximum-accuracy-configuration)
2. [Expected Visual Results](#expected-visual-results)



## Quick Start
Check that the library is working properly by running any of the following commands. Make sure that you are in the **root directory of the project** (i.e. in the OpenPose folder, not inside `build/` nor `windows/` nor `bin/`). In addition, `examples/media/video.avi` and `examples/media` do exist, no need to change the paths.

### Running on Video
```
# Ubuntu
./build/examples/openpose/openpose.bin --video examples/media/video.avi
# With face and hands
./build/examples/openpose/openpose.bin --video examples/media/video.avi --face --hand
```
```
:: Windows - Demo
bin\OpenPoseDemo.exe --video examples\media\video.avi
:: With face and hands
bin\OpenPoseDemo.exe --video examples\media\video.avi --face --hand
```
```
:: Windows - Library
windows\x64\Release\OpenPoseDemo.exe --video examples\media\video.avi
:: With face and hands
windows\x64\Release\OpenPoseDemo.exe --video examples\media\video.avi --face --hand
```



### Running on Webcam
```
# Ubuntu
./build/examples/openpose/openpose.bin
# With face and hands
./build/examples/openpose/openpose.bin --face --hand
```
```
:: Windows - Demo
bin\OpenPoseDemo.exe
:: With face and hands
bin\OpenPoseDemo.exe --face --hand
```
```
:: Windows - Library
windows\x64\Release\OpenPoseDemo.exe
:: With face and hands
windows\x64\Release\OpenPoseDemo.exe --face --hand
```



### Running on Images
```
# Ubuntu
./build/examples/openpose/openpose.bin --image_dir examples/media/
# With face and hands
./build/examples/openpose/openpose.bin --image_dir examples/media/ --face --hand
```
```
:: Windows - Demo
bin\OpenPoseDemo.exe --image_dir examples\media\
:: With face and hands
bin\OpenPoseDemo.exe --image_dir examples\media\ --face --hand
```
```
:: Windows - Library
windows\x64\Release\OpenPoseDemo.exe --image_dir examples\media\
:: With face and hands
windows\x64\Release\OpenPoseDemo.exe --image_dir examples\media\ --face --hand
```



### Maximum Accuracy Configuration
This command provides the most accurate results we have been able to achieve for body, hand and face keypoint detection. However, this command will need around 8 GB of GPU memory and runs around 1 FPS on a Titan X.
```
# Ubuntu
./build/examples/openpose/openpose.bin --net_resolution "1312x736" --scale_number 4 --scale_gap 0.25 --hand --hand_scale_number 6 --hand_scale_range 0.4 --face
```
```
:: Windows - Demo
bin\OpenPoseDemo.exe --net_resolution "1312x736" --scale_number 4 --scale_gap 0.25 --hand --hand_scale_number 6 --hand_scale_range 0.4 --face
```
```
:: Windows - Library
windows\x64\Release\OpenPoseDemo.exe --net_resolution "1312x736" --scale_number 4 --scale_gap 0.25 --hand --hand_scale_number 6 --hand_scale_range 0.4 --face
```



## Expected Visual Results
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
