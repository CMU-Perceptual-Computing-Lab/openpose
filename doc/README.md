OpenPose - Quick Start
==========================

Most users can simply use the OpenPose Demo without reading any C++/Python code. Users that need to add additional functionality (new inputs, outputs, etc) should check the C++/Python APIs:

- The first step for any software, [installing](installation.md) it!

- [**Output information**](doc/output.md): Learn about the output format, keypoint index ordering, etc.

- [**OpenPose Demo**](demo_quick_start.md) for your terminal or PowerShell: Choose your input (e.g., images, video, webcam), set of algorithms (body, hand, face), and output (e.g., display, JSON keypoint saving, image+keypoints).
    - E.g.: Given an input video (`--video`), extract body (by default), face (`--face`) and hand (`--hand`) keypoints, save the keypoints in a JSON file (`--write_json`), and display (by default) the results in the screen. You can remove any of the flags to remove that particular functionality or add any other.
```
# Ubuntu
./build/examples/openpose/openpose.bin --video examples/media/video.avi --face --hand --write_json output_json_folder/

:: Windows - Portable Demo
bin\OpenPoseDemo.exe --video examples\media\video.avi --face --hand --write_json output_json_folder/
```

- [**OpenPose C++ API**](../examples/tutorial_api_cpp/): If you want to read a specific input, and/or add your custom post-processing function, and/or implement your own display/saving.
	- You should be familiar with the [**OpenPose Demo**](demo_quick_start.md) and the main OpenPose flags before trying to read the C++ or Python API examples. Otherwise, it will be way harder to follow.
    - For quick prototyping: You can easily **create your custom code** on [examples/user_code/](examples/user_code/) and CMake will automatically compile it together with the whole OpenPose project. See [examples/user_code/README.md](examples/user_code/README.md) for more details.

- [**OpenPose Python API**](../examples/tutorial_api_python/): Almost the exact same functionality than the C++ API, but in Python!
	- You should be familiar with the [**OpenPose Demo**](demo_quick_start.md) and the main OpenPose flags before trying to read the C++ or Python API examples. Otherwise, it will be way harder to follow.
    - For quick prototyping: You can simply duplicate and rename any of the [existing example files](examples/tutorial_api_python/) within that same folder.

- [**Speeding up OpenPose and benchmark**](speed_up_openpose.md): Check the OpenPose Benchmark as well as some hints to speed up and/or reduce the memory requirements for OpenPose.

- [**Calibration toolbox**](advanced/calibration_module.md) and [**3D OpenPose**](advanced/3d_reconstruction_module.md): Calibrate your cameras for 3D OpenPose (or any other stereo vision tasks) and start obtaining 3D keypoints!

- [**Standalone face or hand detector**](advanced/standalone_face_or_hand_keypoint_detector.md) is useful if you want to do any of the following:
    - **Face** keypoint detection **without body** keypoint detection: Pros: Speedup and RAM/GPU memory reduction. Cons: Worse accuracy and less detected number of faces).
    - **Use your own face/hand detector**: You can use the hand and/or face keypoint detectors with your own face or hand detectors, rather than using the body detector. E.g., useful for camera views at which the hands are visible but not the body (OpenPose detector would fail).
