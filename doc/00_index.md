OpenPose Doc
==========================

The OpenPose documentation is available in 2 different formats, choose your preferred one!
- As a traditional website (recommended): [cmu-perceptual-computing-lab.github.io/openpose](https://cmu-perceptual-computing-lab.github.io/openpose).
- As markdown files: [github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/00_index.md](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/00_index.md).

Most users can simply use the OpenPose Demo without reading any C++/Python code. Users that need to add additional functionality (new inputs, outputs, etc) should check the C++/Python APIs:

- If you face issues with any of these steps, keep in mind to check the [FAQ](installation/05_faq.md) section.

- The first step for any software, [install it](installation/0_index.md)!

- [OpenPose Demo](01_demo.md): Choose your input (e.g., images, video, webcam), set of algorithms (body, hand, face), output (e.g., display, JSON keypoint saving, image+keypoints), and run OpenPose from your terminal or PowerShell!
    - E.g.: Given an input video (`--video`), extract body (by default), face (`--face`) and hand (`--hand`) keypoints, save the keypoints in a JSON file (`--write_json`), and display (by default) the results in the screen. You can remove any of the flags to remove that particular functionality or add any other.
```
# Ubuntu
./build/examples/openpose/openpose.bin --video examples/media/video.avi --face --hand --write_json output_json_folder/

:: Windows - Portable Demo
bin\OpenPoseDemo.exe --video examples\media\video.avi --face --hand --write_json output_json_folder/
```

- [Output information](02_output.md): Learn about the output format, keypoint index ordering, etc.

- [OpenPose Python API](03_python_api.md): Almost all the OpenPose functionality, but in Python! If you want to read a specific input, and/or add your custom post-processing function, and/or implement your own display/saving.

- [OpenPose C++ API](04_cpp_api.md): If you want to read a specific input, and/or add your custom post-processing function, and/or implement your own display/saving.

- [Maximizing OpenPose speed and benchmark](06_maximizing_openpose_speed.md): Check the OpenPose Benchmark as well as some hints to speed up and/or reduce the memory requirements for OpenPose.

- [Calibration toolbox](advanced/calibration_module.md) and [3D OpenPose](advanced/3d_reconstruction_module.md): Calibrate your cameras for 3D OpenPose (or any other stereo vision tasks) and start obtaining 3D keypoints!

- [Standalone face or hand detector](advanced/standalone_face_or_hand_keypoint_detector.md) is useful if you want to do any of the following:
    - **Face** keypoint detection **without body** keypoint detection: Pros: Speedup and RAM/GPU memory reduction. Cons: Worse accuracy and less detected number of faces).
    - **Use your own face/hand detector**: You can use the hand and/or face keypoint detectors with your own face or hand detectors, rather than using the body detector. E.g., useful for camera views at which the hands are visible but not the body (OpenPose detector would fail).
