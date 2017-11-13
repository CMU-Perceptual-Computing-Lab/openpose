<div align="center">
    <img src=".github/Logo_main_black.png", width="480">
</div>

-----------------

[![Build Status](https://travis-ci.org/CMU-Perceptual-Computing-Lab/openpose.svg?branch=master)](https://travis-ci.org/CMU-Perceptual-Computing-Lab/openpose)

OpenPose is a **library for real-time multi-person keypoint detection and multi-threading written in C++** using OpenCV and Caffe.
<p align="center">
    <img src="doc/media/pose_face_hands.gif", width="480">
</p>



## Latest News
- Sep 2017: **CMake** installer and **IP camera** support!
- Jul 2017: [**Windows portable demo**](doc/installation.md#installation---demo)!
- Jul 2017: **Hands** released!
- Jun 2017: **Face** released!
- May 2017: **Windows** version!
- Apr 2017: **Body** released!
- Check all the [release notes](doc/release_notes.md).



## Results
### Body Estimation
<p align="center">
    <img src="doc/media/dance.gif", width="480">
</p>

### Body + Face + Hands Estimation
<p align="center">
    <img src="doc/media/pose_face.gif", width="480">
</p>

### Body + Hands
<p align="center">
    <img src="doc/media/pose_hands.gif", width="480">
</p>



## Contents
1. [Latest News](#latest-news)
2. [Results](#results)
3. [Introduction](#introduction)
4. [Functionality](#functionality)
5. [Installation, Reinstallation and Uninstallation](#installation-reinstallation-and-uninstallation)
6. [Quick Start](#quick-start)
    1. [Demo](#demo)
    2. [OpenPose Wrapper](#openpose-wrapper)
    3. [Adding An Extra Module](#Adding-an-extra-module)
    4. [OpenPose C++ API](#openpose-c++-api)
7. [Output](#output)
8. [Standalone Face Or Hand Keypoint Detector](#standalone-face-or-hand-keypoint-detector)
9. [Speed Up Openpose And Benchmark](#speed-up-openpose-and-benchmark)
10. [Send Us Failure Cases!](#send-us-failure-cases)
11. [Send Us Your Feedback!](#send-us-your-feedback)
12. [Citation](#citation)
12. [Other Contributors](#other-contributors)



## Introduction
OpenPose represents the **first real-time system to jointly detect human body, hand and facial keypoints (in total 130 keypoints) on single images**. In addition, the system computational performance on body keypoint estimation is invariant to the number of detected people in the image. It uses Caffe, but it could easily be ported to other frameworks (Tensorflow, Torch, etc.). If you implement any of those, feel free to make a pull request!

OpenPose is authored by [Gines Hidalgo](https://www.gineshidalgo.com/), [Zhe Cao](http://www.andrew.cmu.edu/user/zhecao), [Tomas Simon](http://www.cs.cmu.edu/~tsimon/), [Shih-En Wei](https://scholar.google.com/citations?user=sFQD3k4AAAAJ&hl=en), [Hanbyul Joo](http://www.cs.cmu.edu/~hanbyulj/), and [Yaser Sheikh](http://www.cs.cmu.edu/~yaser/). Currently, it is being maintained by [Gines Hidalgo](https://www.gineshidalgo.com/) and [Bikramjot Hanzra](https://www.linkedin.com/in/bikz05).

It is freely available for free non-commercial use, and may be redistributed under these conditions. Please note, that all derivative work will become property of Carnegie Mellon University and will be licensed under the same conditions to the contributing author. Please, see the [license](LICENSE) for further details. [Interested in a commercial license? Check this link](https://flintbox.com/public/project/47343/). For commercial queries, contact [Yaser Sheikh](http://www.cs.cmu.edu/~yaser/).

In addition, OpenPose would not be possible without the [CMU Panoptic Studio](http://domedb.perception.cs.cmu.edu/).

The pose estimation work is based on the C++ code from [the ECCV 2016 demo](https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose), "Realtime Multiperson Pose Estimation", [Zhe Cao](http://www.andrew.cmu.edu/user/zhecao), [Tomas Simon](http://www.cs.cmu.edu/~tsimon/), [Shih-En Wei](https://scholar.google.com/citations?user=sFQD3k4AAAAJ&hl=en), [Yaser Sheikh](http://www.cs.cmu.edu/~yaser/). The [original repo](https://github.com/ZheC/Multi-Person-Pose-Estimation) includes Matlab and Python version, as well as the training code.


## Functionality
- Multi-person 15 or **18-keypoint body pose** estimation and rendering. **Running time invariant to number of people** on the image.
- Multi-person **2x21-keypoint hand** estimation and rendering. Note: In this initial version, **running time** linearly **depends** on the **number of people** on the image.
- Multi-person **70-keypoint face** estimation and rendering. Note: In this initial version, **running time** linearly **depends** on the **number of people** on the image.
- Flexible and easy-to-configure **multi-threading** module.
- Image, video, webcam and IP camera reader.
- Able to save and load the results in various formats (JSON, XML, PNG, JPG, ...).
- Small display and GUI for simple result visualization.
- All the functionality is wrapped into a **simple-to-use OpenPose Wrapper class**.



## Installation, Reinstallation and Uninstallation
You can find the installation, reinstallation and uninstallation steps on: [doc/installation.md](doc/installation.md).



## Quick Start
Most users do not need the [OpenPose C++ API](#openpose-c++-api), but they can simply use the basic [Demo](#demo) and/or [OpenPose Wrapper](#openpose-wrapper).

### Demo
Ideal to process images/video/webcam and display/save the results. Check [doc/demo_overview.md](doc/demo_overview.md).

### OpenPose Wrapper
Ideal if you want to read a specific input, and/or add your custom post-processing function, and/or implement your own display/saving. Take a look to the `Wrapper` tutorial on [examples/tutorial_wrapper/](examples/tutorial_wrapper/). You might create your custom code on [examples/user_code/](examples/user_code/) and compile it by using `make all` in the OpenPose folder.

### Adding An Extra Module
Learn how to easily add an extra module to OpenPose in [doc/library_add_new_module.md](./library_add_new_module.md).

### OpenPose C++ API
Your case if you want to use the C++ API. See [doc/library_introduction.md](doc/library_introduction.md).



## Output
Output (format, keypoint index ordering, etc.) in [doc/output.md](doc/output.md).



## Standalone Face Or Hand Keypoint Detector
If you do not need the body detector and want to speed up the face keypoint detection, you can use the OpenCV-based approach, see [doc/standalone_face_or_hand_keypoint_detector.md](doc/standalone_face_or_hand_keypoint_detector.md).

You can also use the OpenPose hand and/or face keypoint detectors with your own face or hand detectors, rather than using the body detector. E.g. useful for camera views at which the hands are visible but not the body, so that the OpenPose detector would fail. See [doc/standalone_face_or_hand_keypoint_detector.md](doc/standalone_face_or_hand_keypoint_detector.md).



## Speed Up OpenPose and Benchmark
Check the OpenPose Benchmark and some hints to speed up OpenPose on [doc/installation.md#faq](doc/installation.md#faq).



## Send Us Failure Cases!
If you find videos or images where OpenPose does not seems to work well, feel free to send them to openposecmu@gmail.com (email only for failure cases!), we will use them to improve the quality of the algorithm. Thanks!



## Send Us Your Feedback!
Our library is open source for research purposes, and we want to continuously improve it! So please, let us know if...

1. ... you find any bug (in functionality or speed).

2. ... you added some functionality to some class or some new Worker<T> subclass which we might potentially incorporate.

3. ... you know how to speed up or improve any part of the library.

4. ... you have a request about possible functionality.

5. ... etc.

Just comment on GitHub or make a pull request and we will answer as soon as possible! Send us an email if you use the library to make a cool demo or YouTube video!



## Citation
Please cite these papers in your publications if it helps your research (the face keypoint detector was trained using the same procedure described in [Simon et al. 2017]):

    @inproceedings{cao2017realtime,
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      year = {2017}
    }

    @inproceedings{simon2017hand,
      author = {Tomas Simon and Hanbyul Joo and Iain Matthews and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Hand Keypoint Detection in Single Images using Multiview Bootstrapping},
      year = {2017}
    }

    @inproceedings{wei2016cpm,
      author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Convolutional pose machines},
      year = {2016}
    }



## Other Contributors
We would like to thank all the people who helped OpenPose in any way. The main contributors are listed in [doc/contributors.md](doc/contributors.md).
