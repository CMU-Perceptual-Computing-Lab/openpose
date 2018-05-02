<div align="center">
    <img src=".github/Logo_main_black.png", width="300">
</div>

-----------------

| **`Linux`** |
|-------------|
|[![Build Status](https://travis-ci.org/CMU-Perceptual-Computing-Lab/openpose.svg?branch=master)](https://travis-ci.org/CMU-Perceptual-Computing-Lab/openpose)|

[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) represents the **first real-time multi-person system to jointly detect human body, hand, and facial keypoints (in total 130 keypoints) on single images**.

<p align="center">
    <img src="doc/media/pose_face_hands.gif", width="480">
</p>

**Functionality**:

- **Real-time multi-person keypoint detection**.
    - 15 or **18-keypoint body estimation**. **Running time invariant to number of detected people**.
    - **2x21-keypoint hand** estimation. Currently, **running time depends** on **number of detected people**.
    - **70-keypoint face** estimation. Currently, **running time depends** on **number of detected people**.
- **Input**: Image, video, webcam, and IP camera. Included C++ demos to add your custom input.
- **Output**: Basic image + keypoint display/saving (PNG, JPG, AVI, ...), keypoint saving (JSON, XML, YML, ...), and/or keypoints as array class.
- Available: command-line demo, C++ wrapper, and C++ API.
- **OS**: Ubuntu (14, 16), Windows (8, 10), Nvidia TX2.



## Latest Features
- Mar 2018: [**CPU version**](doc/installation.md#cpu-version)!
- Mar 2018: Improved [**3-D keypoint reconstruction module**](doc/3d_reconstruction_demo.md) (from multiple camera views)!
- Sep 2017: [**CMake**](doc/installation.md) installer and **IP camera** support!
- Jul 2017: [**Windows portable binaries and demo**](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases)!
- Jul 2017: **Hands** released!
- Jun 2017: **Face** released!
For further details, check [all released features](doc/released_features.md) and [release notes](doc/release_notes.md).



## Contents
1. [Latest Features](#latest-features)
2. [Results](#results)
3. [Installation, Reinstallation and Uninstallation](#installation-reinstallation-and-uninstallation)
4. [Quick Start](#quick-start)
5. [Output](#output)
6. [Speeding Up OpenPose and Benchmark](#speeding-up-openpose-and-benchmark)
7. [Send Us Failure Cases and Feedback!](#send-us-failure-cases-and-feedback)
8. [Authors and Contributors](#authors-and-contributors)
9. [Citation](#citation)
10. [License](#license)



## Results
### 3-D Reconstruction Module
<p align="center">
    <img src="doc/media/openpose3d.png", width="360">
</p>

### Body Estimation
<p align="center">
    <img src="doc/media/dance.gif", width="360">
</p>

### Body, Face, and Hands Estimation
<p align="center">
    <img src="doc/media/pose_face.gif", width="360">
</p>

### Body and Hands Estimation
<p align="center">
    <img src="doc/media/pose_hands.gif", width="360">
</p>



## Installation, Reinstallation and Uninstallation
**Windows portable version**: Simply download and use the latest version from the [Releases](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases) section.

Otherwise, check [doc/installation.md](doc/installation.md) for instructions on how to build OpenPose from source.



## Quick Start
Most users do not need the [OpenPose C++ API](#openpose-c-api), but they can simply use the basic [Demo](#demo) and/or [OpenPose Wrapper](#openpose-wrapper).

- **Demo**: To easily process images/video/webcam and display/save the results. See [doc/demo_overview.md](doc/demo_overview.md). E.g. run OpenPose in a video with:
```
# Ubuntu
./build/examples/openpose/openpose.bin --video examples/media/video.avi
:: Windows - Portable Demo
bin\OpenPoseDemo.exe --video examples\media\video.avi
```

- **OpenPose Wrapper**: If you want to read a specific input, and/or add your custom post-processing function, and/or implement your own display/saving, check the `Wrapper` tutorial on [examples/tutorial_wrapper/](examples/tutorial_wrapper/). You can create your custom code on [examples/user_code/](examples/user_code/) and quickly compile it by using `make all` in the OpenPose folder (assuming Makefile installer).

- **OpenPose C++ API**: See [doc/library_introduction.md](doc/library_introduction.md).

- **Adding an extra module**: Check [doc/library_add_new_module.md](./doc/library_add_new_module.md).

- **Standalone face or hand detector**:
    - **Face** keypoint detection **without body** keypoint detection: If you want to speed it up (but also reduce amount of detected faces), check the OpenCV-face-detector approach in [doc/standalone_face_or_hand_keypoint_detector.md](doc/standalone_face_or_hand_keypoint_detector.md).
    - **Use your own face/hand detector**: You can use the hand and/or face keypoint detectors with your own face or hand detectors, rather than using the body detector. E.g. useful for camera views at which the hands are visible but not the body (OpenPose detector would fail). See [doc/standalone_face_or_hand_keypoint_detector.md](doc/standalone_face_or_hand_keypoint_detector.md).

- **Library dependencies**: OpenPose uses default Caffe and OpenCV, as well as any Caffe dependency. The demos additionally use GFlags. It could easily be ported to other deep learning frameworks (Tensorflow, Torch, ...). Feel free to make a pull request if you implement any of those!



## Output
Output (format, keypoint index ordering, etc.) in [doc/output.md](doc/output.md).



## Speeding Up OpenPose and Benchmark
Check the OpenPose Benchmark and some hints to speed up OpenPose on [doc/faq.md#speed-up-and-benchmark](doc/faq.md#speed-up-and-benchmark).



## Send Us Failure Cases and Feedback!
Our library is open source for research purposes, and we want to continuously improve it! So please, let us know if...

1. ... you find videos or images where OpenPose does not seems to work well. Feel free to send them to openposecmu@gmail.com (email only for failure cases!), we will use them to improve the quality of the algorithm!
2. ... you find any bug (in functionality or speed).
3. ... you added some functionality to some class or some new Worker<T> subclass which we might potentially incorporate.
4. ... you know how to speed up or improve any part of the library.
5. ... you have a request about possible functionality.
6. ... etc.

Just comment on GitHub or make a pull request and we will answer as soon as possible! Send us an email if you use the library to make a cool demo or YouTube video!



## Authors and Contributors
OpenPose is authored by [Gines Hidalgo](https://www.gineshidalgo.com/), [Zhe Cao](http://www.andrew.cmu.edu/user/zhecao), [Tomas Simon](http://www.cs.cmu.edu/~tsimon/), [Shih-En Wei](https://scholar.google.com/citations?user=sFQD3k4AAAAJ&hl=en), [Hanbyul Joo](http://www.cs.cmu.edu/~hanbyulj/), and [Yaser Sheikh](http://www.cs.cmu.edu/~yaser/). Currently, it is being maintained by [Gines Hidalgo](https://www.gineshidalgo.com/) and [Yaadhav Raaj](https://www.linkedin.com/in/yaadhavraaj). The [original CVPR 2017 repo](https://github.com/ZheC/Multi-Person-Pose-Estimation) includes Matlab and Python versions, as well as the training code. The body pose estimation work is based on [the original ECCV 2016 demo](https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose).

In addition, OpenPose would not be possible without the [CMU Panoptic Studio dataset](http://domedb.perception.cs.cmu.edu/).

We would also like to thank all the people who helped OpenPose in any way. The main contributors are listed in [doc/contributors.md](doc/contributors.md).



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



## License
OpenPose is freely available for free non-commercial use, and may be redistributed under these conditions. Please, see the [license](LICENSE) for further details. [Interested in a commercial license? Check this link](https://flintbox.com/public/project/47343/). For commercial queries, contact [Yaser Sheikh](http://www.cs.cmu.edu/~yaser/).
