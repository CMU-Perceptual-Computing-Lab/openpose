# OpenPose
OpenPose is a **library for real-time multi-person keypoint detection and multi-threading written in C++** using OpenCV and Caffe*, authored by [Gines Hidalgo](http://gines-hidalgo.site123.me/), [Zhe Cao](http://www.andrew.cmu.edu/user/zhecao), [Tomas Simon](http://www.cs.cmu.edu/~tsimon/), [Shih-En Wei](https://scholar.google.com/citations?user=sFQD3k4AAAAJ&hl=en), [Hanbyul Joo](http://www.cs.cmu.edu/~hanbyulj/) and [Yaser Sheikh](http://www.cs.cmu.edu/~yaser/).

\* It uses Caffe, but the code is ready to be ported to other frameworks (Tensorflow, Torch, etc.). If you implement any of those, feel free to make a pull request!

## Results
### Body + Hands + Face Estimation
<p align="center">
    <img src="doc/media/pose_face_hands.gif", width="480">
</p>

### Body Estimation
<p align="center">
    <img src="doc/media/dance.gif", width="480">
</p>

### Body + Face Estimation
<p align="center">
    <img src="doc/media/pose_face.gif", width="480">
</p>

### Body + Hands
<p align="center">
    <img src="doc/media/pose_hands.gif", width="480">
</p>

## Latest News
- Jul 2017: **Windows**, New [**portable demo**](doc/installation.md#installation---demo) **and** [**easier library installation**](doc/installation.md#installation---library)!
- Jul 2017: **Hands** released!
- Jun 2017: **Face** released!
- May 2017: **Windows** version released!
- Apr 2017: **Body** released!
- Check all the [release notes](doc/release_notes.md).

We are offering [internships at Carnegie Mellon University as OpenPose programmer](https://docs.google.com/document/d/14SygG39NjIRZfx08clewTdFMGwVdtRu2acyCi3TYcHs/edit?usp=sharing) (need to live in or be willing to move to Pittsburgh).

## Contents
1. [Introduction](#introduction)
2. [Library main functionality](#library-main-functionality)
3. [Operating Systems](#operating-systems)
1. [Installation, Reinstallation and Uninstallation](#installation-reinstallation-and-uninstallation)
2. [Custom Caffe](#custom-caffe)
3. [Quick Start](#quick-start)
    1. [Demo](#demo)
    2. [OpenPose Wrapper](#openpose-wrapper)
    3. [OpenPose Library](#openpose-library)
4. [Output](#output)
5. [Speed Up Openpose And Benchmark](#speed-up-openpose-and-benchmark)
6. [Send Us Failure Cases!](#send-us-failure-cases)
7. [Send Us Your Feedback!](#send-us-your-feedback)
8. [Citation](#citation)
9. [Other Contributors](#other-contributors)


## Introduction

OpenPose represents the **first real-time system to jointly detect human body, hand and facial keypoints (in total 130 keypoints) on single images**. In addition, the system computational performance on body keypoint estimation is invariant to the number of detected people in the image.

OpenPose is freely available for free non-commercial use, and may be redistributed under these conditions. Please, see the [license](LICENSE) for further details. [Interested in a commercial license? Check this link](https://flintbox.com/public/project/47343/). For commercial queries, contact [Yaser Sheikh](http://www.cs.cmu.edu/~yaser/).

In addition, OpenPose would not be possible without the [CMU Panoptic Studio](http://domedb.perception.cs.cmu.edu/).

The pose estimation work is based on the C++ code from [the ECCV 2016 demo](https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose), "Realtime Multiperson Pose Estimation", [Zhe Cao](http://www.andrew.cmu.edu/user/zhecao), [Tomas Simon](http://www.cs.cmu.edu/~tsimon/), [Shih-En Wei](https://scholar.google.com/citations?user=sFQD3k4AAAAJ&hl=en), [Yaser Sheikh](http://www.cs.cmu.edu/~yaser/). The [original repo](https://github.com/ZheC/Multi-Person-Pose-Estimation) includes Matlab and Python version, as well as the training code.


## Library main functionality:

* Multi-person 15 or **18-keypoint body pose** estimation and rendering. **Running time invariant to number of people** on the image.

* Multi-person **2x21-keypoint hand** estimation and rendering. Note: In this initial version, **running time** linearly **depends** on the **number of people** on the image. **Coming soon (in around 1-5 days)!**

* Multi-person **70-keypoint face** estimation and rendering. Note: In this initial version, **running time** linearly **depends** on the **number of people** on the image.

* Flexible and easy-to-configure **multi-threading** module.

* Image, video, and webcam reader.

* Able to save and load the results in various formats (JSON, XML, PNG, JPG, ...).

* Small display and GUI for simple result visualization.

* All the functionality is wrapped into a **simple-to-use OpenPose Wrapper class**.



## Operating Systems
1. **Ubuntu** 14 and 16.
2. **Windows** 8 and 10.
3. OpenPose has also been used on **Windows 7**, **Mac**, **CentOS**, and **Nvidia Jetson (TK1 and TX1)** embedded systems. However, we do not officially support them at the moment.



## Installation, Reinstallation and Uninstallation
You can find the installation, reinstallation and uninstallation steps on: [doc/installation.md](doc/installation.md).



## Custom Caffe
We only modified some Caffe compilation flags and minor details. You can use your own Caffe distribution, these are the files we added and modified:

1. Added files: `install_caffe.sh`; as well as `Makefile.config.Ubuntu14.example`, `Makefile.config.Ubuntu16.example`, `Makefile.config.Ubuntu14_cuda_7.example` and `Makefile.config.Ubuntu16_cuda_7.example` (extracted from `Makefile.config.example`). Basically, you must enable cuDNN.
2. Edited file: Makefile. Search for "# OpenPose: " to find the edited code. We basically added the C++11 flag to avoid issues in some old computers.
3. Optional - deleted Caffe file: `Makefile.config.example`.
4. In order to link it to OpenPose:
    1. Run `make all && make distribute` in your Caffe version.
    2. Open the OpenPose Makefile config file: `./Makefile.config.UbuntuX.example` (where X depends on your OS and CUDA version).
    3. Modify the Caffe folder directory variable (`CAFFE_DIR`) to your custom Caffe `distribute` folder location in the previous OpenPose Makefile config file.



## Quick Start
Most users cases should not need to dive deep into the library, they might just be able to use the [Demo](#demo) or the simple [OpenPose Wrapper](#openpose-wrapper). So you can most probably skip the library details in [OpenPose Library](#openpose-library).



#### Demo
Your case if you just want to process a folder of images or video or webcam and display or save the pose results.

Forget about the OpenPose library details and just read the [doc/demo_overview.md](doc/demo_overview.md) 1-page section.

#### OpenPose Wrapper
Your case if you want to read a specific format of image source and/or add a specific post-processing function and/or implement your own display/saving.

(Almost) forget about the library, just take a look to the `Wrapper` tutorial on [examples/tutorial_wrapper/](examples/tutorial_wrapper/).

Note: you should not need to modify the OpenPose source code nor examples. In this way, you are able to directly upgrade OpenPose anytime in the future without changing your code. You might create your custom code on [examples/user_code/](examples/user_code/) and compile it by using `make all` in the OpenPose folder.

#### OpenPose Library
Your case if you want to change internal functions and/or extend its functionality. First, take a look at the [Demo](#demo) and [OpenPose Wrapper](#openpose-wrapper). Second, read the 2 following subsections: OpenPose Overview and Extending Functionality.

1. OpenPose Overview: Learn the basics about the library source code in [doc/library_overview.md](doc/library_overview.md).

2. Extending Functionality: Learn how to extend the library in [doc/library_extend_functionality.md](doc/library_extend_functionality.md).

3. Adding An Extra Module: Learn how to add an extra module in [doc/library_add_new_module.md](doc/library_add_new_module.md).

#### Doxygen Documentation Autogeneration
You can generate the documentation by running the following command. The documentation will be generated in `doc/doxygen/html/index.html`. You can simply open it with double-click (your default browser should automatically display it).
```
cd doc/
doxygen doc_autogeneration.doxygen
```



## Output
Check the output (format, keypoint index ordering, etc.) in [doc/output.md](doc/output.md).



## Speed Up OpenPose and Benchmark
Check the OpenPose Benchmark and some hints to speed up OpenPose on [doc/installation.md#faq](doc/installation.md#faq).



## Send Us Failure Cases!
If you find videos or images where OpenPose does not seems to work well, feel free to send them to openposecmu@gmail.com, we will use them to improve the quality of the algorithm. Thanks!



## Send Us Your Feedback!
Our library is open source for research purposes, and we want to continuously improve it! So please, let us know if...

1. ... you find any bug (in functionality or speed).

2. ... you added some functionality to some class or some new Worker<T> subclass which we might potentially incorporate.

3. ... you know how to speed up or improve any part of the library.

4. ... you have a request about possible functionality.

5. ... etc.

Just comment on GibHub or make a pull request and we will answer as soon as possible! Send us an email if you use the library to make a cool demo or YouTube video!



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
