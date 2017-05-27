OpenPose
====================================

## Introduction

OpenPose is a **library for real-time multi-person keypoint detection and multi-threading written in C++** using OpenCV and Caffe*, authored by [Gines Hidalgo](https://www.linkedin.com/in/gineshidalgo/), [Zhe Cao](http://www.andrew.cmu.edu/user/zhecao), [Tomas Simon](http://www.cs.cmu.edu/~tsimon/), [Shih-En Wei](https://scholar.google.com/citations?user=sFQD3k4AAAAJ&hl=en), [Hanbyul Joo](http://www.cs.cmu.edu/~hanbyulj/) and [Yaser Sheikh](http://www.cs.cmu.edu/~yaser/).

* It uses Caffe, but the code is ready to be ported to other frameworks (e.g., Tensorflow or Torch). If you implement any of those, please, make a pull request and we will add it!

OpenPose represents the **first real-time system to jointly detect human body, hand and facial keypoints (in total 130 keypoints) on single images**. In addition, the system computational performance on body keypoint estimation is invariant to the number of detected people in the image.

OpenPose is freely available for free non-commercial use, and may be redistributed under these conditions. Please, see the [license](LICENSE) for further details. Contact us for commercial purposes.



Library main functionality:

* Multi-person 15 or **18-keypoint body pose** estimation and rendering.

* Multi-person **2x21-keypoint hand** estimation and rendering (coming soon in around 1-2 months!).

* Multi-person **70-keypoint face** estimation and rendering (coming soon in around 2-3 months!).

* Flexible and easy-to-configure **multi-threading** module.

* Image, video, and webcam reader.

* Able to save and load the results in various formats (JSON, XML, PNG, JPG, ...).

* Small display and GUI for simple result visualization.

* All the functionality is wrapped into a **simple-to-use OpenPose Wrapper class**.

The pose estimation work is based on the C++ code from [the ECCV 2016 demo](https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose), "Realtime Multiperson Pose Estimation", [Zhe Cao](http://www.andrew.cmu.edu/user/zhecao), [Tomas Simon](http://www.cs.cmu.edu/~tsimon/), [Shih-En Wei](https://scholar.google.com/citations?user=sFQD3k4AAAAJ&hl=en), [Yaser Sheikh](http://www.cs.cmu.edu/~yaser/). The [full project repo](https://github.com/ZheC/Multi-Person-Pose-Estimation) includes Matlab and Python version, as well as training code.



## Results

### Body Estimation
<p align="center">
    <img src="doc/media/dance.gif", width="480">
</p>

## Coming Soon (But Already Working!)

### Body + Hands + Face Estimation
<p align="center">
    <img src="doc/media/pose_face_hands.gif", width="480">
</p>

### Body + Face Estimation
<p align="center">
    <img src="doc/media/pose_face.gif", width="480">
</p>

### Body + Hands
<p align="center">
    <img src="doc/media/pose_hands.gif", width="480">
</p>


## Contents
1. [Installation, Reinstallation and Uninstallation](#installation-reinstallation-and-uninstallation)
2. [Custom Caffe](#custom-caffe)
3. [Quick Start](#quick-start)
    1. [Demo](#demo)
    2. [OpenPose Wrapper](#openpose-wrapper)
    3. [OpenPose Library](#openpose-library)
4. [Output](#output)
    1. [Output Format](#output-format)
    2. [Reading Saved Results](#reading-saved-results)
5. [OpenPose Benchmark](#openpose-benchmark)
6. [Send Us Your Feedback!](#send-us-your-feedback)
7. [Citation](#citation)
8. [Other Contributors](#other-contributors)



## Installation, Reinstallation and Uninstallation
You can find the installation, reinstallation and uninstallation steps on: [doc/installation.md](doc/installation.md).



## Custom Caffe
We only modified some Caffe compilation flags and minor details. You can use use your own Caffe distribution, these are the files we added and modified:

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

Note: you should not need to modify OpenPose source code or examples, so that you can directly upgrade the OpenPose library anytime in the future without changing your code. You might create your custom code on [examples/user_code/](examples/user_code/) and compile it by using `make all` in the OpenPose folder.

#### OpenPose Library
Your case if you want to change internal functions and/or extend its functionality. First, take a look at the [Demo](#demo) and [OpenPose Wrapper](#openpose-wrapper). Second, read the 2 following subsections: OpenPose Overview and Extending Functionality.

1. OpenPose Overview: Learn the basics about the library source code in [doc/library_overview.md](doc/library_overview.md).

2. Extending Functionality: Learn how to extend the library in [doc/library_extend_functionality.md](doc/library_extend_functionality.md).

3. Adding An Extra Module: Learn how to add an extra module in [doc/library_add_new_module.md](doc/library_add_new_module.md).

#### Doxygen Documentation Autogeneration
You can generate the documentation by running the following command. The documentation will be generated in `doc/doxygen/html/index.html`. You can simply open it with double click (your default browser should automatically display it).
```
cd doc/
doxygen doc_autogeneration.doxygen
```



## Output
#### Output Format
There are 2 alternatives to save the **(x,y,score) body part locations**. The `write_pose` flag uses the OpenCV cv::FileStorage default formats (JSON, XML and YML). However, the JSON format is only available after OpenCV 3.0. Hence, `write_pose_json` saves the people pose data using a custom JSON writer. For the latter, each JSON file has a `people` array of objects, where each object has an array `body_parts` containing the body part locations and detection confidence formatted as `x1,y1,c1,x2,y2,c2,...`. The coordinates `x` and `y` can be normalized to the range [0,1], [-1,1], [0, source size], [0, output size], etc., depending on the flag `scale_mode`. In addition, `c` is the confidence in the range [0,1].

```
{
    "version":0.1,
    "people":[
        {"body_parts":[1114.15,160.396,0.846207,...]},
        {"body_parts":[...]},
    ]
}
```

The body part order of the COCO (18 body parts) and MPI (15 body parts) keypoints is described in `POSE_BODY_PART_MAPPING` in [include/openpose/pose/poseParameters.hpp](include/openpose/pose/poseParameters.hpp). E.g., for COCO:
```
    POSE_COCO_BODY_PARTS {
        {0,  "Nose"},
        {1,  "Neck"},
        {2,  "RShoulder"},
        {3,  "RElbow"},
        {4,  "RWrist"},
        {5,  "LShoulder"},
        {6,  "LElbow"},
        {7,  "LWrist"},
        {8,  "RHip"},
        {9,  "RKnee"},
        {10, "RAnkle"},
        {11, "LHip"},
        {12, "LKnee"},
        {13, "LAnkle"},
        {14, "REye"},
        {15, "LEye"},
        {16, "REar"},
        {17, "LEar"},
        {18, "Bkg"},
    }
```

For the **heat maps storing format**, instead of individually saving each of the 67 heatmaps (18 body parts + background + 2 x 19 PAFs) individually, the library concatenates them into a huge (width x #heat maps) x (height) matrix, i.e. it concats the heat maps by columns. E.g., columns [0, individual heat map width] contains the first heat map, columns [individual heat map width + 1, 2 * individual heat map width] contains the second heat map, etc. Note that some image viewers are not able to display the resulting images due to the size. However, Chrome and Firefox are able to properly open them.

The saving order is body parts + background + PAFs. Any of them can be disabled with program flags. If background is disabled, then the final image will be body parts + PAFs. The body parts and background follow the order of `POSE_COCO_BODY_PARTS` or `POSE_MPI_BODY_PARTS`, while the PAFs follow the order specified on POSE_BODY_PART_PAIRS in `poseParameters.hpp`. E.g., for COCO:
```
    POSE_COCO_PAIRS    {1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10, 1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17,   2,16,  5,17};
```

Where each index is the key value corresponding to each body part in `POSE_COCO_BODY_PARTS`, e.g., 0 for "Neck", 1 for "RShoulder", etc.

#### Reading Saved Results
We use standard formats (JSON, XML, PNG, JPG, ...) to save our results, so there will be lots of frameworks to read them later, but you might also directly use our functions in [include/openpose/filestream.hpp](include/openpose/filestream.hpp). In particular, `loadData` (for JSON, XML and YML files) and `loadImage` (for image formats such as PNG or JPG) to load the data into cv::Mat format.



## OpenPose Benchmark
Initial library running time benchmark on [OpenPose Benchmark](https://docs.google.com/spreadsheets/d/1-DynFGvoScvfWDA1P4jDInCkbD4lg0IKOYbXgEq0sK0/edit#gid=0). You can comment in that document with your graphics card model and running time for that model, and we will add your results to the benchmark!



## Send Us Your Feedback!
Our library is open source for research purposes, and we want to continuously improve it! So please, let us know if...

1. ... you find any bug (in functionality or speed).

2. ... you added some functionality to some class or some new Worker<T> subclass which we might potentially incorporate.

3. ... you know how to speed up or improve any part of the library.

4. ... you have a request about possible functionality.

5. ... etc.

Just comment on GibHub or make a pull request and we will answer as soon as possible! Send us an email if you use the library to make a cool demo or YouTube video!



## Citation
Please cite the papers in your publications if it helps your research:

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
We would like to thank the following people who also contributed to OpenPose:

1. [Helen Medina](https://github.com/helen-medina): For moving OpenPose to Windows (Visual Studio), making it work there and creating the Windows branch.
