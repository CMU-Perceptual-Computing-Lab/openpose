OpenPose Doc - C++ API
====================================

## Contents
1. [Introduction](#introduction)
2. [Advance Introduction (Optional)](#advance-introduction-optional)
3. [Compatibility](#compatibility)
4. [Installation](#installation)
5. [Testing And Developing](#testing-and-developing)
6. [Exporting Python OpenPose](#exporting-python-openpose)
7. [Common Issues](#common-issues)



## Introduction
Extend the OpenPose functionality with all the power and performance of C++!

When should you look at the [Python](03_python_api.md) or [C++](04_cpp_api.md) APIs? If you want to read a specific input, and/or add your custom post-processing function, and/or implement your own display/saving.

You should be familiar with the [**OpenPose Demo**](01_demo.md) and the main OpenPose flags before trying to read the C++ or Python API examples. Otherwise, it will be way harder to follow.


## Adding your Custom Code
Once you are familiar with the [command line demo](01_demo.md), then you should explore the different C++ examples in the [OpenPose C++ API](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/examples/tutorial_api_cpp) folder.

For quick prototyping, you can simply **duplicate and rename any of the existing sample files** from the [OpenPose C++ API](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/examples/tutorial_api_cpp) folder into the [examples/user_code/](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/examples/user_code) folder and start building in there. Add the name of your new file(s) into the [CMake file from that folder](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/examples/user_code/CMakeLists.txt), and CMake will automatically compile it together with the whole OpenPose project.

See [examples/user_code/README.md](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/examples/user_code/README.md) for more details.
