Deploying OpenPose (Exporting OpenPose to Other Projects)
==========================

## Contents
1. [Introduction](#introduction)
2. [Third-Party Libraries](#third-party-libraries)
3. [Private OpenPose Include Directory](#private-openpose-include-directory)
4. [Crash and Core Dumped Avoidance](#crash-and-core-dumped-avoidance)
5. [Deploying OpenPose](#deploying-openpose)
    1. [Windows](#windows)
    2. [CMake (Windows, Ubuntu, and Mac)](#cmake-windows-ubuntu-and-mac)



### Introduction
Starting in OpenPose 1.6.0 (GitHub code in or after October 2019), OpenPose has considerable refactor its code to get rid of OpenCV in its headers. This makes OpenPose 1.6 headers different to previous versions and a bit harder to use. However, it allows OpenPose to be exported to other projects without requiring any third-party libraries (except in some special cases detailed below). The greatest benefit of this change: if your project already uses OpenCV, and you add your own version of OpenPose, the OpenCV version used in OpenPose and the one used in your project will not interfere with each other anymore, even if they are different versions!



### Third-Party Libraries
While compiling OpenPose from source, the static library files (`*.a` for Ubuntu, `*.lib` for Windows, etc.) and `include/` directories of all the third-party libraries detailed in [doc/installation.md](./installation.md) are required (GFlags, Glog, OpenCV, Caffe, etc.). However, when deploying OpenPose, fewer dependencies are required:
- GFLags and Glog are required only if the `include/openpose/flags.hpp` file is going to be used (e.g., when intenting to use the command-line interface).
- OpenCV can be optionally included if your project already uses it (but make sure to use the same binaries and include directory of OpenCV for both OpenPose and your project or weird runtime crashes will occur!). Including OpenCV does not increase the functionality of OpenPose, but it makes it easier to use by adding some functions that directly take cv::Mat matrices as input (rather than raw pointers). However, it is optional starting in OpenPose 1.6.0.
- Caffe or any other 3rd-party libraries are not required.

The static library files (`*.a` for Ubuntu, `*.lib` for Windows, etc.) and `include/` directories are the files that must be included in your project settings. However, the runtime library files (`*.so` for Ubuntu, `*.dll` for Windows, etc.), which are always required, must simply be placed together with the final executable or in default system paths. I.e., these files are only used during runtime, so they do not require any configuration in your project settings. E.g., for Windows, you can simply copy the content of the auto-generated `build/bin/` directory into the path where your executable is located.



### Private OpenPose Include Directory
Inside `include/`, there are 2 directories: `openpose/` and `openpose_private/`. Adding the `include_private` directory will require to include more libraries (e.g., OpenCV and Eigen). This directory exposes some extra functions used internally, but most of the cases this functionality is not required at all, so the `include/` directory should only contain the `openpose/` directory when exported.

Windows-only: In addition, Windows users have to manually add `OP_API` to all the functions/classes from `openpose_private/` that he desires to use and then re-compile OpenPose.



### Crash and Core Dumped Avoidance
If your project already uses OpenCV, and you add your own version of OpenPose, the OpenCV version of OpenPose and the one from your project will not interfere anymore, even if they are different versions. However, you cannot use the OpenCV functions of OpenPose from a different project if that project uses a different versions of OpenCV. Otherwise, very cryptic runtime DLL errors might occur! Make sure you either:
- Compile OpenPose and your project with the same version of OpenCV.
- Or if that is not possible (new since OpenPose 1.6.0), use the non-OpenCV analog functions of OpenPose to avoid cryptic DLL runtime crashes.



## Deploying OpenPose
### Windows
First of all, make sure to read all the sections above.

Second, note that the CMake option should also work for Windows. Alternatively, we also show the more Windows-like version in which `*.dll`, `*.lib`, and `include/` files are copied, which might be easier to apply when using the portable binaries.



### CMake (Windows, Ubuntu, and Mac)
First of all, make sure to read all the sections above.

If you only intend to use the OpenPose demo, you might skip this step. This step is only recommended if you plan to use the OpenPose API from other projects.

To install the OpenPose headers and libraries into the system environment path (e.g., `/usr/local/` or `/usr/`), run the following command.
```
cd build/
sudo make install
```

Once the installation is completed, you can use OpenPose in your other project using the `find_package` cmake command. Below, is a small example `CMakeLists.txt`. In order to use this script, you also need to copy `FindGFlags.cmake` and `FindGlog.cmake` into your `<project_root_directory>/cmake/Modules/` (create the directory if necessary).
```
cmake_minimum_required(VERSION 2.8.7)

add_definitions(-std=c++11)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

find_package(GFlags)
find_package(Glog)
find_package(OpenCV)
find_package(OpenPose REQUIRED)

include_directories(${OpenPose_INCLUDE_DIRS} ${GFLAGS_INCLUDE_DIR} ${GLOG_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

add_executable(example.bin example.cpp)

target_link_libraries(example.bin ${OpenPose_LIBS} ${GFLAGS_LIBRARY} ${GLOG_LIBRARY} ${OpenCV_LIBS})
```

If Caffe was built with OpenPose, it will automatically find it. Otherwise, you will need to link Caffe again as shown below (otherwise, you might get an error like `/usr/bin/ld: cannot find -lcaffe`).
```
link_directories(<path_to_caffe_installation>/caffe/build/install/lib)
```
