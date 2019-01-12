OpenPose Python Module and Demo
=============================================

## Contents
1. [Introduction](#introduction)
2. [Compatibility](#compatibility)
3. [Installation](#installation)
4. [Testing](#testing)
5. [Exporting Python OpenPose](#exporting-python-openpose)



## Introduction
This module exposes a Python API for OpenPose. It is effectively a wrapper that replicates most of the functionality of the [op::Wrapper class](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/wrapper/wrapper.hpp) and allows you to populate and retrieve data from the [op::Datum class](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/core/datum.hpp) using standard Python and Numpy constructs.



## Compatibility
The OpenPose Python module is compatible with both Python 2 and Python 3. In addition, it will also run in all OpenPose compatible operating systems. It uses [Pybind11](https://github.com/pybind/pybind11) for mapping between C++ and Python datatypes.

To compile, enable `BUILD_PYTHON` in cmake. Pybind selects the latest version of Python by default (Python 3). To use Python 2, change `PYTHON_EXECUTABLE` and `PYTHON_LIBRARY` flags in cmake to your desired python version.



## Installation
Check [doc/installation.md#python-module](../installation.md#python-api) for installation steps.

The Python API requires Numpy for array management, and OpenCV for image loading. They can be installed via:

```
# Python 2
sudo pip install numpy opencv-python
# Python 3 (recommended)
sudo pip3 install numpy opencv-python
```



## Testing
All the Python examples from the Tutorial API Python module can be found in `build/examples/tutorial_api_python` in your build folder. Navigate directly to this path to run examples.

```
# From command line
cd build/examples/tutorial_api_python

# Python 2
python2 1_body_from_image.py
python2 2_whole_body_from_image.py
# python2 [any_other_example.py]

# Python 3 (recommended)
python3 1_body_from_image.py
python3 2_whole_body_from_image.py
# python3 [any_other_example.py]
```



## Exporting Python OpenPose
Note: This step is only required if you are moving the `*.py` files outside their original location, or writting new `*.py` scripts outside `build/examples/tutorial_api_python`.

- Option a, installing OpenPose: On an Ubuntu or OSX based system, you could install OpenPose by running `sudo make install`, you could then set the OpenPose path in your python scripts to the OpenPose installation path (default: `/usr/local/python`) and start using OpenPose at any location. Take a look at `build/examples/tutorial_pose/1_body_from_image.py` for an example.
- Option b, not installing OpenPose: To move the OpenPose Python API demos to a different folder, ensure that the line `sys.path.append('{OpenPose_path}/python')` is properly set in your `*.py` files, where `{OpenPose_path}` points to your build folder of OpenPose. Take a look at `build/examples/tutorial_pose/1_body_from_image.py` for an example.
