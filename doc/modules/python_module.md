OpenPose Python Module and Demo
=============================================

## Contents
1. [Introduction](#introduction)
2. [Compatibility](#compatibility)
3. [Installation](#installation)
4. [Testing](#testing)
5. [Exporting Python OpenPose](#exporting-python-openpose)



## Introduction
This experimental module exposes a Python API for OpenPose. This allows you to construct an OpenPose object, pass in a numpy array for an image, and get a numpy array of the pose positions. This API also exposes an API that allows you to directly pass in heatmaps from a network and extract poses out of it (Requires Python Caffe to be installed seperately)

At present the Python API only supports body pose. Hands and Face will be added in the future.



## Compatibility
The OpenPose Python module is compatible with both Python 2 and Python 3. In addition, it will also run in all OpenPose compatible operating systems.



## Installation
Check [doc/installation.md#python-module](../installation.md#python-api) for installation steps.

The Python API requires Numpy for array management, and OpenCV for image loading. They can be installed via:

```
pip install numpy
pip install opencv-python
```



## Testing
Two examples can be found in `build/examples/tutorial_api_python` in your build folder. Navigate directly to this path to run examples.

- `1_extract_pose` demonstrates a simple use of the API.
- `2_pose_from_heatmaps` demonstrates constructing pose from heatmaps from the caffe network (Requires Python Caffe to be installed seperately, only tested on Ubuntu).

```
# From command line
cd build/examples/tutorial_api_python
python 1_extract_pose.py
```



## Exporting Python OpenPose
Note: This step is only required if you are moving the `*.py` files outside their original location, or writting new `*.py` scripts outside `build/examples/tutorial_api_python`.

- Option a, installing OpenPose: On an Ubuntu or OSX based system, you could install OpenPose by running `sudo make install`, you could then set the OpenPose path in your python scripts to the OpenPose installation path (default: `/usr/local/python`) and start using OpenPose at any location. Take a look at `build/examples/tutorial_pose/1_extract_pose.py` for an example.
- Option b, not installing OpenPose: To move the OpenPose Python API demos to a different folder, ensure that the line `sys.path.append('{OpenPose_path}/python')` is properly set in your `*.py` files, where `{OpenPose_path}` points to your build folder of OpenPose. Take a look at `build/examples/tutorial_pose/1_extract_pose.py` for an example.
