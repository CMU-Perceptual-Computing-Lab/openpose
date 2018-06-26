OpenPose Python Module
=============================================

## Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Compatibility](#compatibility)
4. [Testing](#testing)


## Introduction
This experimental module exposes a Python API for OpenPose. This allows you to construct an OpenPose object, pass in a numpy array for an image, and get a numpy array of the pose positions. This API also exposes an API that allows you to directly pass in heatmaps from a network and extract poses out of it (Requires Python Caffe to be installed seperately)

At present the Python API only supports body pose. Hands and Face will be added in the future.

## Installation
Check [doc/installation.md#python-module](./installation.md#python-module) for installation steps.

To simply test the OpenPose API in your project without installation, ensure that the line `sys.path.append('{OpenPose_path}/python')` is set in your *.py files, where `{OpenPose_path}` points to your build folder of OpenPose. Take a look at `build/examples/tutorial_pose/1_extract_pose.py` for an example.

On an Ubuntu or OSX based system, you may use it globally. Running `sudo make install` will install OpenPose by default into `/usr/local/python`. You can set this into your python path and start using it at any location.

The Python API requires Numpy for array management, and OpenCV for image loading. They can be installed via:

```
pip install numpy
pip install opencv-python
```

## Compatibility
The OpenPose Python module is compatible with both Python 2 and Python 3. In addition, it will also run in all OpenPose compatible operating systems.



## Testing
Two examples can be found in `build/examples/tutorial_python` in your build folder. Navigate directly to this path to run examples.

- `1_extract_pose` demonstrates a simple use of the API.
- `2_pose_from_heatmaps` demonstrates constructing pose from heatmaps from the caffe network. (Requires Python Caffe to be installed seperately)

```
# From command line
cd build/examples/tutorial_python
python 1_extract_pose.py
```



## Code Sample
See `examples/tutorial_python/1_extract_pose.py`.
