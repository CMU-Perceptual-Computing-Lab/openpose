OpenPose Python Module
=============================================

## Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Compatibility](#compatibility)
4. [Testing](#testing)


## Introduction
This experimental module exposes a Python API for OpenPose. This allows you to construct an OpenPose object, pass in a numpy array for an image, and get a numpy array of the pose positions. This API also exposes an API that allows you to directly pass in heatmaps from a network and extract poses out of it.



## Installation
Check [doc/installation.md#python-module](./installation.md#python-module) for installation steps.



## Compatibility
The OpenPose Python module is compatible with both Python 2 and Python 3. In addition, it will also run in all OpenPose compatible operating systems.



## Testing
Two examples can be found in `build/examples/tutorial_python` in your build folder. Navigate directly to this path to run examples.

    - `1_extract_pose` demonstrates a simple use of the API.
    - `2_pose_from_heatmaps` demonstrates constructing pose from heatmaps from the caffe network.

```
# From command line
cd build/examples/tutorial_python
python
```

```python
# From Python
# It requires OpenCV installed for Python
import cv2
import os
import sys

# Remember to add your installation path here
# Option a
sys.path.append('{OpenPose_path}/python')
# Option b
# If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
# sys.path.append('/usr/local/python')

from openpose import *

# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled
params = dict() 
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
params["num_gpu_start"] = 0 
# If GPU version is built, and multiple GPUs are available, set the ID here
params["disable_blending"] = False
params["default_model_folder"] = "/home/user/openpose/models"
# Construct OpenPose object allocates GPU memory
openpose = OpenPose(params)

while 1:
    # Read new image
    img = cv2.imread("image.png")
    # Output keypoints and the image with the human skeleton blended on it
    keypoints, output_image = openpose.forward(img, True)
    # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
    print keypoints
    # Display the image
    cv2.imshow("output", output_image)
    cv2.waitKey(15)
```
