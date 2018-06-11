OpenPose Python Module
=============================================

## Contents
1. [Introduction](#introduction)
2. [Testing and Installation](#testing-and-installation)


## Introduction
This experimental module exposes a Python API for OpenPose. This allows you to construct an OpenPose object, pass in a numpy array for an image, and get a numpy array of the pose positions. This API also exposes an API that allows you to directly pass in heatmaps from a network and extract poses out of it.

## Testing and Installation
To install the API so that it can be used globally, ensure that the `BUILD_PYTHON` flag is turned on, and run `make install` after compilation. This will install the python library at your desired installation path. (default is `/usr/local/python`) Ensure that this is in your python path in order to use it. 

Two examples can be found in `build/examples/tutorial_python` in your build folder. Navigate directly to this path to run examples. `1_extract_pose` demonstrates a simple use of the API. `2_pose_from_heatmaps` demonstrates constructing pose from heatmaps from the caffe network.

```python
import sys
import cv2
import os
# Remember to add your installation path here
sys.path.append('/usr/local/python') 
from openpose import *

# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled
params = dict() 
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "COCO"
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
img = cv2.imread("image.png")
arr, output_image = openpose.forward(img, True)
print arr

while 1:
    cv2.imshow("output", output_image)
    cv2.waitKey(15)

```



