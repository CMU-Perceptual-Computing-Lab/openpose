import sys
import cv2
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../../python')
from openpose import *

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
params["disable_blending"] = False
params["default_model_folder"] = dir_path + "/../../../models/"
openpose = OpenPose(params)
img = cv2.imread(dir_path + "/../../../examples/media/COCO_val2014_000000000192.jpg")
arr, output_image = openpose.forward(img, True)
print arr

while 1:
    cv2.imshow("output", output_image)
    cv2.waitKey(15)
