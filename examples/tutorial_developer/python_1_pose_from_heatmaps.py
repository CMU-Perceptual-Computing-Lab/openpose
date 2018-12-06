from sys import platform
import sys
try:
    import caffe
except ImportError:
    print("This sample can only be run if Python Caffe if available on your system")
    print("Currently OpenPose does not compile Python Caffe. This may be supported in the future")
    sys.exit(-1)

import os
os.environ["GLOG_minloglevel"] = "1"
import caffe
import cv2
import numpy as np
import sys
import time
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../../python')
dir_path + "/../../models/"
try:
    from openpose import OpenPose
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')

# Params for change
# Single-scale
defRes = 368
scales = [1]
# # Multi-scale
# defRes = 736
# scales = [1, 0.75, 0.5, 0.25]
class Param:
    caffemodel = dir_path + "/../../../models/pose/body_25/pose_iter_584000.caffemodel"
    prototxt = dir_path + "/../../../models/pose/body_25/pose_deploy.prototxt"

# Load OpenPose object and Caffe Nets
params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x"+str(defRes)
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.25
params["scale_number"] = len(scales)
params["render_threshold"] = 0.05
params["num_gpu_start"] = 0
params["disable_blending"] = False
params["default_model_folder"] = dir_path + "/../../../models/"
openpose = OpenPose(params)
caffe.set_mode_gpu()
caffe.set_device(0)
nets = []
for scale in scales:
    nets.append(caffe.Net(Param.prototxt, Param.caffemodel, caffe.TEST))
print("Net loaded")

# Test Function
first_run = True
def func(frame):

    # Get image processed for network, and scaled image
    imagesForNet, imagesOrig = OpenPose.process_frames(frame, defRes, scales)

    # Reshape
    global first_run
    if first_run:
        for i in range(0, len(scales)):
            net = nets[i]
            imageForNet = imagesForNet[i]
            in_shape = net.blobs['image'].data.shape
            in_shape = (1, 3, imageForNet.shape[1], imageForNet.shape[2])
            net.blobs['image'].reshape(*in_shape)
            net.reshape()

        first_run = False
        print("Reshaped")

    # Forward pass to get heatmaps
    heatmaps = []
    for i in range(0, len(scales)):
        net = nets[i]
        imageForNet = imagesForNet[i]
        net.blobs['image'].data[0,:,:,:] = imageForNet
        net.forward()
        heatmaps.append(net.blobs['net_output'].data[:,:,:,:])

    # Pose from HM Test
    array, frame = openpose.poseFromHM(frame, heatmaps, scales)

    # Draw Heatmaps instead
    #hm = heatmaps[0][:,0:18,:,:]; frame = OpenPose.draw_all(imagesOrig[0], hm, -1, 1, True)
    #paf = heatmaps[0][:,20:,:,:]; frame = OpenPose.draw_all(imagesOrig[0], paf, -1, 4, False)

    return frame


img = cv2.imread(dir_path + "/../../../examples/media/COCO_val2014_000000000192.jpg")
frame = func(img)
while 1:
    cv2.imshow("output", frame)
    cv2.waitKey(15)
