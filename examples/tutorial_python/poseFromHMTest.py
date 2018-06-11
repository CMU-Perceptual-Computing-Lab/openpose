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
from openpose import OpenPose

# Params
class Param_a:
    caffemodel = dir_path + "/../../../models/pose/coco/pose_iter_440000.caffemodel"
    prototxt = dir_path + "/../../../models/pose/coco/pose_deploy_linevec.prototxt"
    boxsize = 368
    padValue = 0

# Params
class Param_b:
    caffemodel = dir_path + "/../../../models/pose/coco/pose_iter_440000.caffemodel"
    prototxt = dir_path + "/../../../models/pose/coco/pose_deploy_linevec.prototxt"
    boxsize = 368/2
    padValue = 0

# Load net
params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "COCO"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.5
params["scale_number"] = 2
params["render_threshold"] = 0.05
params["num_gpu_start"] = 0
params["disable_blending"] = False
params["default_model_folder"] = dir_path + "/../../../models/"
openpose = OpenPose(params)
caffe.set_mode_gpu()
caffe.set_device(0)
net_a = caffe.Net(Param_a.prototxt, Param_a.caffemodel, caffe.TEST)
net_b = caffe.Net(Param_b.prototxt, Param_b.caffemodel, caffe.TEST)
print "Net loaded"

currIndex = 0
first_run = True
def func(frame):
    # Reshape
    #height, width, channels = frame.shape

    rframe_a, imageForNet_a, padding_a = OpenPose.process_frame(frame, Param_a.boxsize, Param_a.padValue)
    rframe_b, imageForNet_b, padding_b = OpenPose.process_frame(frame, Param_b.boxsize, Param_b.padValue)

    global first_run
    if first_run:
        in_shape = net_a.blobs['image'].data.shape
        in_shape = (1, 3, imageForNet_a.shape[1], imageForNet_a.shape[2])
        net_a.blobs['image'].reshape(*in_shape)
        net_a.reshape()

        in_shape = net_b.blobs['image'].data.shape
        in_shape = (1, 3, imageForNet_b.shape[1], imageForNet_b.shape[2])
        net_b.blobs['image'].reshape(*in_shape)
        net_b.reshape()

        first_run = False
        print "Reshaped"

    net_a.blobs['image'].data[0,:,:,:] = imageForNet_a
    net_a.forward()
    heatmaps_a = net_a.blobs['net_output'].data[:,:,:,:]

    net_b.blobs['image'].data[0,:,:,:] = imageForNet_b
    net_b.forward()
    heatmaps_b = net_b.blobs['net_output'].data[:,:,:,:]

    # Pose from HM Test
    array, frame = openpose.poseFromHM(frame, [heatmaps_a, heatmaps_b], [1,0.5])
    #array, frame = openpose.poseFromHM(frame, [heatmaps_a], [2])

    return frame


img = cv2.imread(dir_path + "/../../../examples/media/COCO_val2014_000000000192.jpg")
frame = func(img)
while 1:
    cv2.imshow("output", frame)
    cv2.waitKey(15)
