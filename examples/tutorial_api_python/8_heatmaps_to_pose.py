# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()

# Load image
imageToProcess = cv2.imread(args[0].image_path)

def get_sample_heatmaps():
    params = dict()
    params["model_folder"] = "../../../models/"
    params["heatmaps_add_parts"] = True
    params["heatmaps_add_bkg"] = True
    params["heatmaps_add_PAFs"] = True
    params["heatmaps_scale"] = 3

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image and get heatmap
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])
    poseHeatMaps = datum.poseHeatMaps.copy()
    opWrapper.stop()

    # Convert to OP Format with Stride 8
    poseHeatMapsResized = np.zeros((poseHeatMaps.shape[0], int(poseHeatMaps.shape[1]/8), int(poseHeatMaps.shape[2]/8)), dtype=np.float32)
    print(poseHeatMapsResized.shape)
    for i in range(0, poseHeatMaps.shape[0]):
        poseHeatMapsResized[i,:,:] = cv2.resize(poseHeatMaps[i,:,:], (int(poseHeatMaps.shape[2]/8),int(poseHeatMaps.shape[1]/8)))
    poseHeatMaps = poseHeatMapsResized
    poseHeatMaps = poseHeatMaps.reshape((1, poseHeatMaps.shape[0], poseHeatMaps.shape[1], poseHeatMaps.shape[2]))
    print(poseHeatMaps.shape)
    return poseHeatMaps

# Get Heatmap
poseHeatMaps = get_sample_heatmaps()

# Starting OpenPose
params = dict()
params["model_folder"] = "../../../models/"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

datum = op.Datum()
datum.cvInputData = imageToProcess
datum.poseNetOutput = poseHeatMaps
opWrapper.emplaceAndPop([datum])

# Display Image
print("Body keypoints: \n" + str(datum.poseKeypoints))
cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)
cv2.waitKey(0)
