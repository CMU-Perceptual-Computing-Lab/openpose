# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np

try:
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
    parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000294.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Load image
    imageToProcess = cv2.imread(args[0].image_path)

    def get_sample_heatmaps():
        # These parameters are globally set. You need to unset variables set here if you have a new OpenPose object. See *
        params = dict()
        params["model_folder"] = "../../../models/"
        params["heatmaps_add_parts"] = True
        params["heatmaps_add_bkg"] = True
        params["heatmaps_add_PAFs"] = True
        params["heatmaps_scale"] = 3
        params["upsampling_ratio"] = 1
        params["body"] = 1

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

        return poseHeatMaps

    # Get Heatmap
    poseHeatMaps = get_sample_heatmaps()

    # Starting OpenPose
    params = dict()
    params["model_folder"] = "../../../models/"
    params["body"] = 2  # Disable OP Network
    params["upsampling_ratio"] = 0 # * Unset this variable
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Pass Heatmap and Run OP
    datum = op.Datum()
    datum.cvInputData = imageToProcess
    datum.poseNetOutput = poseHeatMaps
    opWrapper.emplaceAndPop([datum])

    # Display Image
    print("Body keypoints: \n" + str(datum.poseKeypoints))
    cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
    cv2.waitKey(0)
except Exception as e:
    print(e)
    sys.exit(-1)
