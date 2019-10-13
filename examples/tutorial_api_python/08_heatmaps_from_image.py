# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse

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
    parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"
    params["heatmaps_add_parts"] = True
    params["heatmaps_add_bkg"] = True
    params["heatmaps_add_PAFs"] = True
    params["heatmaps_scale"] = 2

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    # Process outputs
    outputImageF = (datum.inputNetData[0].copy())[0,:,:,:] + 0.5
    outputImageF = cv2.merge([outputImageF[0,:,:], outputImageF[1,:,:], outputImageF[2,:,:]])
    outputImageF = (outputImageF*255.).astype(dtype='uint8')
    heatmaps = datum.poseHeatMaps.copy()
    heatmaps = (heatmaps).astype(dtype='uint8')

    # Display Image
    counter = 0
    while 1:
        num_maps = heatmaps.shape[0]
        heatmap = heatmaps[counter, :, :].copy()
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        combined = cv2.addWeighted(outputImageF, 0.5, heatmap, 0.5, 0)
        cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", combined)
        key = cv2.waitKey(-1)
        if key == 27:
            break
        counter += 1
        counter = counter % num_maps
except Exception as e:
    print(e)
    sys.exit(-1)
