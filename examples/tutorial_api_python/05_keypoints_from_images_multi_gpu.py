# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time

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
    parser.add_argument("--image_dir", default="../../../examples/media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    parser.add_argument("--num_gpu", default=op.get_gpu_number(), help="Number of GPUs.")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"
    params["num_gpu"] = int(vars(args[0])["num_gpu"])
    numberGPUs = int(params["num_gpu"])

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

    # Read frames on directory
    imagePaths = op.get_images_on_directory(args[0].image_dir);

    # Read number of GPUs in your system
    start = time.time()

    # Process and display images
    for imageBaseId in range(0, len(imagePaths), numberGPUs):

        # Create datums
        datums = []
        images = []

        # Read and push images into OpenPose wrapper
        for gpuId in range(0, numberGPUs):

            imageId = imageBaseId+gpuId
            if imageId < len(imagePaths):

                imagePath = imagePaths[imageBaseId+gpuId]
                datum = op.Datum()
                images.append(cv2.imread(imagePath))
                datum.cvInputData = images[-1]
                datums.append(datum)
                opWrapper.waitAndEmplace([datums[-1]])

        # Retrieve processed results from OpenPose wrapper
        for gpuId in range(0, numberGPUs):

            imageId = imageBaseId+gpuId
            if imageId < len(imagePaths):

                datum = datums[gpuId]
                opWrapper.waitAndPop([datum])

                print("Body keypoints: \n" + str(datum.poseKeypoints))

                if not args[0].no_display:
                    cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
                    key = cv2.waitKey(15)
                    if key == 27: break

    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
except Exception as e:
    print(e)
    sys.exit(-1)
