OpenPose Library - Standalone Face Or Hand Keypoint Detector
====================================

In case of hand camera views at which the hands are visible but not the rest of the body, or if you do not need the body keypoint detector and want to speed up the process, you can use the OpenPose face or hand keypoint detectors with your own face or hand detectors, rather than using the body keypoint detector as initial detector for those.

## OpenCV-based Face Keypoint Detector
Note that this method will be faster than the current system if there is few people in the image, but it is also much less accurate (OpenCV face detector only works with big and frontal faces, while OpenPose works with more scales and face rotations).
```
./build/examples/openpose/openpose.bin --body 0 --face --face_detector 1
```

## Custom Standalone Face or Hand Keypoint Detector
Check the examples in `examples/tutorial_api_cpp/`, in particular [examples/tutorial_api_cpp/06_face_from_image.cpp](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/examples/tutorial_api_cpp/06_face_from_image.cpp) and [examples/tutorial_api_cpp/07_hand_from_image.cpp](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/examples/tutorial_api_cpp/07_hand_from_image.cpp). The provide examples of face and/or hand keypoint detection given a known bounding box or rectangle for the face and/or hand locations. These examples are equivalent to use the following flags:
```
# Face
examples/tutorial_api_cpp/06_face_from_image.cpp --body 0 --face --face_detector 2
# Hands
examples/tutorial_api_cpp/07_hand_from_image.cpp --body 0 --hand --hand_detector 2
```

Note: both `FaceExtractor` and `HandExtractor` classes requires as input **squared rectangles**.

Advance solution: If you wanna use the whole OpenPose framework, you can use the synchronous examples of the `tutorial_api_cpp` folder with the configuration used for [examples/tutorial_api_cpp/06_face_from_image.cpp](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/examples/tutorial_api_cpp/06_face_from_image.cpp) and [examples/tutorial_api_cpp/07_hand_from_image.cpp](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/examples/tutorial_api_cpp/07_hand_from_image.cpp).
