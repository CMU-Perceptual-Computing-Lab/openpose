OpenPose Library - Standalone Face Or Hand Keypoint Detector
====================================

In case of hand camera views at which the hands are visible but not the rest of the body, or if you do not need the body keypoint detector and want to considerably speed up the process, you can use the OpenPose face or hand keypoint detectors with your own face or hand detectors, rather than using the body keypoint detector as initial detector for those.

## Standalone Face Keypoint Detector
Note that this method will be much faster than current system, but also much less accurate.
```
./build/examples/openpose/openpose.bin --face --body_disable
```

## Custom Standalone Face Keypoint Detector
There are 2 ways to add the OpenPose face keypoint detector to your own code without using the body pose keypoint extractor as initial face detector:

1. Easiest solution: Forget about the `OpenPose demo` and `wrapper/wrapper.hpp`, and instead use the `include/openpose/face/faceExtractorNet.hpp` class with the output of your face detector. Recommended if you do not wanna use any other OpenPose functionality.

2. Elegant solution: If you wanna use the whole OpenPose framework, simply copy `include/wrapper/wrapper.hpp` as e.g., `examples/userCode/wrapperFace.hpp`, and change our `FaceDetector` or `FaceDetectorOpenCV` class by your custom face detector class inside your `WrapperFace` class. If you wanna omit the Pose keypoint detection for a big speed up if you do not need it, you can simply use the `body_disable` flag.

Note: both `FaceExtractor` and `HandExtractor` classes requires as input **squared rectangles**. In addition, the function **`initializationOnThread()` must be called only once, and inside the same thread where `forwardPass` is gonna be run**.

## Custom Standalone Hand Keypoint Detector
The analogous steps apply to the hand keypoint detector, but modifying `include/openpose/hand/handExtractorNet.hpp`.
