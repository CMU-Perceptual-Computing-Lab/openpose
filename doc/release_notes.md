OpenPose Library - Release Notes
====================================



## OpenPose 1.0.0rc1 (Apr 24, 2017)
1. Initial version, main functionality:
    1. Body keypoint detection and rendering in Ubuntu 14 and 16.
    2. It can read an image directory, video or webcam.
    3. It can display the results or storing them on disk.



## OpenPose 1.0.0rc2 (May 10, 2017)
1. Main improvements:
    1. Rendering max resolution from 720p to >32k images.
    2. Highly improved documentation.
2. Functions or parameters renamed:
    1. Demo renamed from rtpose to openpose.
3. Main bugs fixed:
    1. Demo uses exec instead of start, so it works with more OpenCV custom compiled versions.



## OpenPose 1.0.0rc3 (Jun 8, 2017)
1. Main improvements:
    1. Added face keypoint detection.
    2. Added Windows 10 compatibility.
    3. Auto-detection of the number of GPUs.
    4. MPI visualization more similar to COCO one.
    5. Rendering max resolution from 720p to >32k images.
    6. GUI info adder working when the worker TDatum has more than 1 Datum.
    7. It prints out the error description before throwing the exception (so that it is written on the Windows cmd).
    8. Highly improved documentation.
2. Functions or parameters renamed:
    1. Flag `write_pose` renamed as `write_keypoint` and it also applies to face and/or hands.
    2. Flag `write_pose_json` renamed as `write_keypoint_json` and it also applies to face and/or hands.
    3. Flag `write_pose_format` renamed as `write_keypoint_format` and it also applies to face and/or hands.
    4. PoseSaver and its JSON variant renamed as KeypointSaver.
    5. PoseJsonCocoSaver renamed as CocoJsonSaver.
3. Main bugs fixed:
    1. All visualization functions moved to same thread, so it works with most OpenCV custom compiled versions.
    2. Fixed error on debug mode: `Too many resources requested for launch`.



## OpenPose 1.0.0 (Jul 8, 2017)
1. Main improvements:
    1. Added hand keypoint detection.
    2. Windows branch merged to master branch.
    3. Face and hands use `Maximum` instead of `Nms`, since there is only 1 person / detection.
    4. Increased accuracy on multi-scale (added `Datum::scaleRatios` to save the relative scale ratio when multi-scale).
    5. Increased speed ~5% by adding CPU rendering (but GPU is the default rendering).
    6. Rendering colors modified, visually better results.
    7. Rendering threshold for pose, face and hands becomes user-configurable.
    8. Check() functions give more feedback.
    9. WCocoJsonSaver finished and removed its 3599-image limit.
    10. Added `camera_fps` so generated video will use that frame rate.
    11. Reduced the number of printed information messages. Default logging priority threshold increased to Priority::Max.
    12. Google flags to OpenPose configuration parameters reader moved from each demo to utilities/flagsToOpenPose.
    13. Nms classes do not use `numberParts` for `Reshape`, they deduce the value.
    14. Improved documentation.
2. Functions or parameters renamed:
    1. Render flags renamed in the demo in order to incorporate the CPU/GPU rendering.
    2. Keypoints saved in JSON files (`write_keypoint_json`) are now saved as `pose_keypoints`, `face_keypoints`, `hand_left_keypoints`, and `hand_right_keypoints`. They all were previously saved as `body_parts`.
    3. Flag `num_scales` renamed as `scale_number`.
    4. All hand and pose flags renamed such as they start by `hand_` and `face_` respectively.
3. Main bugs fixed:
    1. Fixed bug in Array::getConstCvMat() if mVolume=0, now returning empty cv::Mat.
    2. Fixed bug: `--process_real_time` threw error with webcam.
    3. Fixed bug: Face not working when input and output resolutions are different.
    4. Fixed some bugs that prevented debug version to run.
    5. Face saved in JSON files were called `body_parts`. Now they are called `face_keypoints`.



## OpenPose 1.0.1 (Jul 11, 2017)
1. Main improvements:
    1. Windows library turned into DLL dynamic library (i.e. portable).
    2. Improved documentation.
2. Functions or parameters renamed:
    1. `openpose/utilities/macros.hpp` moved to `openpose/utilities/macros.hpp`.



## OpenPose 1.0.2 (Sep 3, 2017)
1. Main improvements:
    1. Added OpenCV 3.3 compatibility.
    2. Caffe turned into DLL library.
    3. OpenPose is now completely portable across Windows 10 computers (with Nvidia graphic card).
    4. Added OpenPose 1.0.1 portable demo.
    5. Removed Python and some unnecessary boost dependencies on the VS project.
    6. Replaced all double quotes by angle brackets in include statements (issue #61).
    7. Added 3-D reconstruction demo.
    8. Auto-detection of the camera index.
    9. Speed up of ~30% in op::floatPtrToUCharCvMat.
    10. COCO extractor now extracts image ID from the image name itslef (format "string_%d"). Before, only working with validation test, now applicable to e.g. test sets.
    11. Changed display texts, added `OpenPose` name.
2. Main bugs fixed:
    1. Pycaffe can now be imported from Python.
    2. Fixed `Tutorial/Wrapper` VS linking errors.



## OpenPose 1.1.0 (Sep 19, 2017)
1. Main improvements:
    1. Added CMake installer for Ubuntu.
    2. Added how to use keypoint data in `examples/tutorial_wrapper/`.
    3. Added flag for warnings of type `-Wsign-compare` and removed in code.
    4. Slightly improved accuracy by considering ears-shoulder connection (e.g. +0.4 mAP for 1 scale in validation set).
2. Main bugs fixed:
    1. Windows version crashing with std::map copy.



## OpenPose 1.2.0 (Nov 3, 2017)
1. Main improvements:
    1. Speed increase when processing images with different aspect ratios. E.g. ~20% increase over 3.7k COCO validation images on 1 scale.
    2. Huge speed increase and memory reduction when processing multi-scale. E.g. over 3.7k COCO validation images on 4 scales: ~40% (~770 to ~450 sec) speed increase, ~25% memory reduction (from ~8.9 to ~6.7 GB / GPU).
    3. Slightly increase of accuracy given the fixed mini-bugs.
    4. Added IP camera support.
    5. Output images can have the input size, OpenPose able to change its size for each image and not required fixed size anymore.
        1. FrameDisplayer accepts variable size images by rescaling every time a frame with bigger width or height is displayed (gui module).
        2. OpOutputToCvMat & GuiInfoAdder does not require to know the output size at construction time, deduced from each image.
        3. CvMatToOutput and Renderers allow to keep input resolution as output for images (core module).
    6. New standalone face keypoint detector based on OpenCV face detector: much faster if body keypoint detection is not required but much less accurate.
    7. Face and hand keypoint detectors now can return each keypoint heatmap.
    8. The flag `USE_CUDNN` is no longer required; `USE_CAFFE` and `USE_CUDA` (replacing the old `CPU_ONLY`) are no longer required to use the library, only to build it. In addition, Boost, Caffe, and its dependencies have been removed from the OpenPose header files. Only OpenCV include and lib folders are required when building a project using OpenPose.
    9. OpenPose successfully compiles if the flags `USE_CAFFE` and/or `USE_CUDA` are not enabled, although it will give an error saying they are required.
    10. COCO JSON file outputs 0 as score for non-detected keypoints.
    11. Added example for OpenPose for user asynchronous output and cleaned all `tutorial_wrapper/` examples.
    12. Added `-1` option for `net_resolution` in order to auto-select the best possible aspect ratio given the user input.
    13. Net resolution can be dynamically changed (e.g. for images with different size).
    14. Added example to add functionality/modules to OpenPose.
    15. Added `disable_multi_thread` flag in order to allow debug and/or highly reduce the latency (e.g. when using webcam in real-time).
    16. Allowed to output images without any rendering.
2. Functions or parameters renamed:
    1. OpenPose able to change its size and initial size dynamically:
        1. Flag `resolution` renamed as `output_resolution`.
        2. FrameDisplayer, GuiInfoAdder and Gui constructors arguments modified (gui module).
        3. OpOutputToCvMat constructor removed (core module).
        4. New Renders classes to split GpuRenderers from CpuRenderers.
        5. Etc.
    2. OpenPose able to change its net resolution size dynamically:
        1. Changed several functions on `core/`, `pose/`, `face/`, and `hand/` modules.
    3. `CPU_ONLY` changed by `USE_CUDA` to keep format.
3. Main bugs fixed:
    1. Scaling resize issue fixed: ~1-pixel offset due to not considering 0-based indexes.
    2. Ubuntu installer script now works even if Python pip was not installed previously.
    3. Flags to set first and last frame as well as jumping frames backward and forward now works on image directory reader.



## Current version (future OpenPose 1.2.1)
