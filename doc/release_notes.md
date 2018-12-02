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
    1. Flag `--write_pose` renamed as `--write_keypoint` and it also applies to face and/or hands.
    2. Flag `--write_pose_json` renamed as `--write_keypoint_json` and it also applies to face and/or hands.
    3. Flag `--write_pose_format` renamed as `--write_keypoint_format` and it also applies to face and/or hands.
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
    10. Added `--camera_fps` so generated video (`--write_video`) will use that frame rate.
    11. Reduced the number of printed information messages. Default logging priority threshold increased to Priority::Max.
    12. GFlags to OpenPose configuration parameters reader moved from each demo to utilities/flagsToOpenPose.
    13. Nms classes do not use `numberParts` for `Reshape`, they deduce the value.
    14. Improved documentation.
2. Functions or parameters renamed:
    1. Render flags renamed in the demo in order to incorporate the CPU/GPU rendering.
    2. Keypoints saved in JSON files (`--write_keypoint_json`) are now saved as `pose_keypoints`, `face_keypoints`, `hand_left_keypoints`, and `hand_right_keypoints`. They all were previously saved as `body_parts`.
    3. Flag `--num_scales` renamed as `--scale_number`.
    4. All hand and pose flags renamed such as they start by `--hand_` and `--face_` respectively.
3. Main bugs fixed:
    1. Fixed bug in Array::getConstCvMat() if mVolume=0, now returning empty cv::Mat.
    2. Fixed bug: `--process_real_time` threw error with webcam.
    3. Fixed bug: Face not working when input and output resolutions are different.
    4. Fixed some bugs that prevented debug version to run.
    5. Face saved in JSON files were called `--body_parts`. Now they are called `--face_keypoints`.



## OpenPose 1.0.1 (Jul 11, 2017)
1. Main improvements:
    1. Windows library turned into DLL dynamic library (i.e., portable).
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
    10. COCO extractor now extracts image ID from the image name itslef (format "string_%d"). Before, only working with validation test, now applicable to e.g., test sets.
    11. Changed display texts, added `OpenPose` name.
2. Main bugs fixed:
    1. Pycaffe can now be imported from Python.
    2. Fixed `Tutorial/Wrapper` VS linking errors.



## OpenPose 1.1.0 (Sep 19, 2017)
1. Main improvements:
    1. Added CMake installer for Ubuntu.
    2. Added how to use keypoint data in `examples/tutorial_wrapper/`.
    3. Added flag for warnings of type `-Wsign-compare` and removed in code.
    4. Slightly improved accuracy by considering ears-shoulder connection (e.g., +0.4 mAP for 1 scale in validation set).
2. Main bugs fixed:
    1. Windows version crashing with std::map copy.



## OpenPose 1.2.0 (Nov 3, 2017)
1. Main improvements:
    1. Speed increase when processing images with different aspect ratios. E.g., ~20% increase over 3.7k COCO validation images on 1 scale.
    2. Huge speed increase and memory reduction when processing multi-scale. E.g., over 3.7k COCO validation images on 4 scales: ~40% (~770 to ~450 sec) speed increase, ~25% memory reduction (from ~8.9 to ~6.7 GB / GPU).
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
    12. Added `-1` option for `--net_resolution` in order to auto-select the best possible aspect ratio given the user input.
    13. Net resolution can be dynamically changed (e.g., for images with different size).
    14. Added example to add functionality/modules to OpenPose.
    15. Added `--disable_multi_thread` flag in order to allow debug and/or highly reduce the latency (e.g., when using webcam in real-time).
    16. Allowed to output images without any rendering.
2. Functions or parameters renamed:
    1. OpenPose able to change its size and initial size dynamically:
        1. Flag `--resolution` renamed as `--output_resolution`.
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



## OpenPose 1.2.1 (Jan 9, 2018)
1. Main improvements:
    1. Heatmaps can be saved in floating format.
    2. More efficient non-processing version (i.e., if all keypoint extractors are disabled, and only image extraction and display/saving operations are performed).
    3. Heat maps scaling: Added `--heatmaps_scale` to OpenPoseDemo, added option not to scale the heatmaps, and added custom `float` format to save heatmaps in floating format.
    4. Detector of the number of GPU also considers the initial GPU index given by the user.
    5. Added `--write_json` as new version of `--write_keypoint_json`. It includes the body part candidates (if enabled), as well as any extra information added in the future (e.g., person ID).
    6. Body part candidates can be retrieved in op::Datum and saved with `--write_json`.
2. Functions or parameters renamed:
    1. `PoseParameters` splitted into `PoseParameters` and `PoseParametersRender` and const parameters turned into functions for more clarity.
3. Main bugs fixed:
    1. Render working on images > 4K (#324).
    2. Cleaned redundant arguments on `getAverageScore` and `getKeypointsArea`.
    3. Slight speed up when heatmaps must be returned to the user (not doing a double copy anymore).



## OpenPose 1.3.0 (Mar 24, 2018)
1. Main improvements:
    1. Output of `--write_json` uses less hard disk space (enters and tabs removed).
    2. Removed Boost dependencies.
    3. Caffe added as submodule.
    4. CMake installer compatible with Windows.
    5. Added freeglut download script (3-D reconstruction demo for Windows).
    6. Added Debug version for Windows (CMake).
    7. Runtime verbose about average speed configurable by user with `PROFILER_ENABLED` option (CMake/Makefile.config) and `--profile_speed` flag.
    8. Lighter Caffe version compiled by CMake in Ubuntu: disabled Caffe extra support (e.g., OpenCV, Python) and doc.
    9. Renamed CMake binaries (Ubuntu) to match old Makefile format: `_bin` by `.bin`.
    10. 3-D reconstruction demo cleaned, implemented in Ubuntu too, and now defined as module of OpenPose rather than just a demo.
    11. CMake as default installer in documentation.
    12. Added flag: number_people_max to optionally select the maximum number of people to be detected.
    13. 3-D reconstruction module forces the user to set `number_people_max 1` to avoid errors (as it assumes only 1 person per image).
    14. Removed old `windows/` version. CMake is the only Windows version available.
    15. Camera parameters (flir camera) are read from disk at runtime rather than being compiled.
    16. 3-D reconstruction module can be implemented with different camera brands or custom image sources.
    17. Flag `--write_json` includes 3-D keypoints.
    18. 3-D reconstruction module can be used with images and videos. Flag `--3d_views` added to allow `--image_dir` and `--video` allow loading stereo images.
    19. Flag `--camera_resolution` applicable to `--flir_camera`.
    20. Throw error message if requested GPU IDs does not exist (e.g., asking for 2 GPUs starting in ID 1 if there is only 2 GPUs in total).
    21. VideoSaver (`--write_video`) compatible with multi-camera setting. It will save all the different views concatenated.
    22. OpenPose small GUI rescale the verbose text to the displayed image, to avoid the text to be either too big or small.
    23. OpenPose small GUI shows the frame number w.r.t. the original producer, rather than the frame id. E.g., if video is started at frame 30, OpenPose will display 30 rather than 0 in the first frame.
    24. OpenPose GUI: 'l' and 'k' functionality swapped.
    25. 3-D reconstruction module: Added flag `--3d_min_views` to select minimum number of cameras required for 3-D reconstruction.
    26. Flir camera producer `n` times faster for `n` cameras (multi-threaded). If the number of cameras is greater than the number of the computer threads, the speed up might not be exactly `n` times.
2. Functions or parameters renamed:
    1. Flag `no_display` renamed as `display`, able to select between `NoDisplay`, `Display2D`, `Display3D`, and `DisplayAll`.
    2. 3-D reconstruction demo is now inside the OpenPose demo binary.
    3. Renamed `*_keypoints` by `*_keypoints_2d` to avoid confusion with 3d ones in `--write_json` output file.
    4. CvMatToOpInput requires PoseModel to know the normalization to be performed.
    5. Created `net/` module in order to reduce `core/` number of classes and files and for future scalability.
3. Main bugs fixed:
    1. Slight speed up (~1%) for performing the non-maximum suppression stage only in the body part heatmaps channels, and not also in the PAF channels.
    2. Fixed core-dumped in PoseRenderer with GUI when changed element to be rendered to something else than skeleton.
    3. 3-D visualizer does not crash on exit anymore.
    4. Fake pause ('m' key pressed) works again.



## OpenPose 1.4.0 (Sep 01, 2018)
1. Main improvements:
    1. Model BODY_25 released, that includes the 17 COCO keypoints + neck + midhip + 6 foot keypoints. It is also about 3% more accurate and 30% faster than the original `COCO` model.
    2. New calibration module: Intrinsic and extrinsic camera calibration toolbox based on OpenCV.
    3. Improvements involving Flir cameras:
        1. Added software trigger and a dedicated thread to keep reading images so latency is removed and runtime is faster (analogously to webcamReader).
        2. Undistortion of the images is x3.5 faster per camera, i.e., x3.5 Flir camera producer reading w.r.t previous multi-threaded version, which was x number_cameras faster than the original version.
        3. Added flag `flir_camera_index` to allow running on all the cameras at once, or only on 1 camera at the time.
        4. Added flag `frame_keep_distortion` not to undistort the images. E.g., useful when recording images for camera calibration.
        5. Changed Spinnaker::DEFAULT image extraction mode by Spinnaker::IPP, which does not show a pixelated image while keeping very similar runtime.
    4. 3-D reconstruction:
        1. Added non-linear minimization to further improve 3-D triangulation accuracy by ~5% (Ubuntu only).
        2. It is only run if reprojction error is more than a minimum threshold (improve speed with already good quality results) and also less than another outlier threshold.
        3. Outliers are removed from final result if >= 3 camera views.
        4. Applied RANSAC if >=4 camera views.
        5. Latency highly reduced in multi-GPU setting. Each GPU process a different camera view, instead of a different time-instant sequence.
    5. CMake: All libraries as single variable (simpler to add/remove libraries).
    6. Averaged latency reduced to half.
    7. 15% speed up for default CMake version. CMake was not setting `Release` mode by default.
    8. Light speed up, and body approach much more invariant to number of people. Removed `checkEQ` from tight loop in bodyPartConnectorBase, which took a huge time exponential to the number of people.
    9. Datum includes extrinsic and intrinsic camera parameters.
    10. Function `scaleKeypoints(Array<float>& keypoints, const float scale)` also accepts 3D keypoints.
    11. 3D keypoints and camera parameters in meters (instead of millimeters) in order to reduce numerical errors.
    12. New `PoseExtractor` class to contain future ID and tracking algorithms as well as the current OpenPose keypoint detection algorithm.
    13. Added initial alpha versions of the `tracking` and `identification` modules (for now disabled but available in the source code), including `PersonIdExtractor` and `PersonTracker`. `PersonIdExtractor` includes greedy matrix OP-LK matching.
    14. Added catchs to all demos for higher debug information.
    15. GUI includes the capability of dynamically enable/disable the face, hand, and 3-D rendering, as well as more clear visualization for skeleton, background, heatmap addition, and PAF addition channels.
    16. When GUI changes some parameter from PoseExtractorNet, there is a log to notify the user of the change.
    17. Deprecated flag `--write_keypoint_json` removed (`--write_json` is the equivalent since version 1.2.1).
    18. Speed up of cvMatToOpOutput and opOutputToCvMat: op::Datum::outputData is now H x W x C instead of C x H x W, making it much faster to be copied to/from op::Datum::cvOutputData.
    19. Much faster GUI display by adding the `WITH_OPENCV_WITH_OPENGL` flag to tell whether to use OpenGL support for OpenCV.
    20. Turned sanity check error into warning when using dynamic `net_resolution` for `image_dir` in CPU/OpenCL versions.
    21. Minimized CPU usage when queues are empty or full, in order to prevent problems such as general computer slow down, overheating, or excesive power usage.
2. Functions or parameters renamed:
    1. Removed scale parameter from hand and face rectangle extractor (causing wrong results if custom `--output_resolution`).
    2. Functions `scaleKeypoints`, other than `scaleKeypoints(Array<float>& keypoints, const float scale)`, renamed as `scaleKeypoints2d`.
    3. `(W)PoseExtractor` renamed to `(W)PoseExtractorNet` to distinguish from new `PoseExtractor`. Analogously with `(W)FaceExtractorNet` and `(W)HandExtractorNet`.
    4. Experimental module removed and internal `tracking` folder moved to main openpose folder.
    5. Switched GUI shortcuts for the kind of channel to render (skeleton, heatmap, PAF, ...) in order to make it more intuitive: 1 for skeleton, 1 for background heatmap, 2 for adding all heatmaps, 3 for adding all PAFs, and 4 to 0 for the initial heatmaps.
3. Main bugs fixed:
    1. Fixed hand and face extraction and rendering scaling issues when `--output_resolution` is not the default one.
    2. Part candidates (`--part_candidates`) are saved with the same scale than the final keypoints itself.
    3. Fixed bug in keepTopNPeople.hpp (`--number_people_max`) that provoked core dumped if lots of values equal to the threshold.
    4. Flir cameras: Cameras sorted by serial number. Video and images recorded from flir cameras were (and are) assigned the camera parameters based on serial number order, so it would fail if the cameras order was not the same than if sorted by serial number.
    5. CPU version working in non-Nvidia Windows machines.



## Current version - future OpenPose 1.5.0
1. Main improvements:
    1. Added initial single-person tracker for further speed up or visual smoothing (`--tracking` flag).
    2. Greedy body part connector implemented in CUDA: +~30% speed up in Nvidia (CUDA) version with default flags and +~10% in maximum accuracy configuration. In addition, it provides a small 0.5% boost in accuracy (default flags).
    3. OpenPose can be built as Unity plugin: Added flag `BUILD_UNITY_SUPPORT` and special Unity code.
    4. If camera is unplugged, OpenPose GUI and command line will display a warning and try to reconnect it.
    5. Wrapper classes simplified and renamed. Wrapper renamed as WrapperT, and created Wrapper as the non-templated class equivalent.
    6. API and examples improved:
        1. New header file `flags.hpp` that includes all OpenPose flags, removing the need to copy them repeatedly on each OpenPose example file.
        2. `tutorial_wrapper` renamed as `tutorial_api_cpp` as well as new examples were added.
        2. `tutorial_python` renamed as `tutorial_api_python` as well as new examples were added.
        3. `tutorial_pose` and `tutorial_thread` renamed as `tutorial_developer`, not meant to be used by users, but rather for OpenPose developers.
    7. Added a virtual destructor to almost all clases, so they can be inherited. Exceptions (for performance reasons): Array, Point, Rectangle, CvMatToOpOutput, OpOutputToCvMat.
    8. Auxiliary classes in errorAndLog turned into namespaces (Profiler must be kept as class to allow static parameters).
    9. Added flag `--frame_step` to allow the user to select the step or gap between processed frames. E.g., `--frame_step 5` would read and process frames 0, 5, 10, etc.
    10. Added sanity checks to avoid `--frame_last` to be smaller than `--frame_first` or higher than the number of total frames.
    11. Array improvements for Pybind11 compatibility:
        1. Array::getStride() to get step size of each dimension of the array.
        2. Array::getPybindPtr() to get an editable const pointer.
        3. Array::pData as binding of spData.
        4. Array::Array that takes as input a pointer, so it does not re-allocate memory.
    12. Producer defined inside Wrapper rather than being defined on each example.
    13. Reduced many Visual Studio warnings (e.g., uncontrolled conversions between types).
    14. Added new keypoint-related auxiliary functions in `utilities/keypoints.hpp`.
    15. Function `resizeFixedAspectRatio` can take already allocated memory (e.g., faster if target is an Array<T> object, no intermediate cv::Mat required).
    16. Added compatibility for OpenCV 4.0, while preserving 2.4.X and 3.X compatibility.
    17. Improved and added several functions to `utilities/keypoints.hpp` and Array to simplify keypoint post-processing.
    18. Removed warnings from Spinnaker SDK at compiling time.
    19. All bash scripts incorporate `#!/bin/bash` to tell the terminal that they are bash scripts.
    20. Added flag `--verbose` to plot the progress.
    21. Added find_package(Protobuf) to allow specific versions of Protobuf.
    22. Examples do not end in core dumped if an OpenPose exception occurred during initialization, but it is rather closed returning -1. However, it will still results in core dumped if the exception occurs during multi-threading execution.
    23. Video (`--write_video`) can be generated from images (`--image_dir`), as long as they maintain the same resolution.
    24. Added `--fps_max` flag to limit the maximum processing frame rate of OpenPose (useful to display results at a maximum desired speed).
2. Functions or parameters renamed:
    1. By default, python example `tutorial_developer/python_2_pose_from_heatmaps.py` was using 2 scales starting at -1x736, changed to 1 scale at -1x368.
    2. WrapperStructPose default parameters changed to match those of the OpenPose demo binary.
    3. WrapperT.configure() changed from 1 function that requries all arguments to individual functions that take 1 argument each.
    4. Added `Forward` to all net classes that automatically selects between CUDA, OpenCL, or CPU-only version depending on the defines.
    5. Previously hardcoded `COCO_CHALLENGE` variable turned into user configurable flag `--maximize_positives`.
    6. Removed old COCO 2014 validation scripts.
    7. WrapperStructOutput split into WrapperStructOutput and WrapperStructGui.
    8. Replaced `--camera_fps` flag by `--write_video_fps`, given that it was a confusing name: It did not affect the webcam FPS, but only the FPS of the output video. In addition, default value changed from 30 to -1.
3. Main bugs fixed:
    1. CMake-GUI was forcing to Release mode, allowed Debug modes too.
    2. NMS returns in index 0 the number of found peaks. However, while the number of peaks was truncated to a maximum of 127, this index 0 was saving the real number instead of the truncated one.
    3. Template functions could not be imported in Windows for projects using the OpenPose library DLL.
    4. Function `scaleKeypoints2d` was not working if any of the scales was 1 (e.g., fail if scaleX = 1 but scaleY != 1, or if any offset was not 0).
    5. Fixed bug in `KeepTopNPeople` that could provoke segmentation fault for `number_people_max` > 1.



## All OpenPose Versions
Download and/or check any OpenPose version from [https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases).
