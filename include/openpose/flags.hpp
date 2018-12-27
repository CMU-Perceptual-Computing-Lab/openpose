#ifndef OPENPOSE_FLAGS_HPP
#define OPENPOSE_FLAGS_HPP

// Note: This class is not included within the basic OpenPose `headers.hpp` and must be explicitly included. In
// addition, Google Flags library must also be linked to the resulting binary or library. OpenPose library does
// not force to use Google Flags, but the OpenPose examples do so.

// GFlags: DEFINE_bool, _int32, _int64, _uint64, _double, _string
#include <gflags/gflags.h>
// Allow Google Flags in Ubuntu 14
#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

// See all the available parameter options withe the `--help` flag. E.g., `build/examples/openpose/openpose.bin --help`
// Note: This command will show you flags for other unnecessary 3rdparty files. Check only the flags for the OpenPose
// executable. E.g., for `openpose.bin`, look for `Flags from examples/openpose/openpose.cpp:`.
// Debugging/Other
DEFINE_int32(logging_level,             3,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                                                        " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                                                        " low priority messages and 4 for important ones.");
DEFINE_bool(disable_multi_thread,       false,          "It would slightly reduce the frame rate in order to highly reduce the lag. Mainly useful"
                                                        " for 1) Cases where it is needed a low latency (e.g., webcam in real-time scenarios with"
                                                        " low-range GPU devices); and 2) Debugging OpenPose when it is crashing to locate the"
                                                        " error.");
DEFINE_int32(profile_speed,             1000,           "If PROFILER_ENABLED was set in CMake or Makefile.config files, OpenPose will show some"
                                                        " runtime statistics at this frame number.");
#ifndef OPENPOSE_FLAGS_DISABLE_POSE
#ifndef OPENPOSE_FLAGS_DISABLE_PRODUCER
// Producer
DEFINE_int32(camera,                    -1,             "The camera index for cv::VideoCapture. Integer in the range [0, 9]. Select a negative"
                                                        " number (by default), to auto-detect and open the first available camera.");
DEFINE_string(camera_resolution,        "-1x-1",        "Set the camera resolution (either `--camera` or `--flir_camera`). `-1x-1` will use the"
                                                        " default 1280x720 for `--camera`, or the maximum flir camera resolution available for"
                                                        " `--flir_camera`");
DEFINE_string(video,                    "",             "Use a video file instead of the camera. Use `examples/media/video.avi` for our default"
                                                        " example video.");
DEFINE_string(image_dir,                "",             "Process a directory of images. Use `examples/media/` for our default example folder with 20"
                                                        " images. Read all standard formats (jpg, png, bmp, etc.).");
DEFINE_bool(flir_camera,                false,          "Whether to use FLIR (Point-Grey) stereo camera.");
DEFINE_int32(flir_camera_index,         -1,             "Select -1 (default) to run on all detected flir cameras at once. Otherwise, select the flir"
                                                        " camera index to run, where 0 corresponds to the detected flir camera with the lowest"
                                                        " serial number, and `n` to the `n`-th lowest serial number camera.");
DEFINE_string(ip_camera,                "",             "String with the IP camera URL. It supports protocols like RTSP and HTTP.");
DEFINE_uint64(frame_first,              0,              "Start on desired frame number. Indexes are 0-based, i.e., the first frame has index 0.");
DEFINE_uint64(frame_step,               1,              "Step or gap between processed frames. E.g., `--frame_step 5` would read and process frames"
                                                        " 0, 5, 10, etc..");
DEFINE_uint64(frame_last,               -1,             "Finish on desired frame number. Select -1 to disable. Indexes are 0-based, e.g., if set to"
                                                        " 10, it will process 11 frames (0-10).");
DEFINE_bool(frame_flip,                 false,          "Flip/mirror each frame (e.g., for real time webcam demonstrations).");
DEFINE_int32(frame_rotate,              0,              "Rotate each frame, 4 possible values: 0, 90, 180, 270.");
DEFINE_bool(frames_repeat,              false,          "Repeat frames when finished.");
DEFINE_bool(process_real_time,          false,          "Enable to keep the original source frame rate (e.g., for video). If the processing time is"
                                                        " too long, it will skip frames. If it is too fast, it will slow it down.");
DEFINE_string(camera_parameter_path,    "models/cameraParameters/flir/", "String with the folder where the camera parameters are located. If there"
                                                        " is only 1 XML file (for single video, webcam, or images from the same camera), you must"
                                                        " specify the whole XML file path (ending in .xml).");
DEFINE_bool(frame_undistort,            false,          "If false (default), it will not undistort the image, if true, it will undistortionate them"
                                                        " based on the camera parameters found in `camera_parameter_path`");
#endif // OPENPOSE_FLAGS_DISABLE_PRODUCER
// OpenPose
DEFINE_string(model_folder,             "models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(output_resolution,        "-1x-1",        "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                                        " input image resolution.");
DEFINE_int32(num_gpu,                   -1,             "The number of GPU devices to use. If negative, it will use all the available GPUs in your"
                                                        " machine.");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_int32(keypoint_scale,            0,              "Scaling of the (x,y) coordinates of the final pose data array, i.e., the scale of the (x,y)"
                                                        " coordinates that will be saved with the `write_json` & `write_keypoint` flags."
                                                        " Select `0` to scale it to the original source resolution; `1`to scale it to the net output"
                                                        " size (set with `net_resolution`); `2` to scale it to the final output size (set with"
                                                        " `resolution`); `3` to scale it in the range [0,1], where (0,0) would be the top-left"
                                                        " corner of the image, and (1,1) the bottom-right one; and 4 for range [-1,1], where"
                                                        " (-1,-1) would be the top-left corner of the image, and (1,1) the bottom-right one. Non"
                                                        " related with `scale_number` and `scale_gap`.");
DEFINE_int32(number_people_max,         -1,             "This parameter will limit the maximum number of people detected, by keeping the people with"
                                                        " top scores. The score is based in person area over the image, body part score, as well as"
                                                        " joint score (between each pair of connected body parts). Useful if you know the exact"
                                                        " number of people in the scene, so it can remove false positives (if all the people have"
                                                        " been detected. However, it might also include false negatives by removing very small or"
                                                        " highly occluded people. -1 will keep them all.");
DEFINE_bool(maximize_positives,         false,          "It reduces the thresholds to accept a person candidate. It highly increases both false and"
                                                        " true positives. I.e., it maximizes average recall but could harm average precision.");
DEFINE_double(fps_max,                  -1.,            "Maximum processing frame rate. By default (-1), OpenPose will process frames as fast as"
                                                        " possible. Example usage: If OpenPose is displaying images too quickly, this can reduce"
                                                        " the speed so the user can analyze better each frame from the GUI.");
// OpenPose Body Pose
DEFINE_bool(body_disable,               false,          "Disable body keypoint detection. Option only possible for faster (but less accurate) face"
                                                        " keypoint detection.");
DEFINE_string(model_pose,               "BODY_25",      "Model to be used. E.g., `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
                                                        "`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(net_resolution,           "-1x368",       "Multiples of 16. If it is increased, the accuracy potentially increases. If it is"
                                                        " decreased, the speed increases. For maximum speed-accuracy balance, it should keep the"
                                                        " closest aspect ratio possible to the images or videos to be processed. Using `-1` in"
                                                        " any of the dimensions, OP will choose the optimal aspect ratio depending on the user's"
                                                        " input value. E.g., the default `-1x368` is equivalent to `656x368` in 16:9 resolutions,"
                                                        " e.g., full HD (1980x1080) and HD (1280x720) resolutions.");
DEFINE_int32(scale_number,              1,              "Number of scales to average.");
DEFINE_double(scale_gap,                0.25,           "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
                                                        " If you want to change the initial scale, you actually want to multiply the"
                                                        " `net_resolution` by your desired initial scale.");
// OpenPose Body Pose Heatmaps and Part Candidates
DEFINE_bool(heatmaps_add_parts,         false,          "If true, it will fill op::Datum::poseHeatMaps array with the body part heatmaps, and"
                                                        " analogously face & hand heatmaps to op::Datum::faceHeatMaps & op::Datum::handHeatMaps."
                                                        " If more than one `add_heatmaps_X` flag is enabled, it will place then in sequential"
                                                        " memory order: body parts + bkg + PAFs. It will follow the order on"
                                                        " POSE_BODY_PART_MAPPING in `src/openpose/pose/poseParameters.cpp`. Program speed will"
                                                        " considerably decrease. Not required for OpenPose, enable it only if you intend to"
                                                        " explicitly use this information later.");
DEFINE_bool(heatmaps_add_bkg,           false,          "Same functionality as `add_heatmaps_parts`, but adding the heatmap corresponding to"
                                                        " background.");
DEFINE_bool(heatmaps_add_PAFs,          false,          "Same functionality as `add_heatmaps_parts`, but adding the PAFs.");
DEFINE_int32(heatmaps_scale,            2,              "Set 0 to scale op::Datum::poseHeatMaps in the range [-1,1], 1 for [0,1]; 2 for integer"
                                                        " rounded [0,255]; and 3 for no scaling.");
DEFINE_bool(part_candidates,            false,          "Also enable `write_json` in order to save this information. If true, it will fill the"
                                                        " op::Datum::poseCandidates array with the body part candidates. Candidates refer to all"
                                                        " the detected body parts, before being assembled into people. Note that the number of"
                                                        " candidates is equal or higher than the number of final body parts (i.e., after being"
                                                        " assembled into people). The empty body parts are filled with 0s. Program speed will"
                                                        " slightly decrease. Not required for OpenPose, enable it only if you intend to explicitly"
                                                        " use this information.");
// OpenPose Face
DEFINE_bool(face,                       false,          "Enables face keypoint detection. It will share some parameters from the body pose, e.g."
                                                        " `model_folder`. Note that this will considerable slow down the performance and increse"
                                                        " the required GPU memory. In addition, the greater number of people on the image, the"
                                                        " slower OpenPose will be.");
DEFINE_string(face_net_resolution,      "368x368",      "Multiples of 16 and squared. Analogous to `net_resolution` but applied to the face keypoint"
                                                        " detector. 320x320 usually works fine while giving a substantial speed up when multiple"
                                                        " faces on the image.");
// OpenPose Hand
DEFINE_bool(hand,                       false,          "Enables hand keypoint detection. It will share some parameters from the body pose, e.g."
                                                        " `model_folder`. Analogously to `--face`, it will also slow down the performance, increase"
                                                        " the required GPU memory and its speed depends on the number of people.");
DEFINE_string(hand_net_resolution,      "368x368",      "Multiples of 16 and squared. Analogous to `net_resolution` but applied to the hand keypoint"
                                                        " detector.");
DEFINE_int32(hand_scale_number,         1,              "Analogous to `scale_number` but applied to the hand keypoint detector. Our best results"
                                                        " were found with `hand_scale_number` = 6 and `hand_scale_range` = 0.4.");
DEFINE_double(hand_scale_range,         0.4,            "Analogous purpose than `scale_gap` but applied to the hand keypoint detector. Total range"
                                                        " between smallest and biggest scale. The scales will be centered in ratio 1. E.g., if"
                                                        " scaleRange = 0.4 and scalesNumber = 2, then there will be 2 scales, 0.8 and 1.2.");
DEFINE_bool(hand_tracking,              false,          "Adding hand tracking might improve hand keypoints detection for webcam (if the frame rate"
                                                        " is high enough, i.e., >7 FPS per GPU) and video. This is not person ID tracking, it"
                                                        " simply looks for hands in positions at which hands were located in previous frames, but"
                                                        " it does not guarantee the same person ID among frames.");
// OpenPose 3-D Reconstruction
DEFINE_bool(3d,                         false,          "Running OpenPose 3-D reconstruction demo: 1) Reading from a stereo camera system."
                                                        " 2) Performing 3-D reconstruction from the multiple views. 3) Displaying 3-D reconstruction"
                                                        " results. Note that it will only display 1 person. If multiple people is present, it will"
                                                        " fail.");
DEFINE_int32(3d_min_views,              -1,             "Minimum number of views required to reconstruct each keypoint. By default (-1), it will"
                                                        " require all the cameras to see the keypoint in order to reconstruct it.");
DEFINE_int32(3d_views,                  -1,             "Complementary option for `--image_dir` or `--video`. OpenPose will read as many images per"
                                                        " iteration, allowing tasks such as stereo camera processing (`--3d`). Note that"
                                                        " `--camera_parameter_path` must be set. OpenPose must find as many `xml` files in the"
                                                        " parameter folder as this number indicates.");
// Extra algorithms
DEFINE_bool(identification,             false,          "Experimental, not available yet. Whether to enable people identification across frames.");
DEFINE_int32(tracking,                  -1,             "Experimental, not available yet. Whether to enable people tracking across frames. The"
                                                        " value indicates the number of frames where tracking is run between each OpenPose keypoint"
                                                        " detection. Select -1 (default) to disable it or 0 to run simultaneously OpenPose keypoint"
                                                        " detector and tracking for potentially higher accurary than only OpenPose.");
DEFINE_int32(ik_threads,                0,              "Experimental, not available yet. Whether to enable inverse kinematics (IK) from 3-D"
                                                        " keypoints to obtain 3-D joint angles. By default (0 threads), it is disabled. Increasing"
                                                        " the number of threads will increase the speed but also the global system latency.");
// OpenPose Rendering
DEFINE_int32(part_to_show,              0,              "Prediction channel to visualize (default: 0). 0 for all the body parts, 1-18 for each body"
                                                        " part heat map, 19 for the background heat map, 20 for all the body part heat maps"
                                                        " together, 21 for all the PAFs, 22-40 for each body part pair PAF.");
DEFINE_bool(disable_blending,           false,          "If enabled, it will render the results (keypoint skeletons or heatmaps) on a black"
                                                        " background, instead of being rendered into the original image. Related: `part_to_show`,"
                                                        " `alpha_pose`, and `alpha_pose`.");
// OpenPose Rendering Pose
DEFINE_double(render_threshold,         0.05,           "Only estimated keypoints whose score confidences are higher than this threshold will be"
                                                        " rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
                                                        " while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
                                                        " more false positives (i.e., wrong detections).");
DEFINE_int32(render_pose,               -1,             "Set to 0 for no rendering, 1 for CPU rendering (slightly faster), and 2 for GPU rendering"
                                                        " (slower but greater functionality, e.g., `alpha_X` flags). If -1, it will pick CPU if"
                                                        " CPU_ONLY is enabled, or GPU if CUDA is enabled. If rendering is enabled, it will render"
                                                        " both `outputData` and `cvOutputData` with the original image and desired body part to be"
                                                        " shown (i.e., keypoints, heat maps or PAFs).");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
                                                        " hide it. Only valid for GPU rendering.");
DEFINE_double(alpha_heatmap,            0.7,            "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the"
                                                        " heatmap, 0 will only show the frame. Only valid for GPU rendering.");
// OpenPose Rendering Face
DEFINE_double(face_render_threshold,    0.4,            "Analogous to `render_threshold`, but applied to the face keypoints.");
DEFINE_int32(face_render,               -1,             "Analogous to `render_pose` but applied to the face. Extra option: -1 to use the same"
                                                        " configuration that `render_pose` is using.");
DEFINE_double(face_alpha_pose,          0.6,            "Analogous to `alpha_pose` but applied to face.");
DEFINE_double(face_alpha_heatmap,       0.7,            "Analogous to `alpha_heatmap` but applied to face.");
// OpenPose Rendering Hand
DEFINE_double(hand_render_threshold,    0.2,            "Analogous to `render_threshold`, but applied to the hand keypoints.");
DEFINE_int32(hand_render,               -1,             "Analogous to `render_pose` but applied to the hand. Extra option: -1 to use the same"
                                                        " configuration that `render_pose` is using.");
DEFINE_double(hand_alpha_pose,          0.6,            "Analogous to `alpha_pose` but applied to hand.");
DEFINE_double(hand_alpha_heatmap,       0.7,            "Analogous to `alpha_heatmap` but applied to hand.");
#ifndef OPENPOSE_FLAGS_DISABLE_DISPLAY
// Display
DEFINE_bool(fullscreen,                 false,          "Run in full-screen mode (press f during runtime to toggle).");
DEFINE_bool(no_gui_verbose,             false,          "Do not write text on output images on GUI (e.g., number of current frame and people). It"
                                                        " does not affect the pose rendering.");
DEFINE_int32(display,                   -1,             "Display mode: -1 for automatic selection; 0 for no display (useful if there is no X server"
                                                        " and/or to slightly speed up the processing if visual output is not required); 2 for 2-D"
                                                        " display; 3 for 3-D display (if `--3d` enabled); and 1 for both 2-D and 3-D display.");
#endif // OPENPOSE_FLAGS_DISABLE_DISPLAY
// Command Line Interface Verbose
DEFINE_double(cli_verbose,              -1.f,           "If -1, it will be disabled (default). If it is a positive integer number, it will print on"
                                                        " the command line every `verbose` frames. If number in the range (0,1), it will print the"
                                                        " progress every `verbose` times the total of frames.");
// Result Saving
DEFINE_string(write_images,             "",             "Directory to write rendered frames in `write_images_format` image format.");
DEFINE_string(write_images_format,      "png",          "File extension and format for `write_images`, e.g., png, jpg or bmp. Check the OpenCV"
                                                        " function cv::imwrite for all compatible extensions.");
DEFINE_string(write_video,              "",             "Full file path to write rendered frames in motion JPEG video format. It might fail if the"
                                                        " final path does not finish in `.avi`. It internally uses cv::VideoWriter. Flag"
                                                        " `write_video_fps` controls FPS.");
DEFINE_double(write_video_fps,          -1.,            "Frame rate for the recorded video. By default, it will try to get the input frames producer"
                                                        " frame rate (e.g., input video or webcam frame rate). If the input frames producer does not"
                                                        " have a set FPS (e.g., image_dir or webcam if OpenCV not compiled with its support), set"
                                                        " this value accordingly (e.g., to the frame rate displayed by the OpenPose GUI).");
DEFINE_string(write_video_3d,           "",             "Analogous to `--write_video`, but applied to the 3D output.");
DEFINE_string(write_video_adam,         "",             "Experimental, not available yet. Analogous to `--write_video`, but applied to Adam model.");
DEFINE_string(write_json,               "",             "Directory to write OpenPose output in JSON format. It includes body, hand, and face pose"
                                                        " keypoints (2-D and 3-D), as well as pose candidates (if `--part_candidates` enabled).");
DEFINE_string(write_coco_json,          "",             "Full file path to write people pose data with JSON COCO validation format.");
DEFINE_string(write_coco_foot_json,     "",             "Full file path to write people foot pose data with JSON COCO validation format.");
DEFINE_int32(write_coco_json_variant,   0,             "Currently, this option is experimental and only makes effect on car JSON generation. It"
                                                        " selects the COCO variant for cocoJsonSaver.");
DEFINE_string(write_heatmaps,           "",             "Directory to write body pose heatmaps in PNG format. At least 1 `add_heatmaps_X` flag"
                                                        " must be enabled.");
DEFINE_string(write_heatmaps_format,    "png",          "File extension and format for `write_heatmaps`, analogous to `write_images_format`."
                                                        " For lossless compression, recommended `png` for integer `heatmaps_scale` and `float` for"
                                                        " floating values.");
DEFINE_string(write_keypoint,           "",             "(Deprecated, use `write_json`) Directory to write the people pose keypoint data. Set format"
                                                        " with `write_keypoint_format`.");
DEFINE_string(write_keypoint_format,    "yml",          "(Deprecated, use `write_json`) File extension and format for `write_keypoint`: json, xml,"
                                                        " yaml & yml. Json not available for OpenCV < 3.0, use `write_json` instead.");
// Result Saving - Extra Algorithms
DEFINE_string(write_bvh,                "",             "Experimental, not available yet. E.g., `~/Desktop/mocapResult.bvh`.");
// UDP Communication
DEFINE_string(udp_host,                 "",             "Experimental, not available yet. IP for UDP communication. E.g., `192.168.0.1`.");
DEFINE_string(udp_port,                 "8051",         "Experimental, not available yet. Port number for UDP communication.");
#endif // OPENPOSE_FLAGS_DISABLE_POSE

#endif // OPENPOSE_FLAGS_HPP
