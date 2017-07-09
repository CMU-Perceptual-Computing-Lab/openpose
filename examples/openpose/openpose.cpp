// ------------------------- OpenPose Library Tutorial - Real Time Pose Estimation -------------------------
// If the user wants to learn to use the OpenPose library, we highly recommend to start with the `examples/tutorial_*/` folders.
// This example summarizes all the funcitonality of the OpenPose library:
    // 1. Read folder of images / video / webcam  (`producer` module)
    // 2. Extract and render body keypoint / heatmap / PAF of that image (`pose` module)
    // 3. Extract and render face keypoint / heatmap / PAF of that image (`face` module)
    // 4. Save the results on disc (`filestream` module)
    // 5. Display the rendered pose (`gui` module)
    // Everything in a multi-thread scenario (`thread` module)
    // Points 2 to 5 are included in the `wrapper` module
// In addition to the previous OpenPose modules, we also need to use:
    // 1. `core` module:
        // For the Array<float> class that the `pose` module needs
        // For the Datum struct that the `thread` module sends between the queues
    // 2. `utilities` module: for the error & logging functions, i.e. op::error & op::log respectively
// This file should only be used for the user to take specific examples.

// C++ std library dependencies
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <string>
#include <thread> // std::this_thread
#include <vector>
// Other 3rdparty dependencies
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging

// OpenPose dependencies
// Option a) Importing all modules
#include <openpose/headers.hpp>
// Option b) Manually importing the desired modules. Recommended if you only intend to use a few modules.
// #include <openpose/core/headers.hpp>
// #include <openpose/experimental/headers.hpp>
// #include <openpose/face/headers.hpp>
// #include <openpose/filestream/headers.hpp>
// #include <openpose/gui/headers.hpp>
// #include <openpose/pose/headers.hpp>
// #include <openpose/producer/headers.hpp>
// #include <openpose/thread/headers.hpp>
// #include <openpose/utilities/headers.hpp>
// #include <openpose/wrapper/headers.hpp>

// See all the available parameter options withe the `--help` flag. E.g. `./build/examples/openpose/openpose.bin --help`.
// Note: This command will show you flags for other unnecessary 3rdparty files. Check only the flags for the OpenPose
// executable. E.g. for `openpose.bin`, look for `Flags from examples/openpose/openpose.cpp:`.
// Debugging
DEFINE_int32(logging_level,             4,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                                                        " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                                                        " low priority messages and 4 for important ones.");
// Producer
DEFINE_int32(camera,                    0,              "The camera index for cv::VideoCapture. Integer in the range [0, 9].");
DEFINE_string(camera_resolution,        "1280x720",     "Size of the camera frames to ask for.");
DEFINE_double(camera_fps,               30.0,           "Frame rate for the webcam (only used when saving video from webcam). Set this value to the"
                                                        " minimum value between the OpenPose displayed speed and the webcam real frame rate.");
DEFINE_string(video,                    "",             "Use a video file instead of the camera. Use `examples/media/video.avi` for our default"
                                                        " example video.");
DEFINE_string(image_dir,                "",             "Process a directory of images. Use `examples/media/` for our default example folder with 20"
                                                        " images.");
DEFINE_uint64(frame_first,              0,              "Start on desired frame number. Indexes are 0-based, i.e. the first frame has index 0.");
DEFINE_uint64(frame_last,               -1,             "Finish on desired frame number. Select -1 to disable. Indexes are 0-based, e.g. if set to"
                                                        " 10, it will process 11 frames (0-10).");
DEFINE_bool(frame_flip,                 false,          "Flip/mirror each frame (e.g. for real time webcam demonstrations).");
DEFINE_int32(frame_rotate,              0,              "Rotate each frame, 4 possible values: 0, 90, 180, 270.");
DEFINE_bool(frames_repeat,              false,          "Repeat frames when finished.");
// OpenPose
DEFINE_string(model_folder,             "models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(resolution,               "1280x720",     "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                                        " default images resolution.");
DEFINE_int32(num_gpu,                   -1,             "The number of GPU devices to use. If negative, it will use all the available GPUs in your"
                                                        " machine.");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_int32(keypoint_scale,            0,              "Scaling of the (x,y) coordinates of the final pose data array, i.e. the scale of the (x,y)"
                                                        " coordinates that will be saved with the `write_keypoint` & `write_keypoint_json` flags."
                                                        " Select `0` to scale it to the original source resolution, `1`to scale it to the net output"
                                                        " size (set with `net_resolution`), `2` to scale it to the final output size (set with"
                                                        " `resolution`), `3` to scale it in the range [0,1], and 4 for range [-1,1]. Non related"
                                                        " with `scale_number` and `scale_gap`.");
// OpenPose Body Pose
DEFINE_string(model_pose,               "COCO",         "Model to be used (e.g. COCO, MPI, MPI_4_layers).");
DEFINE_string(net_resolution,           "656x368",      "Multiples of 16. If it is increased, the accuracy usually increases. If it is decreased,"
                                                        " the speed increases.");
DEFINE_int32(scale_number,              1,              "Number of scales to average.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
                                                        " If you want to change the initial scale, you actually want to multiply the"
                                                        " `net_resolution` by your desired initial scale.");
DEFINE_bool(heatmaps_add_parts,         false,          "If true, it will add the body part heatmaps to the final op::Datum::poseHeatMaps array"
                                                        " (program speed will decrease). Not required for our library, enable it only if you intend"
                                                        " to process this information later. If more than one `add_heatmaps_X` flag is enabled, it"
                                                        " will place then in sequential memory order: body parts + bkg + PAFs. It will follow the"
                                                        " order on POSE_BODY_PART_MAPPING in `include/openpose/pose/poseParameters.hpp`.");
DEFINE_bool(heatmaps_add_bkg,           false,          "Same functionality as `add_heatmaps_parts`, but adding the heatmap corresponding to"
                                                        " background.");
DEFINE_bool(heatmaps_add_PAFs,          false,          "Same functionality as `add_heatmaps_parts`, but adding the PAFs.");
// OpenPose Face
DEFINE_bool(face,                       false,          "Enables face keypoint detection. It will share some parameters from the body pose, e.g."
                                                        " `model_folder`.");
DEFINE_string(face_net_resolution,      "368x368",      "Multiples of 16. Analogous to `net_resolution` but applied to the face keypoint detector."
                                                        " 320x320 usually works fine while giving a substantial speed up when multiple faces on the"
                                                        " image.");
// OpenPose Hand
DEFINE_bool(hand,                       false,          "Enables hand keypoint detection. It will share some parameters from the body pose, e.g."
                                                        " `model_folder`.");
DEFINE_string(hand_net_resolution,      "368x368",      "Multiples of 16. Analogous to `net_resolution` but applied to the hand keypoint detector.");
DEFINE_int32(hand_scale_number,         1,              "Analogous to `scale_number` but applied to the hand keypoint detector. Our best results"
                                                        " were found with `hand_scale_number` = 6 and `hand_scale_range` = 0.4");
DEFINE_double(hand_scale_range,         0.4,            "Analogous purpose than `scale_gap` but applied to the hand keypoint detector. Total range"
                                                        " between smallest and biggest scale. The scales will be centered in ratio 1. E.g. if"
                                                        " scaleRange = 0.4 and scalesNumber = 2, then there will be 2 scales, 0.8 and 1.2.");
DEFINE_bool(hand_tracking,              false,          "Adding hand tracking might improve hand keypoints detection for webcam (if the frame rate"
                                                        " is high enough, i.e. >7 FPS per GPU) and video. This is not person ID tracking, it"
                                                        " simply looks for hands in positions at which hands were located in previous frames, but"
                                                        " it does not guarantee the same person ID among frames");
// OpenPose Rendering
DEFINE_int32(part_to_show,              0,              "Part to show from the start.");
DEFINE_bool(disable_blending,           false,          "If blending is enabled, it will merge the results with the original frame. If disabled, it"
                                                        " will only display the results.");
// OpenPose Rendering Pose
DEFINE_double(render_threshold,         0.05,           "Only estimated keypoints whose score confidences are higher than this threshold will be"
                                                        " rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
                                                        " while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
                                                        " more false positives (i.e. wrong detections).");
DEFINE_int32(render_pose,               2,              "Set to 0 for no rendering, 1 for CPU rendering (slightly faster), and 2 for GPU rendering"
                                                        " (slower but greater functionality, e.g. `alpha_X` flags). If rendering is enabled, it will"
                                                        " render both `outputData` and `cvOutputData` with the original image and desired body part"
                                                        " to be shown (i.e. keypoints, heat maps or PAFs).");
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
// Display
DEFINE_bool(fullscreen,                 false,          "Run in full-screen mode (press f during runtime to toggle).");
DEFINE_bool(process_real_time,          false,          "Enable to keep the original source frame rate (e.g. for video). If the processing time is"
                                                        " too long, it will skip frames. If it is too fast, it will slow it down.");
DEFINE_bool(no_gui_verbose,             false,          "Do not write text on output images on GUI (e.g. number of current frame and people). It"
                                                        " does not affect the pose rendering.");
DEFINE_bool(no_display,                 false,          "Do not open a display window.");
// Result Saving
DEFINE_string(write_images,             "",             "Directory to write rendered frames in `write_images_format` image format.");
DEFINE_string(write_images_format,      "png",          "File extension and format for `write_images`, e.g. png, jpg or bmp. Check the OpenCV"
                                                        " function cv::imwrite for all compatible extensions.");
DEFINE_string(write_video,              "",             "Full file path to write rendered frames in motion JPEG video format. It might fail if the"
                                                        " final path does not finish in `.avi`. It internally uses cv::VideoWriter.");
DEFINE_string(write_keypoint,           "",             "Directory to write the people body pose keypoint data. Set format with `write_keypoint_format`.");
DEFINE_string(write_keypoint_format,    "yml",          "File extension and format for `write_keypoint`: json, xml, yaml & yml. Json not available"
                                                        " for OpenCV < 3.0, use `write_keypoint_json` instead.");
DEFINE_string(write_keypoint_json,      "",             "Directory to write people pose data in *.json format, compatible with any OpenCV version.");
DEFINE_string(write_coco_json,          "",             "Full file path to write people pose data with *.json COCO validation format.");
DEFINE_string(write_heatmaps,           "",             "Directory to write heatmaps in *.png format. At least 1 `add_heatmaps_X` flag must be"
                                                        " enabled.");
DEFINE_string(write_heatmaps_format,    "png",          "File extension and format for `write_heatmaps`, analogous to `write_images_format`."
                                                        " Recommended `png` or any compressed and lossless format.");

int openPoseDemo()
{
    // logging_level
    op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
    // op::ConfigureLog::setPriorityThreshold(op::Priority::None); // To print all logging messages

    op::log("Starting pose estimation demo.", op::Priority::High);
    const auto timerBegin = std::chrono::high_resolution_clock::now();

    // Applying user defined configuration - Google flags to program variables
    // outputSize
    const auto outputSize = op::flagsToPoint(FLAGS_resolution, "1280x720");
    // netInputSize
    const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "656x368");
    // faceNetInputSize
    const auto faceNetInputSize = op::flagsToPoint(FLAGS_face_net_resolution, "368x368 (multiples of 16)");
    // handNetInputSize
    const auto handNetInputSize = op::flagsToPoint(FLAGS_hand_net_resolution, "368x368 (multiples of 16)");
    // producerType
    const auto producerSharedPtr = op::flagsToProducer(FLAGS_image_dir, FLAGS_video, FLAGS_camera, FLAGS_camera_resolution, FLAGS_camera_fps);
    // poseModel
    const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
    // keypointScale
    const auto keypointScale = op::flagsToScaleMode(FLAGS_keypoint_scale);
    // heatmaps to add
    const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg, FLAGS_heatmaps_add_PAFs);
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);

    // OpenPose wrapper
    op::log("Configuring OpenPose wrapper.", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    op::Wrapper<std::vector<op::Datum>> opWrapper;
    // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
    const op::WrapperStructPose wrapperStructPose{netInputSize, outputSize, keypointScale, FLAGS_num_gpu,
                                                  FLAGS_num_gpu_start, FLAGS_scale_number, (float)FLAGS_scale_gap,
                                                  op::flagsToRenderMode(FLAGS_render_pose), poseModel,
                                                  !FLAGS_disable_blending, (float)FLAGS_alpha_pose,
                                                  (float)FLAGS_alpha_heatmap, FLAGS_part_to_show, FLAGS_model_folder,
                                                  heatMapTypes, op::ScaleMode::UnsignedChar, (float)FLAGS_render_threshold};
    // Face configuration (use op::WrapperStructFace{} to disable it)
    const op::WrapperStructFace wrapperStructFace{FLAGS_face, faceNetInputSize, op::flagsToRenderMode(FLAGS_face_render, FLAGS_render_pose),
                                                  (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
    // Hand configuration (use op::WrapperStructHand{} to disable it)
    const op::WrapperStructHand wrapperStructHand{FLAGS_hand, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
                                                  FLAGS_hand_tracking, op::flagsToRenderMode(FLAGS_hand_render, FLAGS_render_pose),
                                                  (float)FLAGS_hand_alpha_pose, (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
    // Producer (use default to disable any input)
    const op::WrapperStructInput wrapperStructInput{producerSharedPtr, FLAGS_frame_first, FLAGS_frame_last, FLAGS_process_real_time,
                                                    FLAGS_frame_flip, FLAGS_frame_rotate, FLAGS_frames_repeat};
    // Consumer (comment or use default argument to disable any output)
    const op::WrapperStructOutput wrapperStructOutput{!FLAGS_no_display, !FLAGS_no_gui_verbose, FLAGS_fullscreen, FLAGS_write_keypoint,
                                                      op::stringToDataFormat(FLAGS_write_keypoint_format), FLAGS_write_keypoint_json,
                                                      FLAGS_write_coco_json, FLAGS_write_images, FLAGS_write_images_format, FLAGS_write_video,
                                                      FLAGS_write_heatmaps, FLAGS_write_heatmaps_format};
    // Configure wrapper
    opWrapper.configure(wrapperStructPose, wrapperStructFace, wrapperStructHand, wrapperStructInput, wrapperStructOutput);
    // Set to single-thread running (e.g. for debugging purposes)
    // opWrapper.disableMultiThreading();

    // Start processing
    // Two different ways of running the program on multithread environment
    op::log("Starting thread(s)", op::Priority::High);
    // Option a) Recommended - Also using the main thread (this thread) for processing (it saves 1 thread)
    // Start, run & stop threads
    opWrapper.exec();  // It blocks this thread until all threads have finished

    // // Option b) Keeping this thread free in case you want to do something else meanwhile, e.g. profiling the GPU memory
    // // VERY IMPORTANT NOTE: if OpenCV is compiled with Qt support, this option will not work. Qt needs the main thread to
    // // plot visual results, so the final GUI (which uses OpenCV) would return an exception similar to:
    // // `QMetaMethod::invoke: Unable to invoke methods with return values in queued connections`
    // // Start threads
    // opWrapper.start();
    // // Profile used GPU memory
    //     // 1: wait ~10sec so the memory has been totally loaded on GPU
    //     // 2: profile the GPU memory
    // const auto sleepTimeMs = 10;
    // for (auto i = 0 ; i < 10000/sleepTimeMs && opWrapper.isRunning() ; i++)
    //     std::this_thread::sleep_for(std::chrono::milliseconds{sleepTimeMs});
    // op::Profiler::profileGpuMemory(__LINE__, __FUNCTION__, __FILE__);
    // // Keep program alive while running threads
    // while (opWrapper.isRunning())
    //     std::this_thread::sleep_for(std::chrono::milliseconds{sleepTimeMs});
    // // Stop and join threads
    // op::log("Stopping thread(s)", op::Priority::High);
    // opWrapper.stop();

    // Measuring total time
    const auto now = std::chrono::high_resolution_clock::now();
    const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now-timerBegin).count() * 1e-9;
    const auto message = "Real-time pose estimation demo successfully finished. Total time: " + std::to_string(totalTimeSec) + " seconds.";
    op::log(message, op::Priority::High);

    return 0;
}

int main(int argc, char *argv[])
{
    // Initializing google logging (Caffe uses it for logging)
    google::InitGoogleLogging("openPoseDemo");

    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running openPoseDemo
    return openPoseDemo();
}
