// ------------------------- OpenPose Library Tutorial - Real Time Pose Estimation -------------------------
// If the user wants to learn to use the OpenPose library, we highly recommend to start with the `examples/tutorial_*/` folders.
// This example summarizes all the funcitonality of the OpenPose library:
    // 1. Read folder of images / video / webcam  (`producer` module)
    // 2. Extract and render pose / heatmap / PAF of that image (`pose` module)
    // 3. Save the results on disc (`filestream` module)
    // 4. Display the rendered pose (`gui` module)
    // Everything in a multi-thread scenario (`thread` module)
    // Points 2 to 4 are included in the `wrapper` module
// In addition to the previous OpenPose modules, we also need to use:
    // 1. `core` module:
        // For the Array<float> class that the `pose` module needs
        // For the Datum struct that the `thread` module sends between the queues
    // 2. `utilities` module: for the error & logging functions, i.e. op::error & op::log respectively
// This file should only be used for the user to take specific examples.

// C++ std library dependencies
#include <atomic> // std::atomic
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <cstdio> // sscanf
#include <string> // std::string
#include <thread> // std::this_thread
#include <vector> // std::vector
// OpenCV dependencies
#include <opencv2/core/core.hpp> // cv::Mat & cv::Size
// Other 3rdpary depencencies
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging, CHECK, CHECK_EQ, LOG, VLOG, ...

// OpenPose dependencies
// Option a) Importing all modules
#include <openpose/headers.hpp>
// Option b) Manually importing the desired modules. Recommended if you only intend to use a few modules.
// #include <openpose/core/headers.hpp>
// #include <openpose/experimental/headers.hpp>
// #include <openpose/filestream/headers.hpp>
// #include <openpose/gui/headers.hpp>
// #include <openpose/pose/headers.hpp>
// #include <openpose/producer/headers.hpp>
// #include <openpose/thread/headers.hpp>
// #include <openpose/utilities/headers.hpp>
// #include <openpose/wrapper/headers.hpp>

// Uncomment to avoid needing `op::` before each class and function of the OpenPose library. Analogously for OpenCV and the standard C++ library
// using namespace op;
// using namespace cv;
// using namespace std;

// Gflags in the command line terminal. Check all the options by adding the flag `--help`, e.g. `openpose.bin --help`.
// Note: This command will show you flags for several files. Check only the flags for the file you are checking. E.g. for `openpose.bin`, look for `Flags from examples/openpose/openpose.cpp:`.
// Debugging
DEFINE_int32(logging_level,             3,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while 255 will not output any."
                                                        " Current OpenPose library messages are in the range 0-4: 1 for low priority messages and 4 for important ones.");
// Producer
DEFINE_int32(camera,                    0,              "The camera index for cv::VideoCapture. Integer in the range [0, 9].");
DEFINE_string(camera_resolution,        "1280x720",     "Size of the camera frames to ask for.");
DEFINE_string(video,                    "",             "Use a video file instead of the camera. Use `examples/media/video.avi` for our default example video.");
DEFINE_string(image_dir,                "",             "Process a directory of images. Use `examples/media/` for our default example folder with 20 images.");
DEFINE_uint64(frame_first,              0,              "Start on desired frame number.");
DEFINE_uint64(frame_last,               -1,             "Finish on desired frame number. Select -1 to disable this option.");
DEFINE_bool(frame_flip,                 false,          "Flip/mirror each frame (e.g. for real time webcam demonstrations).");
DEFINE_int32(frame_rotate,              0,              "Rotate each frame, 4 possible values: 0, 90, 180, 270.");
DEFINE_bool(frames_repeat,              false,          "Repeat frames when finished.");
// OpenPose
DEFINE_string(model_pose,               "COCO",         "Model to be used (e.g. COCO, MPI, MPI_4_layers).");
DEFINE_string(model_folder,             "models/",      "Folder where the pose models (COCO and MPI) are located.");
DEFINE_string(net_resolution,           "656x368",      "Multiples of 16.");
DEFINE_string(resolution,               "1280x720",     "The image resolution (display and output). Use \"-1x-1\" to force the program to use the default images resolution.");
DEFINE_int32(num_gpu,                   1,              "The number of GPU devices to use.");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_int32(num_scales,                1,              "Number of scales to average.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless num_scales>1. Initial scale is always 1. If you want to change the initial scale,"
                                                        " you actually want to multiply the `net_resolution` by your desired initial scale.");
DEFINE_int32(scale_mode,                0,              "Scaling of the (x,y) coordinates of the final pose data array (op::Datum::pose), i.e. the scale of the (x,y) coordinates that"
                                                        " will be saved with the `write_pose` & `write_pose_json` flags. Select `0` to scale it to the original source resolution, `1`"
                                                        " to scale it to the net output size (set with `net_resolution`), `2` to scale it to the final output size (set with "
                                                        " `resolution`), `3` to scale it in the range [0,1], and 4 for range [-1,1]. Non related with `num_scales` and `scale_gap`.");
DEFINE_bool(heatmaps_add_parts,         false,          "If true, it will add the body part heatmaps to the final op::Datum::poseHeatMaps array (program speed will decrease). Not"
                                                        " required for our library, enable it only if you intend to process this information later. If more than one `add_heatmaps_X`"
                                                        " flag is enabled, it will place then in sequential memory order: body parts + bkg + PAFs. It will follow the order on"
                                                        " POSE_BODY_PART_MAPPING in `include/openpose/pose/poseParameters.hpp`.");
DEFINE_bool(heatmaps_add_bkg,           false,          "Same functionality as `add_heatmaps_parts`, but adding the heatmap corresponding to background.");
DEFINE_bool(heatmaps_add_PAFs,          false,          "Same functionality as `add_heatmaps_parts`, but adding the PAFs.");
// OpenPose Rendering
DEFINE_bool(no_render_output,           false,          "If false, it will fill both `outputData` and `cvOutputData` with the original image + desired part to be shown."
                                                        " If true, it will leave them empty.");
DEFINE_int32(part_to_show,              0,              "Part to show from the start.");
DEFINE_bool(disable_blending,           false,          "If blending is enabled, it will merge the results with the original frame. If disabled, it will only display the results.");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will hide it.");
DEFINE_double(alpha_heatmap,            0.7,            "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the heatmap, 0 will only show the frame.");
// Consumer
DEFINE_bool(fullscreen,                 false,          "Run in full-screen mode (press f during runtime to toggle).");
DEFINE_bool(process_real_time,          false,          "Enable to keep the original source frame rate (e.g. for video). If the processing time is too long, it will skip frames. If it is"
                                                        " too fast, it will slow it down.");
DEFINE_bool(no_gui_verbose,             false,          "Do not write text on output images on GUI (e.g. number of current frame and people). It does not affect the pose rendering.");
DEFINE_bool(no_display,                 false,          "Do not open a display window.");
DEFINE_string(write_images,             "",             "Directory to write rendered frames in `write_images_format` image format.");
DEFINE_string(write_images_format,      "png",          "File extension and format for `write_images`, e.g. png, jpg or bmp. Check the OpenCV function cv::imwrite"
                                                        " for all compatible extensions.");
DEFINE_string(write_video,              "",             "Full file path to write rendered frames in motion JPEG video format. It might fail if the final path does"
                                                        " not finish in `.avi`. It internally uses cv::VideoWriter.");
DEFINE_string(write_pose,               "",             "Directory to write the people pose data. Desired format on `write_pose_format`.");
DEFINE_string(write_pose_format,        "yml",          "File extension and format for `write_pose`: json, xml, yaml and yml. Json not available for OpenCV < 3.0,"
                                                        " use `write_pose_json` instead.");
DEFINE_string(write_pose_json,          "",             "Directory to write people pose data with *.json format, compatible with any OpenCV version.");
DEFINE_string(write_coco_json,          "",             "Full file path to write people pose data with *.json COCO validation format.");
DEFINE_string(write_heatmaps,           "",             "Directory to write heatmaps with *.png format. At least 1 `add_heatmaps_X` flag must be enabled.");
DEFINE_string(write_heatmaps_format,    "png",          "File extension and format for `write_heatmaps`, analogous to `write_images_format`. Recommended `png` or any compressed and"
                                                        " lossless format.");

op::PoseModel gflagToPoseModel(const std::string& poseModeString)
{
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    if (poseModeString == "COCO")
        return op::PoseModel::COCO_18;
    else if (poseModeString == "MPI")
        return op::PoseModel::MPI_15;
    else if (poseModeString == "MPI_4_layers")
        return op::PoseModel::MPI_15_4;
    else
    {
        op::error("String does not correspond to any model (COCO, MPI, MPI_4_layers)", __LINE__, __FUNCTION__, __FILE__);
        return op::PoseModel::COCO_18;
    }
}

op::ScaleMode gflagToScaleMode(const int scaleMode)
{
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    if (scaleMode == 0)
        return op::ScaleMode::InputResolution;
    else if (scaleMode == 1)
        return op::ScaleMode::NetOutputResolution;
    else if (scaleMode == 2)
        return op::ScaleMode::OutputResolution;
    else if (scaleMode == 3)
        return op::ScaleMode::ZeroToOne;
    else if (scaleMode == 4)
        return op::ScaleMode::PlusMinusOne;
    else
    {
        const std::string message = "String does not correspond to any scale mode: (0, 1, 2, 3, 4) for (InputResolution, NetOutputResolution, OutputResolution, ZeroToOne, PlusMinusOne).";
        op::error(message, __LINE__, __FUNCTION__, __FILE__);
        return op::ScaleMode::InputResolution;
    }
}

// Determine type of frame source
op::ProducerType gflagsToProducerType(const std::string& imageDirectory, const std::string& videoPath, const int webcamIndex)
{
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    // Avoid duplicates (e.g. selecting at the time camera & video)
    if (!imageDirectory.empty() && !videoPath.empty())
        op::error("Selected simultaneously image directory and video. Please, select only one.", __LINE__, __FUNCTION__, __FILE__);
    else if (!imageDirectory.empty() && webcamIndex != 0)
        op::error("Selected simultaneously image directory and webcam. Please, select only one.", __LINE__, __FUNCTION__, __FILE__);
    else if (!videoPath.empty() && webcamIndex != 0)
        op::error("Selected simultaneously video and webcam. Please, select only one.", __LINE__, __FUNCTION__, __FILE__);

    // Get desired op::ProducerType
    if (!imageDirectory.empty())
        return op::ProducerType::ImageDirectory;
    else if (!videoPath.empty())
        return op::ProducerType::Video;
    else
        return op::ProducerType::Webcam;
}

std::shared_ptr<op::Producer> gflagsToProducer(const std::string& imageDirectory, const std::string& videoPath, const int webcamIndex, const cv::Size webcamResolution)
{
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    const auto type = gflagsToProducerType(imageDirectory, videoPath, webcamIndex);

    if (type == op::ProducerType::ImageDirectory)
        return std::make_shared<op::ImageDirectoryReader>(imageDirectory);
    else if (type == op::ProducerType::Video)
        return std::make_shared<op::VideoReader>(videoPath);
    else if (type == op::ProducerType::Webcam)
        return std::make_shared<op::WebcamReader>(webcamIndex, webcamResolution);
    else
    {
        op::error("Undefined Producer selected.", __LINE__, __FUNCTION__, __FILE__);
        return std::shared_ptr<op::Producer>{};
    }
}

std::vector<op::HeatMapType> gflagToHeatMaps(const bool heatmaps_add_parts, const bool heatmaps_add_bkg, const bool heatmaps_add_PAFs)
{

    std::vector<op::HeatMapType> heatMapTypes;
    if (heatmaps_add_parts)
        heatMapTypes.emplace_back(op::HeatMapType::Parts);
    if (heatmaps_add_bkg)
        heatMapTypes.emplace_back(op::HeatMapType::Background);
    if (heatmaps_add_PAFs)
        heatMapTypes.emplace_back(op::HeatMapType::PAFs);
    return heatMapTypes;
}

// Google flags into program variables
std::tuple<cv::Size, cv::Size, cv::Size, std::shared_ptr<op::Producer>, op::PoseModel, op::ScaleMode, std::vector<op::HeatMapType>> gflagsToOpParameters()
{
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    // cameraFrameSize
    cv::Size cameraFrameSize;
    auto nRead = sscanf(FLAGS_camera_resolution.c_str(), "%dx%d", &cameraFrameSize.width, &cameraFrameSize.height);
    op::checkE(nRead, 2, "Error, camera resolution format (" +  FLAGS_camera_resolution + ") invalid, should be e.g., 1280x720", __LINE__, __FUNCTION__, __FILE__);
    // outputSize
    cv::Size outputSize;
    nRead = sscanf(FLAGS_resolution.c_str(), "%dx%d", &outputSize.width, &outputSize.height);
    op::checkE(nRead, 2, "Error, resolution format (" +  FLAGS_resolution + ") invalid, should be e.g., 960x540 ", __LINE__, __FUNCTION__, __FILE__);
    // netInputSize
    cv::Size netInputSize;
    nRead = sscanf(FLAGS_net_resolution.c_str(), "%dx%d", &netInputSize.width, &netInputSize.height);
    op::checkE(nRead, 2, "Error, net resolution format (" +  FLAGS_net_resolution + ") invalid, should be e.g., 656x368 (multiples of 16)", __LINE__, __FUNCTION__, __FILE__);
    // producerType
    const auto producerSharedPtr = gflagsToProducer(FLAGS_image_dir, FLAGS_video, FLAGS_camera, cameraFrameSize);
    // poseModel
    const auto poseModel = gflagToPoseModel(FLAGS_model_pose);
    // scaleMode
    const auto scaleMode = gflagToScaleMode(FLAGS_scale_mode);
    // heatmaps to add
    const auto heatMapTypes = gflagToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg, FLAGS_heatmaps_add_PAFs);
    // Return
    return std::make_tuple(cameraFrameSize, outputSize, netInputSize, producerSharedPtr, poseModel, scaleMode, heatMapTypes);
}

int opRealTimePoseDemo()
{
    // logging_level
    op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
    // op::ConfigureLog::setPriorityThreshold(op::Priority::None); // To print all logging messages

    op::log("Starting pose estimation demo.", op::Priority::Max);
    const auto timerBegin = std::chrono::high_resolution_clock::now();

    // Applying user defined configuration
    cv::Size cameraFrameSize;
    cv::Size outputSize;
    cv::Size netInputSize;
    std::shared_ptr<op::Producer> producerSharedPtr;
    op::PoseModel poseModel;
    op::ScaleMode scaleMode;
    std::vector<op::HeatMapType> heatMapTypes;
    std::tie(cameraFrameSize, outputSize, netInputSize, producerSharedPtr, poseModel, scaleMode, heatMapTypes) = gflagsToOpParameters();

    // OpenPose wrapper
    op::log("Configuring OpenPose wrapper.", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    op::Wrapper<std::vector<op::Datum>> opWrapper;
    const op::WrapperStructPose wrapperStructPose{netInputSize, outputSize, scaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start, FLAGS_num_scales, (float)FLAGS_scale_gap,
                                                  !FLAGS_no_render_output, poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
                                                  FLAGS_part_to_show, FLAGS_model_folder, heatMapTypes, op::ScaleMode::UnsignedChar};
    const op::WrapperStructInput wrapperStructInput{producerSharedPtr, FLAGS_frame_first, FLAGS_frame_last, FLAGS_process_real_time, FLAGS_frame_flip,
                                                    FLAGS_frame_rotate, FLAGS_frames_repeat};
    const op::WrapperStructOutput wrapperStructOutput{!FLAGS_no_display, !FLAGS_no_gui_verbose, FLAGS_fullscreen, FLAGS_write_pose, op::stringToDataFormat(FLAGS_write_pose_format),
                                                      FLAGS_write_pose_json, FLAGS_write_coco_json, FLAGS_write_images, FLAGS_write_images_format, FLAGS_write_video,
                                                      FLAGS_write_heatmaps, FLAGS_write_heatmaps_format};
    // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
    // Producer (use default to disable any input)
    // Consumer (comment or use default argument to disable any output)
    opWrapper.configure(wrapperStructPose, wrapperStructInput, wrapperStructOutput);
    // Set to single-thread running (e.g. for debugging purposes)
    // opWrapper.disableMultiThreading();

    // Start processing
    // Two different ways of running the program on multithread enviroment
    op::log("Starting thread(s)", op::Priority::Max);
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
    // op::log("Stopping thread(s)", op::Priority::Max);
    // opWrapper.stop();

    // Measuring total time
    const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-timerBegin).count() * 1e-9;
    const auto message = "Real-time pose estimation demo successfully finished. Total time: " + std::to_string(totalTimeSec) + " seconds.";
    op::log(message, op::Priority::Max);

    return 0;
}

int main(int argc, char *argv[])
{
    // Initializing google logging (Caffe uses it for logging)
    google::InitGoogleLogging("opRealTimePoseDemo");

    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running opRealTimePoseDemo
    return opRealTimePoseDemo();
}
