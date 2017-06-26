// ------------------------- OpenPose Library Tutorial - Thread - Example 2 - Synchronous -------------------------
// Synchronous mode: ideal for performance. The user can add his own frames producer / post-processor / consumer to the OpenPose wrapper or use the default ones.

// This example shows the user how to use the OpenPose wrapper class:
    // 1. Extract and render keypoint / heatmap / PAF of that image
    // 2. Save the results on disc
    // 3. Display the rendered pose
    // Everything in a multi-thread scenario
// In addition to the previous OpenPose modules, we also need to use:
    // 1. `core` module:
        // For the Array<float> class that the `pose` module needs
        // For the Datum struct that the `thread` module sends between the queues
    // 2. `utilities` module: for the error & logging functions, i.e. op::error & op::log respectively
// This file should only be used for the user to take specific examples.

// C++ std library dependencies
#include <atomic>
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <cstdio> // sscanf
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
DEFINE_string(image_dir,                "examples/media/",      "Process a directory of images.");
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
                                                        " with `num_scales` and `scale_gap`.");
// OpenPose Body Pose
DEFINE_string(model_pose,               "COCO",         "Model to be used (e.g. COCO, MPI, MPI_4_layers).");
DEFINE_string(net_resolution,           "656x368",      "Multiples of 16. If it is increased, the accuracy usually increases. If it is decreased,"
                                                        " the speed increases.");
DEFINE_int32(num_scales,                1,              "Number of scales to average.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless num_scales>1. Initial scale is always 1. If you"
                                                        " want to change the initial scale, you actually want to multiply the `net_resolution` by"
                                                        " your desired initial scale.");
DEFINE_bool(heatmaps_add_parts,         false,          "If true, it will add the body part heatmaps to the final op::Datum::poseHeatMaps array"
                                                        " (program speed will decrease). Not required for our library, enable it only if you intend"
                                                        " to process this information later. If more than one `add_heatmaps_X` flag is enabled, it"
                                                        " will place then in sequential memory order: body parts + bkg + PAFs. It will follow the"
                                                        " order on POSE_BODY_PART_MAPPING in `include/openpose/pose/poseParameters.hpp`.");
DEFINE_bool(heatmaps_add_bkg,           false,          "Same functionality as `add_heatmaps_parts`, but adding the heatmap corresponding to"
                                                        " background.");
DEFINE_bool(heatmaps_add_PAFs,          false,          "Same functionality as `add_heatmaps_parts`, but adding the PAFs.");
DEFINE_int32(heatmaps_scale,            2,              "Set 0 to scale op::Datum::poseHeatMaps in the range [0,1], 1 for [-1,1]; and 2 for integer"
                                                        " rounded [0,255].");
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
DEFINE_int32(hand_detection_mode,       0,              "Set to 0 to perform 1-time keypoint detection (fastest), 1 for iterative detection"
                                                        " (recommended for images and fast videos, slow method), 2 for tracking (recommended for"
                                                        " webcam if the frame rate is >10 FPS per GPU used and for video, in practice as fast as"
                                                        " 1-time detection), 3 for both iterative and tracking (recommended for webcam if the"
                                                        " resulting frame rate is still >10 FPS and for video, ideally best result but slower), or"
                                                        " -1 (default) for automatic selection (fast method for webcam, tracking for video and"
                                                        " iterative for images).");
// OpenPose Rendering
DEFINE_int32(part_to_show,              0,              "Part to show from the start.");
DEFINE_bool(disable_blending,           false,          "If blending is enabled, it will merge the results with the original frame. If disabled, it"
                                                        " will only display the results.");
// OpenPose Rendering Pose
DEFINE_int32(render_pose,               2,              "Set to 0 for no rendering, 1 for CPU rendering (slightly faster), and 2 for GPU rendering"
                                                        " (slower but greater functionality, e.g. `alpha_X` flags). If rendering is enabled, it will"
                                                        " render both `outputData` and `cvOutputData` with the original image and desired body part"
                                                        " to be shown (i.e. keypoints, heat maps or PAFs).");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
                                                        " hide it. Only valid for GPU rendering.");
DEFINE_double(alpha_heatmap,            0.7,            "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the"
                                                        " heatmap, 0 will only show the frame. Only valid for GPU rendering.");
// OpenPose Rendering Face
DEFINE_int32(render_face,               -1,             "Analogous to `render_pose` but applied to the face. Extra option: -1 to use the same"
                                                        " configuration that `render_pose` is using.");
DEFINE_double(alpha_face,               0.6,            "Analogous to `alpha_pose` but applied to face.");
DEFINE_double(alpha_heatmap_face,       0.7,            "Analogous to `alpha_heatmap` but applied to face.");
// OpenPose Rendering Hand
DEFINE_int32(render_hand,               -1,             "Analogous to `render_pose` but applied to the hand. Extra option: -1 to use the same"
                                                        " configuration that `render_pose` is using.");
DEFINE_double(alpha_hand,               0.6,            "Analogous to `alpha_pose` but applied to hand.");
DEFINE_double(alpha_heatmap_hand,       0.7,            "Analogous to `alpha_heatmap` but applied to hand.");
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


// If the user needs his own variables, he can inherit the op::Datum struct and add them
// UserDatum can be directly used by the OpenPose wrapper because it inherits from op::Datum, just define Wrapper<UserDatum> instead of
// Wrapper<op::Datum>
struct UserDatum : public op::Datum
{
    bool boolThatUserNeedsForSomeReason;

    UserDatum(const bool boolThatUserNeedsForSomeReason_ = false) :
        boolThatUserNeedsForSomeReason{boolThatUserNeedsForSomeReason_}
    {}
};

// The W-classes can be implemented either as a template or as simple classes given
// that the user usually knows which kind of data he will move between the queues,
// in this case we assume a std::shared_ptr of a std::vector of UserDatum

// This worker will just read and return all the jpg files in a directory
class WUserInput : public op::WorkerProducer<std::shared_ptr<std::vector<UserDatum>>>
{
public:
    WUserInput(const std::string& directoryPath) :
        mImageFiles{op::getFilesOnDirectory(directoryPath, "jpg")},
        // mImageFiles{op::getFilesOnDirectory(directoryPath, std::vector<std::string>{"jpg", "png"})}, // If we want "jpg" + "png" images
        mCounter{0}
    {
        if (mImageFiles.empty())
            op::error("No images found on: " + directoryPath, __LINE__, __FUNCTION__, __FILE__);
    }

    void initializationOnThread() {}

    std::shared_ptr<std::vector<UserDatum>> workProducer()
    {
        try
        {
            // Close program when empty frame
            if (mImageFiles.size() <= mCounter)
            {
                op::log("Last frame read and added to queue. Closing program after it is processed.", op::Priority::High);
                // This funtion stops this worker, which will eventually stop the whole thread system once all the frames have been processed
                this->stop();
                return nullptr;
            }
            else
            {
                // Create new datum
                auto datumsPtr = std::make_shared<std::vector<UserDatum>>();
                datumsPtr->emplace_back();
                auto& datum = datumsPtr->at(0);

                // Fill datum
                datum.cvInputData = cv::imread(mImageFiles.at(mCounter++));

                // If empty frame -> return nullptr
                if (datum.cvInputData.empty())
                {
                    op::log("Empty frame detected on path: " + mImageFiles.at(mCounter-1) + ". Closing program.", op::Priority::High);
                    this->stop();
                    datumsPtr = nullptr;
                }

                return datumsPtr;
            }
        }
        catch (const std::exception& e)
        {
            op::log("Some kind of unexpected error happened.");
            this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

private:
    const std::vector<std::string> mImageFiles;
    unsigned long long mCounter;
};

// This worker will just invert the image
class WUserPostProcessing : public op::Worker<std::shared_ptr<std::vector<UserDatum>>>
{
public:
    WUserPostProcessing()
    {
        // User's constructor here
    }

    void initializationOnThread() {}

    void work(std::shared_ptr<std::vector<UserDatum>>& datumsPtr)
    {
        // User's post-processing (after OpenPose processing & before OpenPose outputs) here
            // datum.cvOutputData: rendered frame with pose or heatmaps
            // datum.poseKeypoints: Array<float> with the estimated pose
        try
        {
            if (datumsPtr != nullptr && !datumsPtr->empty())
                for (auto& datum : *datumsPtr)
                    cv::bitwise_not(datum.cvOutputData, datum.cvOutputData);
        }
        catch (const std::exception& e)
        {
            op::log("Some kind of unexpected error happened.");
            this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
};

// This worker will just read and return all the jpg files in a directory
class WUserOutput : public op::WorkerConsumer<std::shared_ptr<std::vector<UserDatum>>>
{
public:
    void initializationOnThread() {}

    void workConsumer(const std::shared_ptr<std::vector<UserDatum>>& datumsPtr)
    {
        try
        {
            // User's displaying/saving/other processing here
                // datum.cvOutputData: rendered frame with pose or heatmaps
                // datum.poseKeypoints: Array<float> with the estimated pose
            if (datumsPtr != nullptr && !datumsPtr->empty())
            {
                cv::imshow("User worker GUI", datumsPtr->at(0).cvOutputData);
                cv::waitKey(1); // It displays the image and sleeps at least 1 ms (it usually sleeps ~5-10 msec to display the image)
            }
        }
        catch (const std::exception& e)
        {
            op::log("Some kind of unexpected error happened.");
            this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
};

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

op::ScaleMode gflagToScaleMode(const int keypointScale)
{
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    if (keypointScale == 0)
        return op::ScaleMode::InputResolution;
    else if (keypointScale == 1)
        return op::ScaleMode::NetOutputResolution;
    else if (keypointScale == 2)
        return op::ScaleMode::OutputResolution;
    else if (keypointScale == 3)
        return op::ScaleMode::ZeroToOne;
    else if (keypointScale == 4)
        return op::ScaleMode::PlusMinusOne;
    else
    {
        const std::string message = "String does not correspond to any scale mode: (0, 1, 2, 3, 4) for (InputResolution,"
                                    " NetOutputResolution, OutputResolution, ZeroToOne, PlusMinusOne).";
        op::error(message, __LINE__, __FUNCTION__, __FILE__);
        return op::ScaleMode::InputResolution;
    }
}

std::vector<op::HeatMapType> gflagToHeatMaps(const bool heatMapsAddParts, const bool heatMapsAddBkg, const bool heatMapsAddPAFs)
{
    std::vector<op::HeatMapType> heatMapTypes;
    if (heatMapsAddParts)
        heatMapTypes.emplace_back(op::HeatMapType::Parts);
    if (heatMapsAddBkg)
        heatMapTypes.emplace_back(op::HeatMapType::Background);
    if (heatMapsAddPAFs)
        heatMapTypes.emplace_back(op::HeatMapType::PAFs);
    return heatMapTypes;
}

op::DetectionMode gflagToDetectionMode(const int handDetectionModeFlag, const std::shared_ptr<op::Producer>& producer = nullptr)
{
    if (handDetectionModeFlag == -1)
    {
        if (producer == nullptr)
            op::error("Since there is no default producer, `hand_detection_mode` must be set.", __LINE__, __FUNCTION__, __FILE__);
        const auto producerType = producer->getType();
        if (producerType == op::ProducerType::Webcam)
            return op::DetectionMode::Fast;
        else if (producerType == op::ProducerType::ImageDirectory)
            return op::DetectionMode::Iterative;
        else if (producerType == op::ProducerType::Video)
            return op::DetectionMode::Tracking;

    }
    else if (handDetectionModeFlag == 0)
        return op::DetectionMode::Fast;
    else if (handDetectionModeFlag == 1)
        return op::DetectionMode::Iterative;
    else if (handDetectionModeFlag == 2)
        return op::DetectionMode::Tracking;
    else if (handDetectionModeFlag == 3)
        return op::DetectionMode::IterativeAndTracking;
    // else
    op::error("Undefined DetectionMode selected.", __LINE__, __FUNCTION__, __FILE__);
    return op::DetectionMode::Fast;
}

op::RenderMode gflagToRenderMode(const int renderFlag, const int renderPoseFlag = -2)
{
    if (renderFlag == -1 && renderPoseFlag != -2)
        return gflagToRenderMode(renderPoseFlag, -2);
    else if (renderFlag == 0)
        return op::RenderMode::None;
    else if (renderFlag == 1)
        return op::RenderMode::Cpu;
    else if (renderFlag == 2)
        return op::RenderMode::Gpu;
    else
    {
        op::error("Undefined RenderMode selected.", __LINE__, __FUNCTION__, __FILE__);
        return op::RenderMode::None;
    }
}

// Google flags into program variables
std::tuple<op::Point<int>, op::Point<int>, op::Point<int>, op::Point<int>, op::PoseModel, op::ScaleMode, std::vector<op::HeatMapType>,
           op::ScaleMode> gflagsToOpParameters()
{
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    // outputSize
    op::Point<int> outputSize;
    auto nRead = sscanf(FLAGS_resolution.c_str(), "%dx%d", &outputSize.x, &outputSize.y);
    op::checkE(nRead, 2, "Error, resolution format (" +  FLAGS_resolution + ") invalid, should be e.g., 960x540 ",
               __LINE__, __FUNCTION__, __FILE__);
    // netInputSize
    op::Point<int> netInputSize;
    nRead = sscanf(FLAGS_net_resolution.c_str(), "%dx%d", &netInputSize.x, &netInputSize.y);
    op::checkE(nRead, 2, "Error, net resolution format (" +  FLAGS_net_resolution + ") invalid, should be e.g., 656x368 (multiples of 16)",
               __LINE__, __FUNCTION__, __FILE__);
    // faceNetInputSize
    op::Point<int> faceNetInputSize;
    nRead = sscanf(FLAGS_face_net_resolution.c_str(), "%dx%d", &faceNetInputSize.x, &faceNetInputSize.y);
    op::checkE(nRead, 2, "Error, face net resolution format (" +  FLAGS_face_net_resolution
               + ") invalid, should be e.g., 368x368 (multiples of 16)", __LINE__, __FUNCTION__, __FILE__);
    // handNetInputSize
    op::Point<int> handNetInputSize;
    nRead = sscanf(FLAGS_hand_net_resolution.c_str(), "%dx%d", &handNetInputSize.x, &handNetInputSize.y);
    op::checkE(nRead, 2, "Error, hand net resolution format (" +  FLAGS_hand_net_resolution
               + ") invalid, should be e.g., 368x368 (multiples of 16)", __LINE__, __FUNCTION__, __FILE__);
    // poseModel
    const auto poseModel = gflagToPoseModel(FLAGS_model_pose);
    // keypointScale
    const auto keypointScale = gflagToScaleMode(FLAGS_keypoint_scale);
    // heatmaps to add
    const auto heatMapTypes = gflagToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg, FLAGS_heatmaps_add_PAFs);
    op::check(FLAGS_heatmaps_scale >= 0 && FLAGS_heatmaps_scale <= 2, "Non valid `heatmaps_scale`.", __LINE__, __FUNCTION__, __FILE__);
    const auto heatMapScale = (FLAGS_heatmaps_scale == 0 ? op::ScaleMode::PlusMinusOne
                               : (FLAGS_heatmaps_scale == 1 ? op::ScaleMode::ZeroToOne : op::ScaleMode::UnsignedChar ));
    // return
    return std::make_tuple(outputSize, netInputSize, faceNetInputSize, handNetInputSize, poseModel, keypointScale, heatMapTypes, heatMapScale);
}

int openPoseTutorialWrapper2()
{
    // logging_level
    op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
    // op::ConfigureLog::setPriorityThreshold(op::Priority::None); // To print all logging messages

    op::log("Starting pose estimation demo.", op::Priority::High);
    const auto timerBegin = std::chrono::high_resolution_clock::now();

    // Applying user defined configuration
    op::Point<int> outputSize;
    op::Point<int> netInputSize;
    op::Point<int> faceNetInputSize;
    op::Point<int> handNetInputSize;
    op::PoseModel poseModel;
    op::ScaleMode keypointScale;
    std::vector<op::HeatMapType> heatMapTypes;
    op::ScaleMode heatMapScale;
    std::tie(outputSize, netInputSize, faceNetInputSize, handNetInputSize, poseModel, keypointScale, heatMapTypes, heatMapScale) = gflagsToOpParameters();
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);

    // Initializing the user custom classes
    // Frames producer (e.g. video, webcam, ...)
    auto wUserInput = std::make_shared<WUserInput>(FLAGS_image_dir);
    // Processing
    auto wUserPostProcessing = std::make_shared<WUserPostProcessing>();
    // GUI (Display)
    auto wUserOutput = std::make_shared<WUserOutput>();

    op::Wrapper<std::vector<UserDatum>> opWrapper;
    // Add custom input
    const auto workerInputOnNewThread = false;
    opWrapper.setWorkerInput(wUserInput, workerInputOnNewThread);
    // Add custom processing
    const auto workerProcessingOnNewThread = false;
    opWrapper.setWorkerPostProcessing(wUserPostProcessing, workerProcessingOnNewThread);
    // Add custom output
    const auto workerOutputOnNewThread = true;
    opWrapper.setWorkerOutput(wUserOutput, workerOutputOnNewThread);
    // Configure OpenPose
    const op::WrapperStructPose wrapperStructPose{netInputSize, outputSize, keypointScale, FLAGS_num_gpu, FLAGS_num_gpu_start,
                                                  FLAGS_num_scales, (float)FLAGS_scale_gap, gflagToRenderMode(FLAGS_render_pose), poseModel,
                                                  !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
                                                  FLAGS_part_to_show, FLAGS_model_folder, heatMapTypes, heatMapScale};
    // Face configuration (use op::WrapperStructFace{} to disable it)
    const op::WrapperStructFace wrapperStructFace{FLAGS_face, faceNetInputSize, gflagToRenderMode(FLAGS_render_face, FLAGS_render_pose),
                                                  (float)FLAGS_alpha_face, (float)FLAGS_alpha_heatmap_face};
    // Hand configuration (use op::WrapperStructHand{} to disable it)
    const op::WrapperStructHand wrapperStructHand{FLAGS_hand, handNetInputSize, gflagToDetectionMode(FLAGS_hand_detection_mode),
                                                  gflagToRenderMode(FLAGS_render_hand, FLAGS_render_pose), (float)FLAGS_alpha_hand,
                                                  (float)FLAGS_alpha_heatmap_hand};
    // Consumer (comment or use default argument to disable any output)
    const bool displayGui = false;
    const bool guiVerbose = false;
    const bool fullScreen = false;
    const op::WrapperStructOutput wrapperStructOutput{displayGui, guiVerbose, fullScreen, FLAGS_write_keypoint,
                                                      op::stringToDataFormat(FLAGS_write_keypoint_format), FLAGS_write_keypoint_json,
                                                      FLAGS_write_coco_json, FLAGS_write_images, FLAGS_write_images_format, FLAGS_write_video,
                                                      FLAGS_write_heatmaps, FLAGS_write_heatmaps_format};
    // Configure wrapper
    opWrapper.configure(wrapperStructPose, wrapperStructFace, wrapperStructHand, op::WrapperStructInput{}, wrapperStructOutput);
    // Set to single-thread running (e.g. for debugging purposes)
    // opWrapper.disableMultiThreading();

    op::log("Starting thread(s)", op::Priority::High);
    // Two different ways of running the program on multithread environment
    // // Option a) Recommended - Also using the main thread (this thread) for processing (it saves 1 thread)
    // // Start, run & stop threads
    opWrapper.exec();  // It blocks this thread until all threads have finished

    // Option b) Keeping this thread free in case you want to do something else meanwhile, e.g. profiling the GPU memory
    // // VERY IMPORTANT NOTE: if OpenCV is compiled with Qt support, this option will not work. Qt needs the main thread to
    // // plot visual results, so the final GUI (which uses OpenCV) would return an exception similar to:
    // // `QMetaMethod::invoke: Unable to invoke methods with return values in queued connections`
    // // Start threads
    // opWrapper.start();
    // // Profile used GPU memory
    //     // 1: wait ~10sec so the memory has been totally loaded on GPU
    //     // 2: profile the GPU memory
    // std::this_thread::sleep_for(std::chrono::milliseconds{1000});
    // op::log("Random task here...", op::Priority::High);
    // // Keep program alive while running threads
    // while (opWrapper.isRunning())
    //     std::this_thread::sleep_for(std::chrono::milliseconds{33});
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
    google::InitGoogleLogging("openPoseTutorialWrapper2");

    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running openPoseTutorialWrapper2
    return openPoseTutorialWrapper2();
}
