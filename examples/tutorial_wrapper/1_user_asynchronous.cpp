// ------------------------- OpenPose Library Tutorial - Thread - Example 1 - Asynchronous -------------------------
// Asynchronous mode: ideal for fast prototyping when performance is not an issue. The user emplaces/pushes and pops frames from the OpenPose wrapper when he desires to.

// This example shows the user how to use the OpenPose wrapper class:
    // 1. Extract and render pose / heatmap / PAF of that image
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
DEFINE_string(image_dir,                "examples/media/",      "Process a directory of images.");
// OpenPose
DEFINE_string(model_pose,               "COCO",         "Model to be used (e.g. COCO, MPI, MPI_4_layers).");
DEFINE_string(model_folder,             "models/",      "Folder where the pose models (COCO and MPI) are located.");
DEFINE_string(net_resolution,           "656x368",      "Multiples of 16.");
DEFINE_string(resolution,               "1280x720",     "The image resolution (display and output).");
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
DEFINE_int32(heatmaps_scale_mode,       2,              "Set 0 to scale op::Datum::poseHeatMaps in the range [0,1], 1 for [-1,1]; and 2 for integer rounded [0,255].");
// OpenPose Rendering
DEFINE_bool(no_render_output,           false,          "If false, it will fill both `outputData` and `cvOutputData` with the original image + desired part to be shown."
                                                        " If true, it will leave them empty.");
DEFINE_int32(part_to_show,              0,              "Part to show from the start.");
DEFINE_bool(disable_blending,           false,          "If false, it will blend the results with the original frame. If true, it will only display the results.");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will hide it.");
DEFINE_double(alpha_heatmap,            0.7,            "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the heatmap, 0 will only show the frame.");
// Consumer
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


// If the user needs his own variables, he can inherit the op::Datum struct and add them
// UserDatum can be directly used by the OpenPose wrapper because it inherits from op::Datum, just define Wrapper<UserDatum> instead of Wrapper<op::Datum>
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
class UserInputClass
{
public:
    UserInputClass(const std::string& directoryPath) :
        mImageFiles{op::getFilesOnDirectory(directoryPath, "jpg")},
        // mImageFiles{op::getFilesOnDirectory(directoryPath, std::vector<std::string>{"jpg", "png"})}, // If we want "jpg" + "png" images
        mCounter{0},
        mClosed{false}
    {
        if (mImageFiles.empty())
            op::error("No images found on: " + directoryPath, __LINE__, __FUNCTION__, __FILE__);
    }

    std::shared_ptr<std::vector<UserDatum>> createDatum()
    {
        // Close program when empty frame
        if (mClosed || mImageFiles.size() <= mCounter)
        {
            op::log("Last frame read and added to queue. Closing program after it is processed.", op::Priority::Max);
            // This funtion stops this worker, which will eventually stop the whole thread system once all the frames have been processed
            mClosed = true;
            return nullptr;
        }
        else // if (!mClosed)
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
                op::log("Empty frame detected on path: " + mImageFiles.at(mCounter-1) + ". Closing program.", op::Priority::Max);
                mClosed = true;
                datumsPtr = nullptr;
            }

            return datumsPtr;
        }
    }

    bool isFinished() const
    {
        return mClosed;
    }

private:
    const std::vector<std::string> mImageFiles;
    unsigned long long mCounter;
    bool mClosed;
};

// This worker will just read and return all the jpg files in a directory
class UserOutputClass
{
public:
    void display(const std::shared_ptr<std::vector<UserDatum>>& datumsPtr)
    {
        // User's displaying/saving/other processing here
            // datum.cvOutputData: rendered frame with pose or heatmaps
            // datum.poseKeyPoints: Array<float> with the estimated pose
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            cv::imshow("User worker GUI", datumsPtr->at(0).cvOutputData);
            cv::waitKey(500); // It sleeps 500 ms just to let the user see the output. Change to 33ms for normal 30 fps display
        }
        else
            op::log("Nullptr or empty datumsPtr found.", op::Priority::Max, __LINE__, __FUNCTION__, __FILE__);
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
std::tuple<cv::Size, cv::Size, op::PoseModel, op::ScaleMode, std::vector<op::HeatMapType>, op::ScaleMode> gflagsToOpParameters()
{
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    // outputSize
    cv::Size outputSize;
    auto nRead = sscanf(FLAGS_resolution.c_str(), "%dx%d", &outputSize.width, &outputSize.height);
    op::checkE(nRead, 2, "Error, resolution format (" +  FLAGS_resolution + ") invalid, should be e.g., 960x540 ", __LINE__, __FUNCTION__, __FILE__);
    // netInputSize
    cv::Size netInputSize;
    nRead = sscanf(FLAGS_net_resolution.c_str(), "%dx%d", &netInputSize.width, &netInputSize.height);
    op::checkE(nRead, 2, "Error, net resolution format (" +  FLAGS_net_resolution + ") invalid, should be e.g., 656x368 (multiples of 16)", __LINE__, __FUNCTION__, __FILE__);
    // poseModel
    const auto poseModel = gflagToPoseModel(FLAGS_model_pose);
    // scaleMode
    const auto scaleMode = gflagToScaleMode(FLAGS_scale_mode);
    // heatmaps to add
    const auto heatMapTypes = gflagToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg, FLAGS_heatmaps_add_PAFs);
    op::check(FLAGS_heatmaps_scale_mode >= 0 && FLAGS_heatmaps_scale_mode <= 2, "Non valid `heatmaps_scale_mode`.", __LINE__, __FUNCTION__, __FILE__);
    const auto heatMapsScaleMode = (FLAGS_heatmaps_scale_mode == 0 ? op::ScaleMode::PlusMinusOne : (FLAGS_heatmaps_scale_mode == 1 ? op::ScaleMode::ZeroToOne : op::ScaleMode::UnsignedChar ));
    // return
    return std::make_tuple(outputSize, netInputSize, poseModel, scaleMode, heatMapTypes, heatMapsScaleMode);
}

int openPoseTutorialWrapper1()
{
    // logging_level
    op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
    // op::ConfigureLog::setPriorityThreshold(op::Priority::None); // To print all logging messages

    op::log("Starting pose estimation demo.", op::Priority::Max);
    const auto timerBegin = std::chrono::high_resolution_clock::now();

    // Applying user defined configuration
    cv::Size outputSize;
    cv::Size netInputSize;
    op::PoseModel poseModel;
    op::ScaleMode scaleMode;
    std::vector<op::HeatMapType> heatMapTypes;
    op::ScaleMode heatMapsScaleMode;
    std::tie(outputSize, netInputSize, poseModel, scaleMode, heatMapTypes, heatMapsScaleMode) = gflagsToOpParameters();
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);

    // Configure OpenPose
    op::Wrapper<std::vector<UserDatum>> opWrapper{op::ThreadManagerMode::Asynchronous};
    const bool displayGui = false;
    const bool guiVerbose = false;
    const bool fullScreen = false;
    const op::WrapperStructPose wrapperStructPose{netInputSize, outputSize, scaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start, FLAGS_num_scales, (float)FLAGS_scale_gap,
                                                  !FLAGS_no_render_output, poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
                                                  FLAGS_part_to_show, FLAGS_model_folder, heatMapTypes, heatMapsScaleMode};
    const op::WrapperStructOutput wrapperStructOutput{displayGui, guiVerbose, fullScreen, FLAGS_write_pose, op::stringToDataFormat(FLAGS_write_pose_format),
                                                      FLAGS_write_pose_json, FLAGS_write_coco_json, FLAGS_write_images, FLAGS_write_images_format, FLAGS_write_video,
                                                      FLAGS_write_heatmaps, FLAGS_write_heatmaps_format};
    // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
    // Producer (use default to disable any input)
    // Consumer (comment or use default argument to disable any output)
    opWrapper.configure(wrapperStructPose, op::WrapperStructInput{}, wrapperStructOutput);
    // Set to single-thread running (e.g. for debugging purposes)
    // opWrapper.disableMultiThreading();

    op::log("Starting thread(s)", op::Priority::Max);
    opWrapper.start();

    // User processing
    UserInputClass userInputClass(FLAGS_image_dir);
    UserOutputClass userOutputClass;
    while (!userInputClass.isFinished())
    {
        // Push frame
        auto datumToProcess = userInputClass.createDatum();
        if (datumToProcess != nullptr)
        {
            auto successfullyEmplaced = opWrapper.waitAndEmplace(datumToProcess);
            // Pop frame
            std::shared_ptr<std::vector<UserDatum>> datumProcessed;
            if (successfullyEmplaced && opWrapper.waitAndPop(datumProcessed))
                userOutputClass.display(datumProcessed);
            else
                op::log("Processed datum could not be emplaced.", op::Priority::Max, __LINE__, __FUNCTION__, __FILE__);
        }
    }

    op::log("Stopping thread(s)", op::Priority::Max);
    opWrapper.stop();

    // Measuring total time
    const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-timerBegin).count() * 1e-9;
    const auto message = "Real-time pose estimation demo successfully finished. Total time: " + std::to_string(totalTimeSec) + " seconds.";
    op::log(message, op::Priority::Max);

    return 0;
}

int main(int argc, char *argv[])
{
    // Initializing google logging (Caffe uses it for logging)
    google::InitGoogleLogging("openPoseTutorialWrapper1");

    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running openPoseTutorialWrapper1
    return openPoseTutorialWrapper1();
}
