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
DEFINE_int32(logging_level,             3,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                                                        " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                                                        " low priority messages and 4 for important ones.");
// Producer
DEFINE_string(image_dir,                "examples/media/",      "Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).");
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
DEFINE_string(model_pose,               "COCO",         "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
                                                        "`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(net_resolution,           "656x368",      "Multiples of 16. If it is increased, the accuracy potentially increases. If it is decreased,"
                                                        " the speed increases. For maximum speed-accuracy balance, it should keep the closest aspect"
                                                        " ratio possible to the images or videos to be processed. E.g. the default `656x368` is"
                                                        " optimal for 16:9 videos, e.g. full HD (1980x1080) and HD (1280x720) videos.");
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
DEFINE_int32(heatmaps_scale,            2,              "Set 0 to scale op::Datum::poseHeatMaps in the range [0,1], 1 for [-1,1]; and 2 for integer"
                                                        " rounded [0,255].");
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
                                                        " were found with `hand_scale_number` = 6 and `hand_scale_range` = 0.4");
DEFINE_double(hand_scale_range,         0.4,            "Analogous purpose than `scale_gap` but applied to the hand keypoint detector. Total range"
                                                        " between smallest and biggest scale. The scales will be centered in ratio 1. E.g. if"
                                                        " scaleRange = 0.4 and scalesNumber = 2, then there will be 2 scales, 0.8 and 1.2.");

DEFINE_bool(hand_tracking,              false,          "Adding hand tracking might improve hand keypoints detection for webcam (if the frame rate"
                                                        " is high enough, i.e. >7 FPS per GPU) and video. This is not person ID tracking, it"
                                                        " simply looks for hands in positions at which hands were located in previous frames, but"
                                                        " it does not guarantee the same person ID among frames");
// OpenPose Rendering
DEFINE_int32(part_to_show,              0,              "Prediction channel to visualize (default: 0). 0 for all the body parts, 1-18 for each body"
                                                        " part heat map, 19 for the background heat map, 20 for all the body part heat maps"
                                                        " together, 21 for all the PAFs, 22-40 for each body part pair PAF");
DEFINE_bool(disable_blending,           false,          "If blending is enabled, it will merge the results with the original frame. If disabled, it"
                                                        " will only display the results on a black background.");
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
                // Show in command line the resulting pose keypoints for body, face and hands
                op::log("\nKeypoints:");
                // Accesing each element of the keypoints
                const auto& poseKeypoints = datumsPtr->at(0).poseKeypoints;
                op::log("Person pose keypoints:");
                for (auto person = 0 ; person < poseKeypoints.getSize(0) ; person++)
                {
                    op::log("Person " + std::to_string(person) + " (x, y, score):");
                    for (auto bodyPart = 0 ; bodyPart < poseKeypoints.getSize(1) ; bodyPart++)
                    {
                        std::string valueToPrint;
                        for (auto xyscore = 0 ; xyscore < poseKeypoints.getSize(2) ; xyscore++)
                        {
                            valueToPrint += std::to_string(   poseKeypoints[{person, bodyPart, xyscore}]   ) + " ";
                        }
                        op::log(valueToPrint);
                    }
                }
                op::log(" ");
                // Alternative: just getting std::string equivalent
                op::log("Face keypoints: " + datumsPtr->at(0).faceKeypoints.toString());
                op::log("Left hand keypoints: " + datumsPtr->at(0).handKeypoints[0].toString());
                op::log("Right hand keypoints: " + datumsPtr->at(0).handKeypoints[1].toString());

                // Display rendered output image
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

int openPoseTutorialWrapper2()
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
    // poseModel
    const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
    // keypointScale
    const auto keypointScale = op::flagsToScaleMode(FLAGS_keypoint_scale);
    // heatmaps to add
    const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg, FLAGS_heatmaps_add_PAFs);
    op::check(FLAGS_heatmaps_scale >= 0 && FLAGS_heatmaps_scale <= 2, "Non valid `heatmaps_scale`.", __LINE__, __FUNCTION__, __FILE__);
    const auto heatMapScale = (FLAGS_heatmaps_scale == 0 ? op::ScaleMode::PlusMinusOne
                               : (FLAGS_heatmaps_scale == 1 ? op::ScaleMode::ZeroToOne : op::ScaleMode::UnsignedChar ));
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
    const op::WrapperStructPose wrapperStructPose{netInputSize, outputSize, keypointScale, FLAGS_num_gpu,
                                                  FLAGS_num_gpu_start, FLAGS_scale_number, (float)FLAGS_scale_gap,
                                                  op::flagsToRenderMode(FLAGS_render_pose), poseModel,
                                                  !FLAGS_disable_blending, (float)FLAGS_alpha_pose,
                                                  (float)FLAGS_alpha_heatmap, FLAGS_part_to_show, FLAGS_model_folder,
                                                  heatMapTypes, heatMapScale, (float)FLAGS_render_threshold};
    // Face configuration (use op::WrapperStructFace{} to disable it)
    const op::WrapperStructFace wrapperStructFace{FLAGS_face, faceNetInputSize, op::flagsToRenderMode(FLAGS_face_render, FLAGS_render_pose),
                                                  (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
    // Hand configuration (use op::WrapperStructHand{} to disable it)
    const op::WrapperStructHand wrapperStructHand{FLAGS_hand, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
                                                  FLAGS_hand_tracking, op::flagsToRenderMode(FLAGS_hand_render, FLAGS_render_pose),
                                                  (float)FLAGS_hand_alpha_pose, (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
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
