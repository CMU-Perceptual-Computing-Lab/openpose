// --------------- OpenPose C++ API Tutorial - Example 5 - Body from images and multi GPU ---------------
// It reads images, process them, and display them with the pose (and optionally hand and face) keypoints. In addition,
// it includes all the OpenPose configuration flags (enable/disable hand, face, output saving, etc.).

// Command-line user intraface
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

// Custom OpenPose flags
// Producer
DEFINE_string(image_dir,                "examples/media/",
    "Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).");
// OpenPose
DEFINE_bool(latency_is_irrelevant_and_computer_with_lots_of_ram, false,
    "If false, it will read and then then process images right away. If true, it will first store all the frames and"
    " later process them (slightly faster). However: 1) Latency will hugely increase (no frames will be processed"
    " until they have all been read). And 2) The program might go out of RAM memory with long videos or folders with"
    " many images (so the computer might freeze).");
// Display
DEFINE_bool(no_display,                 false,
    "Enable to disable the visual display.");

// This worker will just read and return all the jpg files in a directory
bool display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try
    {
        // User's displaying/saving/other processing here
            // datum.cvOutputData: rendered frame with pose or heatmaps
            // datum.poseKeypoints: Array<float> with the estimated pose
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            // Display image and sleeps at least 1 ms (it usually sleeps ~5-10 msec to display the image)
            cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", datumsPtr->at(0)->cvOutputData);
        }
        else
            op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
        const auto key = (char)cv::waitKey(1);
        return (key == 27);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return true;
    }
}

void printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try
    {
        // Example: How to use the pose keypoints
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            op::log("Body keypoints: " + datumsPtr->at(0)->poseKeypoints.toString(), op::Priority::High);
            op::log("Face keypoints: " + datumsPtr->at(0)->faceKeypoints.toString(), op::Priority::High);
            op::log("Left hand keypoints: " + datumsPtr->at(0)->handKeypoints[0].toString(), op::Priority::High);
            op::log("Right hand keypoints: " + datumsPtr->at(0)->handKeypoints[1].toString(), op::Priority::High);
        }
        else
            op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void configureWrapper(op::Wrapper& opWrapper)
{
    try
    {
        // Configuring OpenPose

        // logging_level
        op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
                  __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // outputSize
        const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(FLAGS_face_net_resolution, "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(FLAGS_hand_net_resolution, "368x368 (multiples of 16)");
        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::log("Flag `write_keypoint` is deprecated and will eventually be removed."
                    " Please, use `write_json` instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, outputSize, keypointScaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start,
            FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
            poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
            FLAGS_part_to_show, FLAGS_model_folder, heatMapTypes, heatMapScaleMode, FLAGS_part_candidates,
            (float)FLAGS_render_threshold, FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max,
            FLAGS_prototxt_path, FLAGS_caffemodel_path, (float)FLAGS_upsampling_ratio, enableGoogleLogging};
        opWrapper.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
            FLAGS_face, faceDetector, faceNetInputSize,
            op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
            (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
        opWrapper.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
            op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
            (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
        opWrapper.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
        opWrapper.configure(wrapperStructExtra);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, FLAGS_write_keypoint, op::stringToDataFormat(FLAGS_write_keypoint_format),
            FLAGS_write_json, FLAGS_write_coco_json, FLAGS_write_coco_json_variants, FLAGS_write_coco_json_variant,
            FLAGS_write_images, FLAGS_write_images_format, FLAGS_write_video, FLAGS_write_video_fps,
            FLAGS_write_video_with_audio, FLAGS_write_heatmaps, FLAGS_write_heatmaps_format, FLAGS_write_video_3d,
            FLAGS_write_video_adam, FLAGS_write_bvh, FLAGS_udp_host, FLAGS_udp_port};
        opWrapper.configure(wrapperStructOutput);
        // No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

int tutorialApiCpp()
{
    try
    {
        op::log("Starting OpenPose demo...", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // Configuring OpenPose
        op::log("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
        configureWrapper(opWrapper);
        // Increase maximum wrapper queue size
        if (FLAGS_latency_is_irrelevant_and_computer_with_lots_of_ram)
            opWrapper.setDefaultMaxSizeQueues(std::numeric_limits<long long>::max());

        // Starting OpenPose
        op::log("Starting thread(s)...", op::Priority::High);
        opWrapper.start();

        // Read frames on directory
        const auto imagePaths = op::getFilesOnDirectory(FLAGS_image_dir, op::Extensions::Images);

        // Process and display images
        // Option a) Harder to implement but the fastest method
        // Create 2 different threads:
        //     1. One pushing images to OpenPose all the time.
        //     2. A second one retrieving those frames.
        // Option b) Much easier and faster to implement but slightly slower runtime performance
        if (!FLAGS_latency_is_irrelevant_and_computer_with_lots_of_ram)
        {
            // Read number of GPUs in your system
            const auto numberGPUs = op::getGpuNumber();

            for (auto imageBaseId = 0u ; imageBaseId < imagePaths.size() ; imageBaseId+=numberGPUs)
            {
                // Read and push images into OpenPose wrapper
                for (auto gpuId = 0 ; gpuId < numberGPUs ; gpuId++)
                {
                    const auto imageId = imageBaseId+gpuId;
                    if (imageId < imagePaths.size())
                    {
                        const auto& imagePath = imagePaths.at(imageId);
                        // Faster alternative that moves imageToProcess
                        auto imageToProcess = cv::imread(imagePath);
                        opWrapper.waitAndEmplace(imageToProcess);
                        // // Slower but safer alternative that copies imageToProcess
                        // const auto imageToProcess = cv::imread(imagePath);
                        // opWrapper.waitAndPush(imageToProcess);
                    }
                }
                // Retrieve processed results from OpenPose wrapper
                for (auto gpuId = 0 ; gpuId < numberGPUs ; gpuId++)
                {
                    const auto imageId = imageBaseId+gpuId;
                    if (imageId < imagePaths.size())
                    {
                        std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> datumProcessed;
                        const auto status = opWrapper.waitAndPop(datumProcessed);
                        if (status && datumProcessed != nullptr)
                        {
                            printKeypoints(datumProcessed);
                            if (!FLAGS_no_display)
                            {
                                const auto userWantsToExit = display(datumProcessed);
                                if (userWantsToExit)
                                {
                                    op::log("User pressed Esc to exit demo.", op::Priority::High);
                                    break;
                                }
                            }
                        }
                        else
                            op::log("Image could not be processed.", op::Priority::High);
                    }
                }
            }
        }
        // Option c) Even easier and faster to implement than option b. In addition, its runtime performance should
        // be slightly faster too, but:
        //  - Latency will hugely increase (no frames will be processed until they have all been read).
        //  - The program might go out of RAM memory with long videos or folders with many images (so the computer
        //    might freeze).
        else
        {
            // Read and push all images into OpenPose wrapper
            op::log("Loading images into OpenPose wrapper...", op::Priority::High);
            for (const auto& imagePath : imagePaths)
            {
                // Faster alternative that moves imageToProcess
                auto imageToProcess = cv::imread(imagePath);
                opWrapper.waitAndEmplace(imageToProcess);
                // // Slower but safer alternative that copies imageToProcess
                // const auto imageToProcess = cv::imread(imagePath);
                // opWrapper.waitAndPush(imageToProcess);
            }
            // Retrieve processed results from OpenPose wrapper
            op::log("Retrieving results from OpenPose wrapper...", op::Priority::High);
            for (auto imageId = 0u ; imageId < imagePaths.size() ; imageId++)
            {
                std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> datumProcessed;
                const auto status = opWrapper.waitAndPop(datumProcessed);
                if (status && datumProcessed != nullptr)
                {
                    printKeypoints(datumProcessed);
                    if (!FLAGS_no_display)
                    {
                        const auto userWantsToExit = display(datumProcessed);
                        if (userWantsToExit)
                        {
                            op::log("User pressed Esc to exit demo.", op::Priority::High);
                            break;
                        }
                    }
                }
                else
                    op::log("Image could not be processed.", op::Priority::High);
            }
        }

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Return
        return 0;
    }
    catch (const std::exception& e)
    {
        return -1;
    }
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running tutorialApiCpp
    return tutorialApiCpp();
}
