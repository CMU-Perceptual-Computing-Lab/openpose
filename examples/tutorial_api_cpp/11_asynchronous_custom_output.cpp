// ------------------------- OpenPose C++ API Tutorial - Example 11 - Custom Output -------------------------
// Asynchronous mode: ideal for fast prototyping when performance is not an issue.
// In this function, the user can implement its own way to render/display/storage the results.

// Command-line user intraface
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

// Custom OpenPose flags
// Display
DEFINE_bool(no_display,                 false,
    "Enable to disable the visual display.");

// This worker will just read and return all the jpg files in a directory
class UserOutputClass
{
public:
    bool display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
    {
        // User's displaying/saving/other processing here
            // datumPtr->cvOutputData: rendered frame with pose or heatmaps
            // datumPtr->poseKeypoints: Array<float> with the estimated pose
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", datumsPtr->at(0)->cvOutputData);
            // Display image and sleeps at least 1 ms (it usually sleeps ~5-10 msec to display the image)
        }
        else
            op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
        const auto key = (char)cv::waitKey(1);
        return (key == 27);
    }
    void printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
    {
        // Example: How to use the pose keypoints
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            op::log("\nKeypoints:");
            // Accesing each element of the keypoints
            const auto& poseKeypoints = datumsPtr->at(0)->poseKeypoints;
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
            op::log("Face keypoints: " + datumsPtr->at(0)->faceKeypoints.toString(), op::Priority::High);
            op::log("Left hand keypoints: " + datumsPtr->at(0)->handKeypoints[0].toString(), op::Priority::High);
            op::log("Right hand keypoints: " + datumsPtr->at(0)->handKeypoints[1].toString(), op::Priority::High);
            // Heatmaps
            const auto& poseHeatMaps = datumsPtr->at(0)->poseHeatMaps;
            if (!poseHeatMaps.empty())
            {
                op::log("Pose heatmaps size: [" + std::to_string(poseHeatMaps.getSize(0)) + ", "
                        + std::to_string(poseHeatMaps.getSize(1)) + ", "
                        + std::to_string(poseHeatMaps.getSize(2)) + "]");
                const auto& faceHeatMaps = datumsPtr->at(0)->faceHeatMaps;
                op::log("Face heatmaps size: [" + std::to_string(faceHeatMaps.getSize(0)) + ", "
                        + std::to_string(faceHeatMaps.getSize(1)) + ", "
                        + std::to_string(faceHeatMaps.getSize(2)) + ", "
                        + std::to_string(faceHeatMaps.getSize(3)) + "]");
                const auto& handHeatMaps = datumsPtr->at(0)->handHeatMaps;
                op::log("Left hand heatmaps size: [" + std::to_string(handHeatMaps[0].getSize(0)) + ", "
                        + std::to_string(handHeatMaps[0].getSize(1)) + ", "
                        + std::to_string(handHeatMaps[0].getSize(2)) + ", "
                        + std::to_string(handHeatMaps[0].getSize(3)) + "]");
                op::log("Right hand heatmaps size: [" + std::to_string(handHeatMaps[1].getSize(0)) + ", "
                        + std::to_string(handHeatMaps[1].getSize(1)) + ", "
                        + std::to_string(handHeatMaps[1].getSize(2)) + ", "
                        + std::to_string(handHeatMaps[1].getSize(3)) + "]");
            }
        }
        else
            op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
};

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
        // producerType
        op::ProducerType producerType;
        std::string producerString;
        std::tie(producerType, producerString) = op::flagsToProducer(
            FLAGS_image_dir, FLAGS_video, FLAGS_ip_camera, FLAGS_camera, FLAGS_flir_camera, FLAGS_flir_camera_index);
        // cameraSize
        const auto cameraSize = op::flagsToPoint(FLAGS_camera_resolution, "-1x-1");
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
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1 || FLAGS_flir_camera);
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
        // Producer (use default to disable any input)
        const op::WrapperStructInput wrapperStructInput{
            producerType, producerString, FLAGS_frame_first, FLAGS_frame_step, FLAGS_frame_last,
            FLAGS_process_real_time, FLAGS_frame_flip, FLAGS_frame_rotate, FLAGS_frames_repeat,
            cameraSize, FLAGS_camera_parameter_path, FLAGS_frame_undistort, FLAGS_3d_views};
        opWrapper.configure(wrapperStructInput);
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
        op::Wrapper opWrapper{op::ThreadManagerMode::AsynchronousOut};
        configureWrapper(opWrapper);

        // Start, run, and stop processing - exec() blocks this thread until OpenPose wrapper has finished
        op::log("Starting thread(s)...", op::Priority::High);
        opWrapper.start();

        // User processing
        UserOutputClass userOutputClass;
        bool userWantsToExit = false;
        while (!userWantsToExit)
        {
            // Pop frame
            std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> datumProcessed;
            if (opWrapper.waitAndPop(datumProcessed))
            {
                if (!FLAGS_no_display)
                    userWantsToExit = userOutputClass.display(datumProcessed);;
                userOutputClass.printKeypoints(datumProcessed);
            }
            // If OpenPose finished reading images
            else if (!opWrapper.isRunning())
                break;
            // Something else happened
            else
                op::log("Processed datum could not be emplaced.", op::Priority::High);
        }

        op::log("Stopping thread(s)", op::Priority::High);
        opWrapper.stop();

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
