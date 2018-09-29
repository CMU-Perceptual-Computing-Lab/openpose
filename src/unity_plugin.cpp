#ifdef USE_UNITY_SUPPORT
// ------------------------- OpenPose Unity Binding -------------------------

// Command-line user intraface
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

// Output callback register in Unity
typedef void(__stdcall * OutputCallback) (const char* str, int type);

// This worker will just read and return all the jpg files in a directory
class UnityPluginUserOutput : public op::WorkerConsumer<std::shared_ptr<std::vector<op::Datum>>>
{
public:
	OutputCallback gOutputCallback;
	bool gOutputEnabled = true;

    void initializationOnThread() {}

	void outputTest() {
		outputToUnity("test_output", -1);
	}

    void terminate() { // Tianyi's code
		op::log("OP_End");
        this->stop();
    }

protected:
    // const char* keypointsData;
    // uchar* imageData;

    std::string vectorToJson(const float x, const float y, const float z) {
        return "{\"x\":" + std::to_string(x) + ",\"y\":" + std::to_string(y) + ",\"z\":" + std::to_string(z) + "}";
    }

    std::string vectorToJson(const int x, const int y, const int z) {
        return "{\"x\":" + std::to_string(x) + ",\"y\":" + std::to_string(y) + ",\"z\":" + std::to_string(z) + "}";
    }

	void outputToUnity(const std::string& message, const int type)
	{
		if (gOutputCallback) {
			try {
				gOutputCallback(message.c_str(), type);
			}
			catch (const std::exception& e)
			{
				op::log("Output error");
				op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
			}
		}
	}

    void sendData(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr) { // Tianyi's code
        const auto& poseKeypoints = datumsPtr->at(0).poseKeypoints;
        const auto& handKeypoints_L = datumsPtr->at(0).handKeypoints[0];
        const auto& handKeypoints_R = datumsPtr->at(0).handKeypoints[1];
        const auto& faceKeypoints = datumsPtr->at(0).faceKeypoints;
        const auto personCount = poseKeypoints.getSize(0);

        std::string dataString = "";

        std::string unitsString = "\"units\":[";
        for (auto person = 0; person < personCount; person++)
        {
            // Every person

            // poseKeypoints:
            std::string poseKeypointsString = "\"poseKeypoints\":[";
            for (auto part = 0; part < poseKeypoints.getSize(1); part++)
            {
                // Every part
                std::string partString = "";
                if (poseKeypoints.getSize(2) == 3) {
                    float x = poseKeypoints[{person, part, 0}],
                        y = poseKeypoints[{person, part, 1}],
                        z = poseKeypoints[{person, part, 2}];
                    partString = vectorToJson(x, y, z);
                }
                else {
                    partString = vectorToJson(0.0f, 0.0f, 0.0f);
                }
                poseKeypointsString += partString;
                if (part != poseKeypoints.getSize(1) - 1) poseKeypointsString += ",";
            }
            poseKeypointsString += "]";

            // handKeypoints_L:
            std::string handKeypointsString_L = "\"handKeypoints_L\":[";
            for (auto part = 0; part < handKeypoints_L.getSize(1); part++)
            {
                // Every part
                std::string partString = "";
                if (handKeypoints_L.getSize(2) == 3) {
                    float x = handKeypoints_L[{person, part, 0}],
                        y = handKeypoints_L[{person, part, 1}],
                        z = handKeypoints_L[{person, part, 2}];
                    partString = vectorToJson(x, y, z);
                }
                else {
                    partString = vectorToJson(0.0f, 0.0f, 0.0f);
                }
                handKeypointsString_L += partString;
                if (part != handKeypoints_L.getSize(1) - 1) handKeypointsString_L += ",";
            }
            handKeypointsString_L += "]";

            // handKeypoints_R:
            std::string handKeypointsString_R = "\"handKeypoints_R\":[";
            for (auto part = 0; part < handKeypoints_R.getSize(1); part++)
            {
                // Every part
                std::string partString = "";
                if (handKeypoints_R.getSize(2) == 3) {
                    float x = handKeypoints_R[{person, part, 0}],
                        y = handKeypoints_R[{person, part, 1}],
                        z = handKeypoints_R[{person, part, 2}];
                    partString = vectorToJson(x, y, z);
                }
                else {
                    partString = vectorToJson(0.0f, 0.0f, 0.0f);
                }
                handKeypointsString_R += partString;
                if (part != handKeypoints_R.getSize(1) - 1) handKeypointsString_R += ",";
            }
            handKeypointsString_R += "]";

            // faceKeypoints:
            std::string faceKeypointsString = "\"faceKeypoints\":[";
            for (auto part = 0; part < faceKeypoints.getSize(1); part++)
            {
                // Every part
                std::string partString = "";
                if (faceKeypoints.getSize(2) == 3) {
                    float x = faceKeypoints[{person, part, 0}],
                        y = faceKeypoints[{person, part, 1}],
                        z = faceKeypoints[{person, part, 2}];
                    partString = vectorToJson(x, y, z);
                }
                else {
                    partString = vectorToJson(0.0f, 0.0f, 0.0f);
                }
                faceKeypointsString += partString;
                if (part != faceKeypoints.getSize(1) - 1) faceKeypointsString += ",";
            }
            faceKeypointsString += "]";

            std::string personString = "{" + poseKeypointsString + "," + handKeypointsString_L + "," + handKeypointsString_R + "," + faceKeypointsString + "}";

            unitsString += personString;
            if (person != personCount - 1) unitsString += ",";
        }
        unitsString += "]";

        dataString = ("{" + unitsString + "}").c_str();
        outputToUnity(dataString, 0);
        // delete dataString;
    }

    void sendImage(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr) {
        const auto& cvOutput = datumsPtr->at(0).cvOutputData;

        std::string sizeString = "\"size\":{\"x\":" + std::to_string(cvOutput.cols) + ",\"y\":" + std::to_string(cvOutput.rows) + "}";

        // imageData = cvOutput.data;

        // std::string pixelsString = "\"pixels\":[";
        // for (int x = 0; x < cvOutput.cols; x++) {
        //     for (int y = 0; y < cvOutput.rows; y++) {
        //         int r = 127;// cvOutput.at<uchar>(x, y, 0);
        //         int g = 0;// cvOutput.at<uchar>(x, y, 1);
        //         int b = 0;// cvOutput.at<uchar>(x, y, 2);
        //         std::string vectorString = vectorToJson(r, g, b);
        //         pixelsString += vectorString;
        //         if (x != cvOutput.cols - 1 || y != cvOutput.rows - 1) {
        //             pixelsString += ",";
        //         }
        //     }
        // }
        // pixelsString += "]";

        // std::string imageString = "{" + sizeString + "," + pixelsString + "}";

        // outputToUnity(imageString, 1);
    }

    void workConsumer(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr)
    {
        try
        {
            if (datumsPtr != nullptr && !datumsPtr->empty())
            {
                // op::log("Output");
                if (gOutputEnabled) sendData(datumsPtr); // Tianyi's code
                // sendImage(datumsPtr); // Tianyi's code
                // outputToUnity(keypointsData, imageData, 0);

                // // Display image and sleeps at least 1 ms (it usually sleeps ~5-10 msec to display the image)
                // const char key = (char)cv::waitKey(1);
                // if (key == 27)
                //     this->stop();

                // Display rendered output image
                // cv::imshow("User worker GUI", datumsPtr->at(0).cvOutputData);
            }
        }
        catch (const std::exception& e)
        {
			// this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
};

// Global op instance
auto spUserOutput = std::make_shared<UnityPluginUserOutput>();
op::Wrapper spWrapper;

int OpenPose_Run()
{
    try
    {
        op::log("Starting OpenPose demo...", op::Priority::High);
        const auto timerBegin = std::chrono::high_resolution_clock::now();

        // logging_level
        op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
                  __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);
        // // For debugging
        // // Print all logging messages
        // op::ConfigureLog::setPriorityThreshold(op::Priority::None);
        // // Print out speed values faster
        // op::Profiler::setDefaultX(100);

        // Applying user defined configuration - Google flags to program variables
        // outputSize
        const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(FLAGS_face_net_resolution, "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(FLAGS_hand_net_resolution, "368x368 (multiples of 16)");
        // producerType
        const auto producerSharedPtr = op::flagsToProducer(FLAGS_image_dir, FLAGS_video, FLAGS_ip_camera, FLAGS_camera,
                                                           FLAGS_flir_camera, FLAGS_camera_resolution, FLAGS_camera_fps,
                                                           FLAGS_camera_parameter_folder, !FLAGS_frame_keep_distortion,
                                                           (unsigned int) FLAGS_3d_views, FLAGS_flir_camera_index);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::log("Flag `write_keypoint` is deprecated and will eventually be removed."
                    " Please, use `write_json` instead.", op::Priority::Max);
        // keypointScale
        const auto keypointScale = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScale = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1 || FLAGS_flir_camera);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;
        // Logging
        op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);

        // OpenPose wrapper
        op::log("Configuring OpenPose wrapper...", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        // op::Wrapper<std::vector<op::Datum>> opWrapper;
        //op::Wrapper spWrapper;

        // Initializing the user custom classes
        // GUI (Display)
        //spUserOutput = std::make_shared<UnityPluginUserOutput>();
        // Add custom processing
        const auto workerOutputOnNewThread = true;
        spWrapper.setWorker(op::WorkerType::Output, spUserOutput, workerOutputOnNewThread);

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            !FLAGS_body_disable, netInputSize, outputSize, keypointScale, FLAGS_num_gpu, FLAGS_num_gpu_start,
            FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
            poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
            FLAGS_part_to_show, FLAGS_model_folder, heatMapTypes, heatMapScale, FLAGS_part_candidates,
            (float)FLAGS_render_threshold, FLAGS_number_people_max, enableGoogleLogging};
		spWrapper.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
            FLAGS_face, faceNetInputSize, op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
            (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
		spWrapper.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range, FLAGS_hand_tracking,
            op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
            (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
		spWrapper.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
		spWrapper.configure(wrapperStructExtra);
        // Producer (use default to disable any input)
        const op::WrapperStructInput wrapperStructInput{
            producerSharedPtr, FLAGS_frame_first, FLAGS_frame_last, FLAGS_process_real_time, FLAGS_frame_flip,
            FLAGS_frame_rotate, FLAGS_frames_repeat};
		spWrapper.configure(wrapperStructInput);
        // Consumer (comment or use default argument to disable any output)
        // const op::WrapperStructOutput wrapperStructOutput{op::flagsToDisplayMode(FLAGS_display, FLAGS_3d),
        //                                                   !FLAGS_no_gui_verbose, FLAGS_fullscreen, FLAGS_write_keypoint,
        const auto displayMode = op::DisplayMode::NoDisplay;
        const bool guiVerbose = false;
        const bool fullScreen = false;
        const op::WrapperStructOutput wrapperStructOutput{
            displayMode, guiVerbose, fullScreen, FLAGS_write_keypoint,
            op::stringToDataFormat(FLAGS_write_keypoint_format), FLAGS_write_json, FLAGS_write_coco_json,
            FLAGS_write_coco_foot_json, FLAGS_write_images, FLAGS_write_images_format, FLAGS_write_video,
            FLAGS_camera_fps, FLAGS_write_heatmaps, FLAGS_write_heatmaps_format, FLAGS_write_video_adam,
            FLAGS_write_bvh, FLAGS_udp_host, FLAGS_udp_port};
		spWrapper.configure(wrapperStructOutput);
        // Set to single-thread running (to debug and/or reduce latency)
        if (FLAGS_disable_multi_thread)
            spWrapper.disableMultiThreading();

        // Start processing
        op::log("Starting thread(s)...", op::Priority::High);
        spWrapper.exec();

        // Measuring total time
        const auto now = std::chrono::high_resolution_clock::now();
        const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now-timerBegin).count()
                                * 1e-9;
        const auto message = "OpenPose demo successfully finished. Total time: "
                           + std::to_string(totalTimeSec) + " seconds.";
        op::log(message, op::Priority::High);

        // Return successful message
        return 0;
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return -1;
    }
}

// Tianyi's code
extern "C" {
	OP_API void OP_RegisterOutputCallback(OutputCallback callback) {
		if (spUserOutput) spUserOutput->gOutputCallback = callback;
	}

	OP_API void OP_SetOutputEnable(bool enable) {
		if (spUserOutput) spUserOutput->gOutputEnabled = enable;
	}
	
	OP_API void OP_ConfigurePose(
		bool body_disable = false, 
		int net_resolution_x = -1, int net_resolution_y = 368, // Point
		int output_resolution_x = -1, int output_resolution_y = -1, // Point
		int keypoint_scale = 0, // ScaleMode
		int num_gpu = -1, int num_gpu_start = 0, int scale_number = 1, float scale_gap = 0.3f,
		int render_pose = -1, bool _3d = false, int _3d_views = 1, bool flir_camera = false, // RenderMode
		std::string model_pose = "BODY_25", // PoseModel
		bool disable_blending = false, float alpha_pose = 0.6f, float alpha_heatmap = 0.7f,
		int part_to_show = 0, std::string model_folder = "models/", // moved
		bool heatmaps_add_parts = false, bool heatmaps_add_bkg = false, bool heatmaps_add_PAFs = false, // HeatMapType
		int heatmaps_scale = 2, // HeatMapScaleMode
		bool part_candidates = false, float render_threshold = 0.05f, int number_people_max = -1) {

		const op::WrapperStructPose wrapperStructPose{
			!body_disable,
			op::Point<int>{net_resolution_x, net_resolution_y}, // Point
			op::Point<int>{output_resolution_x, output_resolution_y}, // Point
			op::flagsToScaleMode(keypoint_scale), // ScaleMode
			num_gpu, num_gpu_start, scale_number, scale_gap,
			op::flagsToRenderMode(render_pose, (_3d || _3d_views > 1 || flir_camera)), // RenderMode
			op::flagsToPoseModel(model_pose), // PoseModel
			!disable_blending, alpha_pose, alpha_heatmap,
			part_to_show, model_folder, 
			op::flagsToHeatMaps(heatmaps_add_parts, heatmaps_add_bkg, heatmaps_add_PAFs), // HeatMapType
			op::flagsToHeatMapScaleMode(heatmaps_scale), 
			part_candidates, render_threshold, number_people_max, true };

		spWrapper.configure(wrapperStructPose);
	}
	OP_API void OP_ConfigureFace() {

	}
	OP_API void OP_ConfigureHand(
		bool hand = false, 
		int hand_net_resolution_x = 368, int hand_net_resolution_y = 368, // Point
		int hand_scale_number = 1, float hand_scale_range = 0.4f, bool hand_tracking = false,
		int hand_render = -1, bool _3d = false, int _3d_views = 1, bool flir_camera = false, int render_pose = -1, // RenderMode
		float hand_alpha_pose = 0.6f, float hand_alpha_heatmap = 0.7f, float hand_render_threshold = 0.2f) {

		const op::WrapperStructHand wrapperStructHand{
			hand, 
			op::Point<int>{hand_net_resolution_x, hand_net_resolution_y}, // Point
			hand_scale_number, hand_scale_range, hand_tracking,
			op::flagsToRenderMode(hand_render, (_3d || _3d_views > 1 || flir_camera), render_pose), 
			hand_alpha_pose, hand_alpha_heatmap, hand_render_threshold };

		spWrapper.configure(wrapperStructHand);
	}
	OP_API void OP_ConfigureExtra() {

	}
	OP_API void OP_ConfigureInput() {

	}

	OP_API void OP_SetParameters(int argc, char *argv[]) {

		// USE THESE TO FAKE THE INPUT ARGUMENTS
		for (int i = 1; i < argc; i++) {
			std::string s = argv[i];
			if (s.compare("--hand") == 0) {
				FLAGS_hand = true;
			}
			if (s.compare("--model_folder") == 0) {
				FLAGS_model_folder = argv[i + 1];
			}
			if (s.compare("--number_people_max") == 0) {
				FLAGS_number_people_max = 1;
			}
		}

		//gflags::ParseCommandLineFlags(&argc, &argv, true); // ---------------------------THIS ONE CRASH IN UNITY

	}

	OP_API void OP_Run() {
		OpenPose_Run();
	}

	OP_API void OP_Shutdown() {
		if (spUserOutput) spUserOutput->terminate();
	}

	OP_API void OP_LogTest() { op::log("test"); }
	OP_API void OP_OutputTest() { spUserOutput->outputTest(); }
}
#endif
