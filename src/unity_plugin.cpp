#ifdef USE_UNITY_SUPPORT
// ------------------------- OpenPose Unity Binding -------------------------

// OpenPose dependencies
#include <openpose/headers.hpp>

// Output callback register in Unity
typedef void(__stdcall * OutputCallback) (uchar** val, int* sizes, int sizeSize, int valType, int outputType);

// Output type enum
enum ValType : int{
	Byte = 0,
	Int = 1,
	Long = 2,
	Float = 3, 
	String = 4
};

enum OutputType : int {
	None = 0, 
	Ids = 1, 
	Name = 2, 
	FrameNumber = 3, 
	PoseKeypoints = 4, 
	PoseIds = 5, 
	PoseScores = 6, 
	PoseHeatMaps = 7,
	PoseCandidates = 8, 
	FaceRectangles = 9, 
	FaceKeypoints = 10, 
	FaceHeatMaps = 11, 
	HandRectangles = 12, 
	HandKeypoints = 13, 
	HandHeightMaps = 14, 
	PoseKeypoints3D = 15, 
	FaceKeypoints3D = 16, 
	HandKeypoints3D = 17, 
	CameraMatrix = 18, 
	CameraExtrinsics = 19, 
	CameraIntrinsics = 20
};

// Global setting structs
op::WrapperStructPose *wrapperStructPose;
op::WrapperStructHand *wrapperStructHand;
op::WrapperStructFace *wrapperStructFace;
op::WrapperStructExtra *wrapperStructExtra;
op::WrapperStructInput *wrapperStructInput;
op::WrapperStructOutput *wrapperStructOutput;

// Global flags
bool opStoppingFlag = false;
bool opRunningFlag = false;

// This worker will just read and return all the jpg files in a directory
class UnityPluginUserOutput : public op::WorkerConsumer<std::shared_ptr<std::vector<op::Datum>>> {
public:
	OutputCallback unityOutputCallback;
	bool unityOutputEnabled = true;

    void initializationOnThread() {}

    //void terminate() {
	//	op::log("OP_End");
    //    this->stop();
    //}

protected:
	template<class T>
	void outputValue(T * val, int * sizes, int sizeSize, ValType valType, OutputType outputType) {
		uchar * bytes = static_cast<uchar*>(static_cast<void*>(val));
		unityOutputCallback(&bytes, sizes, sizeSize, (int)valType, (int)outputType);
	}

	void workConsumer(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr) {
		try	{
			if (datumsPtr != nullptr && !datumsPtr->empty()) {
				if (opStoppingFlag) {
					op::log("Stopping");
					this->stop();
					return;
				}
				
				if (unityOutputEnabled) {
					sendPoseKeypoints(datumsPtr);
					sendHandKeypoints(datumsPtr);
				}
			}
		} catch (const std::exception& e) {
			// this->stop();
			op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

	void sendPoseKeypoints(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr) {
		const auto& data = datumsPtr->at(0).poseKeypoints;
		if (!data.empty()) {
			auto sizeVector = data.getSize();
			int sizeSize = sizeVector.size();
			int * sizes = &sizeVector[0];
			int volume = data.getVolume();
			float * val = new float[volume];
			for (int i = 0; i < volume; i++) {
				val[i] = data[i]; // Is there a better way to do this?
			}
			outputValue(val, sizes, sizeSize, Float, PoseKeypoints);
			delete val;
		}
	}

	void sendHandKeypoints(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr) {
		const auto& data = datumsPtr->at(0).handKeypoints;
		if (!data.empty()) {
			int sizeSize = 4;
			int * sizes = new int[sizeSize] {2, data[0].getSize(0), data[0].getSize(1), data[0].getSize(2)};
			int volume = sizes[0] * sizes[1] * sizes[2] * sizes[3];
			float * val = new float[volume];
			for (int i = 0; i < sizes[0]; i++) {
				for (int j = 0; j < volume / sizes[0]; j++) {
					val[i * j] = data[i][j]; // Is there a better way to do this?
				}
			}			
			outputValue(val, sizes, sizeSize, Float, HandKeypoints);
			delete val;
		}
	}

    /*void sendData(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr) {
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
        //outputToUnity(dataString, 0);
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
    }
	
	std::string vectorToJson(const float x, const float y, const float z) {
		return "{\"x\":" + std::to_string(x) + ",\"y\":" + std::to_string(y) + ",\"z\":" + std::to_string(z) + "}";
	}

	std::string vectorToJson(const int x, const int y, const int z) {
		return "{\"x\":" + std::to_string(x) + ",\"y\":" + std::to_string(y) + ",\"z\":" + std::to_string(z) + "}";
	}*/
};

// Global user output
//std::shared_ptr<UnityPluginUserOutput> spUserOutput;

// Title
void openpose_main(bool enableOutput, OutputCallback callback);

// Unity Interface
extern "C" {
	/*OP_API void OPT_RegisterTest(OutputIntTest intTest, OutputFloatTest floatTest, OutputByte byteTest) {
		unityTestIntArray = intTest;
		unityTestFloatArray = floatTest;
		unityTestBytes = byteTest;
	}

	OP_API void OPT_CallbackTestFunctions() {
		int* intArr = new int[9]{ 9,8,7,6,5,4,3,2,1 };
		float* floatArr = new float[9]{ 9.f,8.f,7.f,6.f,5.f,4.f,3.f,2.f,1.f };
		int* size = new int(9);
		//unityTestIntArray(&intArr, size);	
		//unityTestFloatArray(&floatArr, size);
		uchar * bytes = static_cast<uchar*>(static_cast<void*>(intArr));
		unityTestBytes(&bytes, size, 0);
		//bytes = (uchar*)floatArr;
		//unityTestBytes(bytes, size, 1);
	}*/

	OP_API void OP_ConfigurePose(
		bool body_disable = false,
		char* model_folder = "models/", int number_people_max = -1, // moved
		int net_resolution_x = -1, int net_resolution_y = 368, // Point
		int output_resolution_x = -1, int output_resolution_y = -1, // Point
		int keypoint_scale = 0, // ScaleMode
		int num_gpu = -1, int num_gpu_start = 0, int scale_number = 1, float scale_gap = 0.3f,
		int render_pose = -1, // bool _3d = false, int _3d_views = 1, bool flir_camera = false, // RenderMode
		char* model_pose = "BODY_25", // PoseModel
		bool disable_blending = false, float alpha_pose = 0.6f, float alpha_heatmap = 0.7f,
		int part_to_show = 0,
		bool heatmaps_add_parts = false, bool heatmaps_add_bkg = false, bool heatmaps_add_PAFs = false, // HeatMapType
		int heatmaps_scale = 2, // HeatMapScaleMode
		bool part_candidates = false, float render_threshold = 0.05f) {
		try {
			wrapperStructPose = new op::WrapperStructPose{
				!body_disable,
				op::Point<int>{net_resolution_x, net_resolution_y}, // Point
				op::Point<int>{output_resolution_x, output_resolution_y}, // Point
				op::flagsToScaleMode(keypoint_scale), // ScaleMode
				num_gpu, num_gpu_start, scale_number, scale_gap,
				op::flagsToRenderMode(render_pose, false/*(_3d || _3d_views > 1 || flir_camera)*/), // RenderMode
				op::flagsToPoseModel(model_pose), // PoseModel
				!disable_blending, alpha_pose, alpha_heatmap,
				part_to_show, model_folder,
				op::flagsToHeatMaps(heatmaps_add_parts, heatmaps_add_bkg, heatmaps_add_PAFs), // HeatMapType
				op::flagsToHeatMapScaleMode(heatmaps_scale),
				part_candidates, render_threshold, number_people_max, true };

			//spWrapper->configure(wrapperStructPose);
		}
		catch (const std::exception& e)
		{
			op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}
	OP_API void OP_ConfigureFace(
		bool face = false, int face_net_resolution_x = 368, int face_net_resolution_y = 368,
		int face_renderer = -1, int render_pose = -1,
		float face_alpha_pose = 0.6f, float face_alpha_heatmap = 0.7f, float face_render_threshold = 0.4f) {
		try {
			wrapperStructFace = new op::WrapperStructFace{
				face, op::Point<int>{face_net_resolution_x, face_net_resolution_y},
				op::flagsToRenderMode(face_renderer, false/*(_3d || _3d_views > 1 || flir_camera)*/, render_pose), // RenderMode
				face_alpha_pose, face_alpha_heatmap, face_render_threshold };
			//spWrapper->configure(wrapperStructFace);
		}
		catch (const std::exception& e)
		{
			op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}
	OP_API void OP_ConfigureHand(
		bool hand = false,
		int hand_net_resolution_x = 368, int hand_net_resolution_y = 368, // Point
		int hand_scale_number = 1, float hand_scale_range = 0.4f, bool hand_tracking = false,
		int hand_render = -1, bool _3d = false, int _3d_views = 1, bool flir_camera = false, int render_pose = -1, // RenderMode
		float hand_alpha_pose = 0.6f, float hand_alpha_heatmap = 0.7f, float hand_render_threshold = 0.2f) {
		try {
			wrapperStructHand = new op::WrapperStructHand{
				hand,
				op::Point<int>{hand_net_resolution_x, hand_net_resolution_y}, // Point
				hand_scale_number, hand_scale_range, hand_tracking,
				op::flagsToRenderMode(hand_render, (_3d || _3d_views > 1 || flir_camera), render_pose),
				hand_alpha_pose, hand_alpha_heatmap, hand_render_threshold };

			//spWrapper->configure(wrapperStructHand);
		}
		catch (const std::exception& e)
		{
			op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}
	OP_API void OP_ConfigureExtra(
		bool _3d = false, int _3d_min_views = -1, bool _identification = false, int _tracking = -1, int _ik_threads = 0) {
		try {
			wrapperStructExtra = new op::WrapperStructExtra{
				_3d, _3d_min_views, _identification, _tracking, _ik_threads };
			//spWrapper->configure(wrapperStructExtra);
		}
		catch (const std::exception& e)
		{
			op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}
	OP_API void OP_ConfigureInput(
		int frame_first = 0, int frame_last = -1, // unsigned long long (uint64)
		bool process_real_time = false, bool frame_flip = false,
		int frame_rotate = 0, bool frames_repeat = false,
		// Producer
		char* image_dir = "", char* video = "", char* ip_camera = "", int camera = -1,
		bool flir_camera = false, int camera_resolution_x = -1, int camera_resolution_y = -1, double camera_fps = 30.0,
		char* camera_parameter_folder = "models / cameraParameters / flir / ", bool frame_keep_distortion = false,
		int _3d_views = 1, int flir_camera_index = -1) {
		try {
			auto producerSharedPtr = op::flagsToProducer(
				image_dir, video, ip_camera, camera,
				flir_camera, std::to_string(camera_resolution_x) + "x" + std::to_string(camera_resolution_y), camera_fps,
				camera_parameter_folder, !frame_keep_distortion,
				(unsigned int)_3d_views, flir_camera_index);

			wrapperStructInput = new op::WrapperStructInput{
				producerSharedPtr, (unsigned long long) frame_first, (unsigned long long) frame_last, process_real_time, frame_flip,
				frame_rotate, frames_repeat };
			//spWrapper->configure(wrapperStructInput);
		}
		catch (const std::exception& e)
		{
			op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}
	OP_API void OP_ConfigureOutput(
		char* write_keypoint = "",
		char* write_keypoint_format = "yml", char* write_json = "", char* write_coco_json = "",
		char* write_coco_foot_json = "", char* write_images = "", char* write_images_format = "png", char* write_video = "",
		double camera_fps = 30.,
		char* write_heatmaps = "", char* write_heatmaps_format = "png", char* write_video_adam = "",
		char* write_bvh = "", char* udp_host = "", char* udp_port = "8051") {
		try {
			const auto displayMode = op::DisplayMode::NoDisplay;
			const bool guiVerbose = false;
			const bool fullScreen = false;
			wrapperStructOutput = new op::WrapperStructOutput{
				displayMode, guiVerbose, fullScreen, write_keypoint,
				op::stringToDataFormat(write_keypoint_format), write_json, write_coco_json,
				write_coco_foot_json, write_images, write_images_format, write_video,
				camera_fps,
				write_heatmaps, write_heatmaps_format, write_video_adam,
				write_bvh, udp_host, udp_port };
			//spWrapper->configure(wrapperStructOutput);
		}
		catch (const std::exception& e)
		{
			op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

	OP_API void OP_Run(bool enableOutput, OutputCallback callback) {
		openpose_main(enableOutput, callback);
	}

	OP_API void OP_Shutdown() {
		if (opRunningFlag) opStoppingFlag = true;
	}

	//OP_API void OP_RegisterOutputCallback(OutputCallback callback) {
	//	//if (spUserOutput) spUserOutput->gOutputCallback = callback;
	//}
	//OP_API void OP_SetOutputEnable(bool enable) {
	//	//if (spUserOutput) spUserOutput->gOutputEnabled = enable;
	//}
	//OP_API void OP_SetParameters(int argc, char *argv[]) {
	//	//gflags::ParseCommandLineFlags(&argc, &argv, true);
	//}
	//OP_API void OP_LogTest() { op::log("test"); }
	//OP_API void OP_OutputTest() { spUserOutput->outputTest(); }
}

// Main
void openpose_main(bool enableOutput, OutputCallback callback) {
	try {
		// Starting
		if (opRunningFlag) return;
		opRunningFlag = true;
		op::log("Starting OpenPose demo...", op::Priority::High);
		const auto timerBegin = std::chrono::high_resolution_clock::now();

		// OpenPose wrapper
		auto spWrapper = std::make_shared<op::Wrapper>();

		// Initializing the user custom classes
		auto spUserOutput = std::make_shared<UnityPluginUserOutput>();

		// Register Unity output callback
		spUserOutput->unityOutputEnabled = enableOutput;
		if (callback) spUserOutput->unityOutputCallback = callback;

		// Add custom processing
		const auto workerOutputOnNewThread = true;
		spWrapper->setWorker(op::WorkerType::Output, spUserOutput, workerOutputOnNewThread);

		// TODO: fix Producer ptr problem to enable Unity users to set this in Unity (then delete this code)
		OP_ConfigureInput();

		// Apply configurations
		spWrapper->configure(*wrapperStructPose);
		spWrapper->configure(*wrapperStructHand);
		spWrapper->configure(*wrapperStructFace);
		spWrapper->configure(*wrapperStructExtra);
		spWrapper->configure(*wrapperStructInput);
		spWrapper->configure(*wrapperStructOutput);

		// Start processing
		op::log("Starting thread(s)...", op::Priority::High);
		spWrapper->exec();

		// Running ...... Ending

		// Measuring total time
		const auto now = std::chrono::high_resolution_clock::now();
		const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now - timerBegin).count()
			* 1e-9;
		const auto message = "OpenPose demo successfully finished. Total time: "
			+ std::to_string(totalTimeSec) + " seconds.";
		op::log(message, op::Priority::High);

		// Clean up configs
		delete wrapperStructPose;
		delete wrapperStructHand;
		delete wrapperStructFace;
		delete wrapperStructExtra;
		delete wrapperStructInput;
		delete wrapperStructOutput;

		// Reset flags
		opStoppingFlag = false;
		opRunningFlag = false;

	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}
#endif