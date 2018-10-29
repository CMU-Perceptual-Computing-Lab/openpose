#ifdef USE_UNITY_SUPPORT
// ------------------------- OpenPose Unity Binding -------------------------

// OpenPose dependencies
#include <openpose/headers.hpp>

// Output callback register in Unity
typedef void(__stdcall * OutputCallback) (uchar ** ptrs, int ptrSize, int * sizes, int sizeSize, int outputType);

// Global output callback
OutputCallback unityOutputCallback;
bool unityOutputEnabled = true;

// This worker will just read and return all the jpg files in a directory
class UnityPluginUserOutput : public op::WorkerConsumer<std::shared_ptr<std::vector<op::Datum>>> {
public:	
    void initializationOnThread() {}

	// Output type enum
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

protected:
	template<class T>
	void outputValue(T ** ptrs, int ptrSize, int * sizes, int sizeSize, OutputType outputType) {
		if (!unityOutputCallback) return;
		uchar ** bytePtrs = static_cast<uchar**>(static_cast<void*>(ptrs));
		unityOutputCallback(bytePtrs, ptrSize, sizes, sizeSize, (int)outputType);
	}

	void workConsumer(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr) {
		try	{
			if (datumsPtr != nullptr && !datumsPtr->empty()) {				
				if (unityOutputEnabled) {
					sendPoseKeypoints(datumsPtr);
					sendHandKeypoints(datumsPtr);
					sendFaceKeypoints(datumsPtr);
					sendFaceRectangles(datumsPtr);
				}
			}
		} catch (const std::exception& e) {
			op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

	void sendPoseKeypoints(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr) {
		auto& data = datumsPtr->at(0).poseKeypoints; // Array<float>
		if (!data.empty()) {
			auto sizeVector = data.getSize();
			int sizeSize = sizeVector.size();
			int * sizes = &sizeVector[0];
			float * val = data.getPtr();
			outputValue(&val, 1, sizes, sizeSize, PoseKeypoints);
		}
	}

	void sendHandKeypoints(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr) {
		auto& data = datumsPtr->at(0).handKeypoints; // std::array<Array<float>, 2>
		if (data.size() == 2 && !data[0].empty()) {
			auto sizeVector = data[0].getSize();
			int sizeSize = sizeVector.size();
			int * sizes = &sizeVector[0];
			//float ptrs[] = { data[0].getPtr(), data[1].getPtr() };
			auto ptrs = new float*[2];
			ptrs[0] = data[0].getPtr();
			ptrs[1] = data[1].getPtr();	
			outputValue(ptrs, 2, sizes, sizeSize, HandKeypoints);
			delete ptrs;
		}
	}

	void sendFaceKeypoints(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr) {
		auto& data = datumsPtr->at(0).faceKeypoints; // Array<float>
		if (!data.empty()) {
			auto sizeVector = data.getSize();
			int sizeSize = sizeVector.size();
			int * sizes = &sizeVector[0];
			float * val = data.getPtr();
			outputValue(&val, 1, sizes, sizeSize, FaceKeypoints);
		}
	}

	void sendFaceRectangles(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr) {
		auto& data = datumsPtr->at(0).faceRectangles; // std::vector<Rectangle<float>>
		if (data.size() > 0) {
			int sizes[] = { data.size(), 4 };
			std::vector<float> vals(data.size() * 4);
			for (int i = 0; i < data.size(); i++) {
				vals[4 * i + 0] = data[i].x;
				vals[4 * i + 1] = data[i].y;
				vals[4 * i + 2] = data[i].width;
				vals[4 * i + 3] = data[i].height;
			}
			float * val = &vals[0];
			outputValue(&val, 1, sizes, 2, FaceRectangles);
		}
	}
};

// Global user output
UnityPluginUserOutput* ptrUserOutput = nullptr;

// Global setting structs
std::shared_ptr<op::WrapperStructPose> spWrapperStructPose;
std::shared_ptr<op::WrapperStructHand> spWrapperStructHand;
std::shared_ptr<op::WrapperStructFace> spWrapperStructFace;
std::shared_ptr<op::WrapperStructExtra> spWrapperStructExtra;
std::shared_ptr<op::WrapperStructInput> spWrapperStructInput;
std::shared_ptr<op::WrapperStructOutput> spWrapperStructOutput;

// Main
void openpose_main() {
	try {
		// Starting
		if (ptrUserOutput) return;
		op::log("Starting OpenPose demo...", op::Priority::High);
		const auto timerBegin = std::chrono::high_resolution_clock::now();

		// OpenPose wrapper
		auto spWrapper = std::make_shared<op::Wrapper>();

		// Initializing the user custom classes
		auto spUserOutput = std::make_shared<UnityPluginUserOutput>();
		ptrUserOutput = spUserOutput.get();
		
		// Add custom processing
		const auto workerOutputOnNewThread = true;
		spWrapper->setWorker(op::WorkerType::Output, spUserOutput, workerOutputOnNewThread);

		// Apply configurations
		spWrapper->configure(*spWrapperStructPose);
		spWrapper->configure(*spWrapperStructHand);
		spWrapper->configure(*spWrapperStructFace);
		spWrapper->configure(*spWrapperStructExtra);
		spWrapper->configure(*spWrapperStructInput);
		spWrapper->configure(*spWrapperStructOutput);

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

		// Reset pointer
		ptrUserOutput = nullptr;
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}

// Functions called from Unity
extern "C" {
	// Start openpose safely
	OP_API void OP_Run() {
		if (ptrUserOutput == nullptr) 
			openpose_main();
	}
	// Stop openpose safely
	OP_API void OP_Shutdown() {
		if (ptrUserOutput != nullptr) {
			op::log("Stopping...");
			ptrUserOutput->stop();
		}
	}
	// Register Unity output callback function
	OP_API void OP_RegisterOutputCallback(OutputCallback callback) {
		unityOutputCallback = callback;
	}
	// Enable/disable output callback
	OP_API void OP_SetOutputEnable(bool enable) {
		unityOutputEnabled = enable;
	}
	// Configs
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
			spWrapperStructPose = std::make_shared<op::WrapperStructPose>(
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
				part_candidates, render_threshold, number_people_max, true);

			//spWrapper->configure(spWrapperStructPose);
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
			spWrapperStructFace = std::make_shared<op::WrapperStructFace>(
				face, op::Point<int>{face_net_resolution_x, face_net_resolution_y},
				op::flagsToRenderMode(face_renderer, false/*(_3d || _3d_views > 1 || flir_camera)*/, render_pose), // RenderMode
				face_alpha_pose, face_alpha_heatmap, face_render_threshold);
			//spWrapper->configure(spWrapperStructFace);
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
			spWrapperStructHand = std::make_shared<op::WrapperStructHand>(
				hand,
				op::Point<int>{hand_net_resolution_x, hand_net_resolution_y}, // Point
				hand_scale_number, hand_scale_range, hand_tracking,
				op::flagsToRenderMode(hand_render, (_3d || _3d_views > 1 || flir_camera), render_pose),
				hand_alpha_pose, hand_alpha_heatmap, hand_render_threshold);

			//spWrapper->configure(spWrapperStructHand);
		}
		catch (const std::exception& e)
		{
			op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}
	OP_API void OP_ConfigureExtra(
		bool _3d = false, int _3d_min_views = -1, bool _identification = false, int _tracking = -1, int _ik_threads = 0) {
		try {
			spWrapperStructExtra = std::make_shared<op::WrapperStructExtra>(
				_3d, _3d_min_views, _identification, _tracking, _ik_threads);
			//spWrapper->configure(spWrapperStructExtra);
		}
		catch (const std::exception& e)
		{
			op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}
	OP_API void OP_ConfigureInput(
		uchar producerType = 5, char* producerString = "",
		unsigned long long frame_first = 0, 
		unsigned long long frame_step = 1, 
		unsigned long long  frame_last = std::numeric_limits<unsigned long long>::max(),
		bool process_real_time = false, bool frame_flip = false,
		int frame_rotate = 0, bool frames_repeat = false, 
		int camera_resolution_x = -1, int camera_resolution_y = -1, double webcam_fps = 30., 
		char* camera_parameter_path = "models/cameraParameters/", 
		bool undistort_image = true, uint image_directory_stereo = 1) {
		try {
			spWrapperStructInput = std::make_shared<op::WrapperStructInput>(
				(op::ProducerType) producerType, producerString,
				frame_first, frame_step, frame_last, 
				process_real_time, frame_flip,
				frame_rotate, frames_repeat,
				op::Point<int>{ camera_resolution_x, camera_resolution_y }, webcam_fps,
				camera_parameter_path,
				undistort_image, image_directory_stereo);
			//spWrapper->configure(spWrapperStructInput);
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
			spWrapperStructOutput = std::make_shared<op::WrapperStructOutput>(
				displayMode, guiVerbose, fullScreen, write_keypoint,
				op::stringToDataFormat(write_keypoint_format), write_json, write_coco_json,
				write_coco_foot_json, write_images, write_images_format, write_video,
				camera_fps,
				write_heatmaps, write_heatmaps_format, write_video_adam,
				write_bvh, udp_host, udp_port);
			//spWrapper->configure(spWrapperStructOutput);
		}
		catch (const std::exception& e)
		{
			op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}
}

#endif