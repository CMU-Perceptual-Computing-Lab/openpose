#ifndef OPENPOSE_PYTHON_HPP
#define OPENPOSE_PYTHON_HPP
#define BOOST_DATE_TIME_NO_LIB

// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/hand/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include <caffe/caffe.hpp>
#include <stdlib.h>

#include <openpose/net/nmsCaffe.hpp>
#include <openpose/net/resizeAndMergeCaffe.hpp>
#include <openpose/pose/bodyPartConnectorCaffe.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/pose/poseExtractor.hpp>
#include <openpose/gpu/cuda.hpp>
#include <openpose/gpu/opencl.hcl>
#include <openpose/core/macros.hpp>

#ifdef _WIN32
    #define OP_EXPORT __declspec(dllexport)
#else
    #define OP_EXPORT
#endif

#define default_logging_level 3
#define default_output_resolution "-1x-1"
#define default_net_resolution "-1x368"
#define default_model_pose "COCO"
#define default_alpha_pose 0.6
#define default_scale_gap 0.3
#define default_scale_number 1
#define default_render_threshold 0.05
#define default_num_gpu_start 0
#define default_disable_blending false
#define default_model_folder "models/"

#define default_hand_net_resolution "368x368"

// Todo, have GPU Number, handle, OpenCL/CPU Cases
OP_API class OpenPose {
public:
	std::unique_ptr<op::PoseExtractorCaffe> poseExtractorCaffe;
	std::unique_ptr<op::HandExtractorCaffe> handExtractorCaffe;
	std::unique_ptr<op::HandDetector> handDetector;
	std::unique_ptr<op::HandCpuRenderer> handRenderer;
	std::unique_ptr<op::PoseCpuRenderer> poseRenderer;
	std::unique_ptr<op::FrameDisplayer> frameDisplayer;
	std::unique_ptr<op::ScaleAndSizeExtractor> scaleAndSizeExtractor;

	std::unique_ptr<op::ResizeAndMergeCaffe<float>> resizeAndMergeCaffe;
	std::unique_ptr<op::NmsCaffe<float>> nmsCaffe;
	std::unique_ptr<op::BodyPartConnectorCaffe<float>> bodyPartConnectorCaffe;
	std::shared_ptr<caffe::Blob<float>> heatMapsBlob;
	std::shared_ptr<caffe::Blob<float>> peaksBlob;
	op::Array<float> mPoseKeypoints;
	op::Array<float> mPoseScores;
	op::PoseModel poseModel;
	int mGpuID;

	OpenPose(int FLAGS_logging_level = default_logging_level,
		std::string FLAGS_output_resolution = default_output_resolution,
		std::string FLAGS_net_resolution = default_net_resolution,
		std::string FLAGS_model_pose = default_model_pose,
		float FLAGS_alpha_pose = default_alpha_pose,
		float FLAGS_scale_gap = default_scale_gap,
		int FLAGS_scale_number = default_scale_number,
		float FLAGS_render_threshold = default_render_threshold,
		int FLAGS_num_gpu_start = default_num_gpu_start,
		int FLAGS_disable_blending = default_disable_blending,
		std::string FLAGS_model_folder = default_model_folder,
		std::string FLAGS_hand_net_resolution = default_hand_net_resolution
	) {
		mGpuID = FLAGS_num_gpu_start;
#ifdef USE_CUDA
		caffe::Caffe::set_mode(caffe::Caffe::GPU);
		caffe::Caffe::SetDevice(mGpuID);
#elif USE_OPENCL
		caffe::Caffe::set_mode(caffe::Caffe::GPU);
		std::vector<int> devices;
		const int maxNumberGpu = op::OpenCL::getTotalGPU();
		for (auto i = 0; i < maxNumberGpu; i++)
			devices.emplace_back(i);
		caffe::Caffe::SetDevices(devices);
		caffe::Caffe::SelectDevice(mGpuID, true);
		op::OpenCL::getInstance(mGpuID, CL_DEVICE_TYPE_GPU, true);
#else
		caffe::Caffe::set_mode(caffe::Caffe::CPU);
#endif
		op::log("OpenPose Library Python Wrapper", op::Priority::High);
		// ------------------------- INITIALIZATION -------------------------
		// Step 1 - Set logging level
		// - 0 will output all the logging messages
		// - 255 will output nothing
		op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
		op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
		// Step 2 - Read Google flags (user defined configuration)
		// outputSize
		const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
		// netInputSize
		const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
		const auto handNetInputSize = op::flagsToPoint(FLAGS_hand_net_resolution, "368x368");
		// poseModel
		poseModel = op::flagsToPoseModel(FLAGS_model_pose);
		// Check no contradictory flags enabled
		if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
			op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
		if (FLAGS_scale_gap <= 0. && FLAGS_scale_number > 1)
			op::error("Incompatible flag configuration: scale_gap must be greater than 0 or scale_number = 1.",
				__LINE__, __FUNCTION__, __FILE__);
		// Logging
		op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
		// Step 3 - Initialize all required classes
		scaleAndSizeExtractor = std::unique_ptr<op::ScaleAndSizeExtractor>(new op::ScaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap));

		poseExtractorCaffe = std::unique_ptr<op::PoseExtractorCaffe>(new op::PoseExtractorCaffe{ poseModel, FLAGS_model_folder, FLAGS_num_gpu_start });

		handDetector = std::unique_ptr<op::HandDetector>(new op::HandDetector{ poseModel });
		handExtractorCaffe = std::unique_ptr<op::HandExtractorCaffe>(new op::HandExtractorCaffe{ handNetInputSize, handNetInputSize, FLAGS_model_folder, FLAGS_num_gpu_start });

		poseRenderer = std::unique_ptr<op::PoseCpuRenderer>(new op::PoseCpuRenderer{ poseModel, (float)FLAGS_render_threshold, !FLAGS_disable_blending,
			(float)FLAGS_alpha_pose });
		handRenderer = std::unique_ptr<op::HandCpuRenderer>(new op::HandCpuRenderer{ (float)FLAGS_render_threshold });
		frameDisplayer = std::unique_ptr<op::FrameDisplayer>(new op::FrameDisplayer{ "OpenPose Tutorial - Example 1", outputSize });

		// Custom
		resizeAndMergeCaffe = std::unique_ptr<op::ResizeAndMergeCaffe<float>>(new op::ResizeAndMergeCaffe<float>{});
		nmsCaffe = std::unique_ptr<op::NmsCaffe<float>>(new op::NmsCaffe<float>{});
		bodyPartConnectorCaffe = std::unique_ptr<op::BodyPartConnectorCaffe<float>>(new op::BodyPartConnectorCaffe<float>{});
		heatMapsBlob = { std::make_shared<caffe::Blob<float>>(1,1,1,1) };
		peaksBlob = { std::make_shared<caffe::Blob<float>>(1,1,1,1) };
		bodyPartConnectorCaffe->setPoseModel(poseModel);

		// Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
		poseExtractorCaffe->initializationOnThread();

		handExtractorCaffe->initializationOnThread();

		poseRenderer->initializationOnThread();
		handRenderer->initializationOnThread();

	}

	std::vector<caffe::Blob<float>*> caffeNetSharedToPtr(
		std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob)
	{
		try
		{
			// Prepare spCaffeNetOutputBlobss
			std::vector<caffe::Blob<float>*> caffeNetOutputBlobs(caffeNetOutputBlob.size());
			for (auto i = 0u; i < caffeNetOutputBlobs.size(); i++)
				caffeNetOutputBlobs[i] = caffeNetOutputBlob[i].get();
			return caffeNetOutputBlobs;
		}
		catch (const std::exception& e)
		{
			op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
			return{};
		}
	}

	void forward(const cv::Mat& inputImage, op::Array<float>& poseKeypoints, std::array<op::Array<float>, 2>& handKeypoints, cv::Mat& displayImage, bool display = false, bool hands = false) {
		op::OpOutputToCvMat opOutputToCvMat;
		op::CvMatToOpInput cvMatToOpInput;
		op::CvMatToOpOutput cvMatToOpOutput;
		if (inputImage.empty())
			op::error("Could not open or find the image: ", __LINE__, __FUNCTION__, __FILE__);
		const op::Point<int> imageSize{ inputImage.cols, inputImage.rows };
		// Step 2 - Get desired scale sizes
		std::vector<double> scaleInputToNetInputs;
		std::vector<op::Point<int>> netInputSizes;
		double scaleInputToOutput;
		op::Point<int> outputResolution;
		std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
			= scaleAndSizeExtractor->extract(imageSize);
		// Step 3 - Format input image to OpenPose input and output formats
		const auto netInputArray = cvMatToOpInput.createArray(inputImage, scaleInputToNetInputs, netInputSizes);

		// Step 4 - Estimate poseKeypoints
		poseExtractorCaffe->forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
		poseKeypoints = poseExtractorCaffe->getPoseKeypoints();

		// Step 4.5 - Estimate handKeypoints
		if (hands) {
			const auto numberPeople = poseKeypoints.getSize(0);
			std::vector<std::array<op::Rectangle<float>, 2>> handRectangles(numberPeople);

			handRectangles = handDetector->detectHands(poseKeypoints);
			handExtractorCaffe->forwardPass(handRectangles, inputImage);
			handKeypoints = handExtractorCaffe->getHandKeypoints();
		}

		if (display) {
			auto outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
			// Step 5 - Render poseKeypoints
			poseRenderer->renderPose(outputArray, poseKeypoints, scaleInputToOutput);
			if (hands) {
				handRenderer->renderHand(outputArray, handKeypoints, scaleInputToOutput);
			}
			// Step 6 - OpenPose output format to cv::Mat
			displayImage = opOutputToCvMat.formatToCvMat(outputArray);
		}
	}

	void poseFromHeatmap(const cv::Mat& inputImage, std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob, op::Array<float>& poseKeypoints, cv::Mat& displayImage, std::vector<op::Point<int>>& imageSizes) {
		// Get Scale
		const op::Point<int> inputDataSize{ inputImage.cols, inputImage.rows };

		// Convert to Ptr
		//std::vector<boost::shared_ptr<caffe::Blob<float>>> a;
		//caffeNetOutputBlob.emplace_back(caffeHmPtr);
		const auto caffeNetOutputBlobs = caffeNetSharedToPtr(caffeNetOutputBlob);

		// To be called once only
		resizeAndMergeCaffe->Reshape(caffeNetOutputBlobs, { heatMapsBlob.get() },
			op::getPoseNetDecreaseFactor(poseModel), 1.f / 1.f, true,
			0);
		nmsCaffe->Reshape({ heatMapsBlob.get() }, { peaksBlob.get() }, op::getPoseMaxPeaks(poseModel),
			op::getPoseNumberBodyParts(poseModel), 0);
		bodyPartConnectorCaffe->Reshape({ heatMapsBlob.get(), peaksBlob.get() });

		// Normal
		op::OpOutputToCvMat opOutputToCvMat;
		op::CvMatToOpInput cvMatToOpInput;
		op::CvMatToOpOutput cvMatToOpOutput;
		if (inputImage.empty())
			op::error("Could not open or find the image: ", __LINE__, __FUNCTION__, __FILE__);
		const op::Point<int> imageSize{ inputImage.cols, inputImage.rows };
		// Step 2 - Get desired scale sizes
		std::vector<double> scaleInputToNetInputs;
		std::vector<op::Point<int>> netInputSizes;
		double scaleInputToOutput;
		op::Point<int> outputResolution;

		std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
			= scaleAndSizeExtractor->extract(imageSize);

		const auto netInputArray = cvMatToOpInput.createArray(inputImage, scaleInputToNetInputs, netInputSizes);

		// Run the modes
		const std::vector<float> floatScaleRatios(scaleInputToNetInputs.begin(), scaleInputToNetInputs.end());
		resizeAndMergeCaffe->setScaleRatios(floatScaleRatios);
		std::vector<caffe::Blob<float>*> heatMapsBlobs{ heatMapsBlob.get() };
		std::vector<caffe::Blob<float>*> peaksBlobs{ peaksBlob.get() };
#ifdef USE_CUDA
		resizeAndMergeCaffe->Forward_gpu(caffeNetOutputBlobs, heatMapsBlobs); // ~5ms
#elif USE_OPENCL
		resizeAndMergeCaffe->Forward_ocl(caffeNetOutputBlobs, heatMapsBlobs); // ~5ms
#else
		resizeAndMergeCaffe->Forward_cpu(caffeNetOutputBlobs, heatMapsBlobs); // ~5ms
#endif

		nmsCaffe->setThreshold((float)poseExtractorCaffe->get(op::PoseProperty::NMSThreshold));
#ifdef USE_CUDA
		nmsCaffe->Forward_gpu(heatMapsBlobs, peaksBlobs);// ~2ms
#elif USE_OPENCL
		nmsCaffe->Forward_ocl(heatMapsBlobs, peaksBlobs);// ~2ms
#else
		nmsCaffe->Forward_cpu(heatMapsBlobs, peaksBlobs);// ~2ms
#endif
		op::cudaCheck(__LINE__, __FUNCTION__, __FILE__);

		float mScaleNetToOutput = 1. / scaleInputToNetInputs[0];
		bodyPartConnectorCaffe->setScaleNetToOutput(mScaleNetToOutput);
		bodyPartConnectorCaffe->setInterMinAboveThreshold(
			(float)poseExtractorCaffe->get(op::PoseProperty::ConnectInterMinAboveThreshold)
		);
		bodyPartConnectorCaffe->setInterThreshold((float)poseExtractorCaffe->get(op::PoseProperty::ConnectInterThreshold));
		bodyPartConnectorCaffe->setMinSubsetCnt((int)poseExtractorCaffe->get(op::PoseProperty::ConnectMinSubsetCnt));
		bodyPartConnectorCaffe->setMinSubsetScore((float)poseExtractorCaffe->get(op::PoseProperty::ConnectMinSubsetScore));

		bodyPartConnectorCaffe->Forward_cpu({ heatMapsBlob.get(),
			peaksBlob.get() },
			mPoseKeypoints, mPoseScores);
		poseKeypoints = mPoseKeypoints;

		auto outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
		// Step 5 - Render poseKeypoints
		poseRenderer->renderPose(outputArray, mPoseKeypoints, scaleInputToOutput);
		// Step 6 - OpenPose output format to cv::Mat
		displayImage = opOutputToCvMat.formatToCvMat(outputArray);
	}
};

#ifdef __cplusplus
extern "C" {
#endif

	typedef void* c_OP;
	op::Array<float> output;
	std::array<op::Array<float>, 2> handOutput;

    OP_EXPORT c_OP newOP(int logging_level,
		char* output_resolution,
		char* net_resolution,
		char* model_pose,
		float alpha_pose,
		float scale_gap,
		int scale_number,
		float render_threshold,
		int num_gpu_start,
		bool disable_blending,
		char* model_folder,
		char* hand_net_resolution
	) {
		return new OpenPose(logging_level, output_resolution, net_resolution, model_pose, alpha_pose,
			scale_gap, scale_number, render_threshold, num_gpu_start, disable_blending, model_folder,
			hand_net_resolution);
	}
    OP_EXPORT void delOP(c_OP op) {
		delete (OpenPose *)op;
	}
    OP_EXPORT void forward(c_OP op, unsigned char* img, size_t rows, size_t cols, int* size, unsigned char* displayImg, bool display, bool hands) {
		OpenPose* openPose = (OpenPose*)op;
		cv::Mat image(rows, cols, CV_8UC3, img);
		cv::Mat displayImage(rows, cols, CV_8UC3, displayImg);
		openPose->forward(image, output, handOutput, displayImage, display, hands);
		if (output.getSize().size()) {
			size[0] = output.getSize()[0];
			size[1] = output.getSize()[1];
			size[2] = output.getSize()[2];
		}
		else {
			size[0] = 0; size[1] = 0; size[2] = 0;
		}
		if (display) memcpy(displayImg, displayImage.ptr(), sizeof(unsigned char)*rows*cols * 3);
	}
    OP_EXPORT void getOutputs(c_OP op, float* array) {
		if (output.getSize().size())
			memcpy(array, output.getPtr(), output.getSize()[0] * output.getSize()[1] * output.getSize()[2] * sizeof(float));
	}
	OP_EXPORT void getHandSize(c_OP op, int* size) {
		if (handOutput[0].getSize().size()) {
			size[0] = handOutput[0].getSize()[0];
			size[1] = handOutput[0].getSize()[1];
			size[2] = handOutput[0].getSize()[2];
		}
		else {
			size[0] = 0; size[1] = 0; size[2] = 0;
		}
	}
	OP_EXPORT void getHandOutputs(c_OP op, float* left, float* right) {
		// Don't really like how this looks. Maybe there is a better solution
		// utilising std::array more efficiently?
		std::size_t leftHandSize = handOutput[0].getSize()[0] * handOutput[0].getSize()[1] * handOutput[0].getSize()[2] * sizeof(float);
		std::size_t rightHandSize = handOutput[1].getSize()[0] * handOutput[1].getSize()[1] * handOutput[1].getSize()[2] * sizeof(float);
		if (handOutput[0].getSize().size() && handOutput[1].getSize().size())
			memcpy(left, handOutput[0].getPtr(), leftHandSize);
			memcpy(right, handOutput[1].getPtr(), rightHandSize);
	}
    OP_EXPORT void poseFromHeatmap(c_OP op, unsigned char* img, size_t rows, size_t cols, unsigned char* displayImg, float* hm, int* size, float* ratios) {
		OpenPose* openPose = (OpenPose*)op;
		cv::Mat image(rows, cols, CV_8UC3, img);
		cv::Mat displayImage(rows, cols, CV_8UC3, displayImg);

		std::vector<boost::shared_ptr<caffe::Blob<float>>> caffeNetOutputBlob;

		for (int i = 0; i<size[0]; i++) {
			boost::shared_ptr<caffe::Blob<float>> caffeHmPtr(new caffe::Blob<float>());
			caffeHmPtr->Reshape(1, size[1], size[2] * ((float)ratios[i] / (float)ratios[0]), size[3] * ((float)ratios[i] / (float)ratios[0]));
			float* startIndex = &hm[i*size[1] * size[2] * size[3]];
			for (int d = 0; d<caffeHmPtr->shape()[1]; d++) {
				for (int r = 0; r<caffeHmPtr->shape()[2]; r++) {
					for (int c = 0; c<caffeHmPtr->shape()[3]; c++) {
						int toI = d*caffeHmPtr->shape()[2] * caffeHmPtr->shape()[3] + r*caffeHmPtr->shape()[3] + c;
						int fromI = d*size[2] * size[3] + r*size[3] + c;
						caffeHmPtr->mutable_cpu_data()[toI] = startIndex[fromI];
					}
				}
			}
			caffeNetOutputBlob.emplace_back(caffeHmPtr);
		}

		std::vector<op::Point<int>> imageSizes;
		for (int i = 0; i<size[0]; i++) {
			op::Point<int> point(cols*ratios[i], rows*ratios[i]);
			imageSizes.emplace_back(point);
		}

		openPose->poseFromHeatmap(image, caffeNetOutputBlob, output, displayImage, imageSizes);
		memcpy(displayImg, displayImage.ptr(), sizeof(unsigned char)*rows*cols * 3);
		// Copy back kp size
		if (output.getSize().size()) {
			size[0] = output.getSize()[0];
			size[1] = output.getSize()[1];
			size[2] = output.getSize()[2];
		}
		else {
			size[0] = 0; size[1] = 0; size[2] = 0;
		}
	}

#ifdef __cplusplus
}
#endif

#endif
