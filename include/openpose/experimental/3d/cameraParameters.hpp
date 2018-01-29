#ifndef OPENPOSE_EXPERIMENTAL_3D_CAMERA_PARAMETERS_HPP
#define OPENPOSE_EXPERIMENTAL_3D_CAMERA_PARAMETERS_HPP

#include <opencv2/core/core.hpp>
#include <openpose/core/common.hpp>

namespace op
{
	OP_API const cv::Mat getIntrinsics(const int cameraIndex);

	OP_API const cv::Mat getDistorsion(const int cameraIndex);

	OP_API const cv::Mat getM(const int cameraIndex);

	OP_API std::vector<cv::Mat> getMs();

	OP_API unsigned long long getNumberCameras();
}

#endif // OPENPOSE_EXPERIMENTAL_3D_CAMERA_PARAMETERS_HPP
