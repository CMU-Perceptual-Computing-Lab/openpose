#ifndef OPENPOSE_EXPERIMENTAL_3D_CAMERA_PARAMETER_READER_HPP
#define OPENPOSE_EXPERIMENTAL_3D_CAMERA_PARAMETER_READER_HPP

#include <opencv2/core/core.hpp>
#include <openpose/core/common.hpp>

namespace op
{
    class OP_API CameraParameterReader
    {
    public:
        explicit CameraParameterReader();

        void readParameters(const std::string& cameraParameterPath,
                            const std::vector<std::string>& serialNumbers);

        unsigned long long getNumberCameras() const;

        const std::vector<cv::Mat>& getCameraMatrices() const;

        const std::vector<cv::Mat>& getCameraIntrinsics() const;

        const std::vector<cv::Mat>& getCameraDistortions() const;

    private:
        std::vector<std::string> mSerialNumbers;
        unsigned long long mNumberCameras;
        std::vector<cv::Mat> mCameraMatrices;
        std::vector<cv::Mat> mCameraIntrinsics;
        std::vector<cv::Mat> mCameraDistortions;

        DELETE_COPY(CameraParameterReader);
    };
}

#endif // OPENPOSE_EXPERIMENTAL_3D_CAMERA_PARAMETER_READER_HPP
