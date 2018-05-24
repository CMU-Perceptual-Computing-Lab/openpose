#ifndef OPENPOSE_3D_CAMERA_PARAMETER_READER_HPP
#define OPENPOSE_3D_CAMERA_PARAMETER_READER_HPP

#include <opencv2/core/core.hpp>
#include <openpose/core/common.hpp>

namespace op
{
    class OP_API CameraParameterReader
    {
    public:
        explicit CameraParameterReader();

        // cameraExtrinsics is optional
        explicit CameraParameterReader(const std::string& serialNumber,
                                       const cv::Mat& cameraIntrinsics,
                                       const cv::Mat& cameraDistortion,
                                       const cv::Mat& cameraExtrinsics = cv::Mat());

        void readParameters(const std::string& cameraParameterPath,
                            const std::vector<std::string>& serialNumbers);

        void writeParameters(const std::string& cameraParameterPath) const;

        unsigned long long getNumberCameras() const;

        const std::vector<cv::Mat>& getCameraMatrices() const;

        const std::vector<cv::Mat>& getCameraExtrinsics() const;

        const std::vector<cv::Mat>& getCameraIntrinsics() const;

        const std::vector<cv::Mat>& getCameraDistortions() const;

    private:
        std::vector<std::string> mSerialNumbers;
        unsigned long long mNumberCameras;
        std::vector<cv::Mat> mCameraMatrices;
        std::vector<cv::Mat> mCameraExtrinsics;
        std::vector<cv::Mat> mCameraIntrinsics;
        std::vector<cv::Mat> mCameraDistortions;

        DELETE_COPY(CameraParameterReader);
    };
}

#endif // OPENPOSE_3D_CAMERA_PARAMETER_READER_HPP
