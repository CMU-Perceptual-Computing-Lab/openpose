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

        virtual ~CameraParameterReader();

        // cameraExtrinsics is optional
        explicit CameraParameterReader(const std::string& serialNumber,
                                       const cv::Mat& cameraIntrinsics,
                                       const cv::Mat& cameraDistortion,
                                       const cv::Mat& cameraExtrinsics = cv::Mat(),
                                       const cv::Mat& cameraExtrinsicsInitial = cv::Mat());

        // serialNumbers is optional. If empty, it will load all the XML files available in the
        // cameraParameterPath folder
        void readParameters(const std::string& cameraParameterPath,
                            const std::vector<std::string>& serialNumbers = {});

        // It simply calls the previous readParameters with a single element
        void readParameters(const std::string& cameraParameterPath,
                            const std::string& serialNumber);

        void writeParameters(const std::string& cameraParameterPath) const;

        unsigned long long getNumberCameras() const;

        const std::vector<std::string>& getCameraSerialNumbers() const;

        const std::vector<cv::Mat>& getCameraMatrices() const;

        const std::vector<cv::Mat>& getCameraDistortions() const;

        const std::vector<cv::Mat>& getCameraIntrinsics() const;

        const std::vector<cv::Mat>& getCameraExtrinsics() const;

        const std::vector<cv::Mat>& getCameraExtrinsicsInitial() const;

        bool getUndistortImage() const;

        void setUndistortImage(const bool undistortImage);

        void undistort(cv::Mat& frame, const unsigned int cameraIndex = 0u);

    private:
        std::vector<std::string> mSerialNumbers;
        std::vector<cv::Mat> mCameraMatrices;
        std::vector<cv::Mat> mCameraDistortions;
        std::vector<cv::Mat> mCameraIntrinsics;
        std::vector<cv::Mat> mCameraExtrinsics;
        std::vector<cv::Mat> mCameraExtrinsicsInitial;

        // Undistortion (optional)
        bool mUndistortImage;
        std::vector<cv::Mat> mRemoveDistortionMaps1;
        std::vector<cv::Mat> mRemoveDistortionMaps2;

        DELETE_COPY(CameraParameterReader);
    };
}

#endif // OPENPOSE_3D_CAMERA_PARAMETER_READER_HPP
