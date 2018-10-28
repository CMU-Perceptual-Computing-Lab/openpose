#ifndef OPENPOSE_PRODUCER_IP_CAMERA_READER_HPP
#define OPENPOSE_PRODUCER_IP_CAMERA_READER_HPP

#include <openpose/core/common.hpp>
#include <openpose/producer/videoCaptureReader.hpp>

namespace op
{
    /**
     * IpCameraReader is a wrapper of the cv::VideoCapture class for IP camera streaming.
     */
    class OP_API IpCameraReader : public VideoCaptureReader
    {
    public:
        /**
         * Constructor of IpCameraReader. It opens the IP camera as a wrapper of cv::VideoCapture.
         * @param cameraPath const std::string parameter with the full camera IP link.
         */
        explicit IpCameraReader(const std::string& cameraPath);

        virtual ~IpCameraReader();

        std::vector<cv::Mat> getCameraMatrices();

        std::vector<cv::Mat> getCameraExtrinsics();

        std::vector<cv::Mat> getCameraIntrinsics();

        std::string getNextFrameName();

        inline bool isOpened() const
        {
            return VideoCaptureReader::isOpened();
        }

        inline double get(const int capProperty)
        {
            return VideoCaptureReader::get(capProperty);
        }

        inline void set(const int capProperty, const double value)
        {
            VideoCaptureReader::set(capProperty, value);
        }

    private:
        const std::string mPathName;

        cv::Mat getRawFrame();

        std::vector<cv::Mat> getRawFrames();

        DELETE_COPY(IpCameraReader);
    };
}

#endif // OPENPOSE_PRODUCER_IP_CAMERA_READER_HPP
