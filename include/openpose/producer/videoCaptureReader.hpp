#ifndef OPENPOSE_PRODUCER_VIDEO_CAPTURE_READER_HPP
#define OPENPOSE_PRODUCER_VIDEO_CAPTURE_READER_HPP

#include <opencv2/core/core.hpp> // cv::Mat
#include <opencv2/highgui/highgui.hpp> // cv::VideoCapture
#include <openpose/core/common.hpp>
#include <openpose/producer/producer.hpp>

namespace op
{
    /**
     *  VideoCaptureReader is an abstract class to extract frames from a cv::VideoCapture source (video file,
     * webcam stream, etc.). It has the basic and common functions of the cv::VideoCapture class (e.g. get, set, etc.).
     */
    class OP_API VideoCaptureReader : public Producer
    {
    public:
        /**
         * This constructor of VideoCaptureReader wraps cv::VideoCapture(const int).
         * @param index const int indicating the cv::VideoCapture constructor int argument, in the range [0, 9].
         */
        explicit VideoCaptureReader(const int index, const bool throwExceptionIfNoOpened);

        /**
         * This constructor of VideoCaptureReader wraps cv::VideoCapture(const std::string).
         * @param path const std::string indicating the cv::VideoCapture constructor string argument.
         * @param producerType const std::string indicating whether the frame source is an IP camera or video.
         */
        explicit VideoCaptureReader(const std::string& path, const ProducerType producerType);

        /**
         * Destructor of VideoCaptureReader. It releases the cv::VideoCapture member. It is virtual so that
         * any children class can implement its own destructor.
         */
        virtual ~VideoCaptureReader();

        virtual std::string getNextFrameName() = 0;

        inline bool isOpened() const
        {
            return mVideoCapture.isOpened();
        }

        void release();

        virtual double get(const int capProperty) = 0;

        virtual void set(const int capProperty, const double value) = 0;

    protected:
        virtual cv::Mat getRawFrame() = 0;

        virtual std::vector<cv::Mat> getRawFrames() = 0;

    private:
        cv::VideoCapture mVideoCapture;

        DELETE_COPY(VideoCaptureReader);
    };
}

#endif // OPENPOSE_PRODUCER_VIDEO_CAPTURE_READER_HPP
