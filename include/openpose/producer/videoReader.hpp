#ifndef OPENPOSE_PRODUCER_VIDEO_READER_HPP
#define OPENPOSE_PRODUCER_VIDEO_READER_HPP

#include <openpose/core/common.hpp>
#include <openpose/producer/videoCaptureReader.hpp>

namespace op
{
    /**
     * VideoReader is a wrapper of the cv::VideoCapture class for video. It allows controlling a webcam (extracting frames,
     * setting resolution & fps, etc).
     */
    class OP_API VideoReader : public VideoCaptureReader
    {
    public:
        /**
         * Constructor of VideoReader. It opens the video as a wrapper of cv::VideoCapture. It includes a flag to indicate
         * whether the video should be repeated once it is completely read.
         * @param videoPath const std::string parameter with the full video path location.
         */
        explicit VideoReader(const std::string& videoPath);

        std::string getFrameName();

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

        DELETE_COPY(VideoReader);
    };
}

#endif // OPENPOSE_PRODUCER_VIDEO_READER_HPP
