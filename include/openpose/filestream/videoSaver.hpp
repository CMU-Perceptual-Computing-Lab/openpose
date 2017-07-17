#ifndef OPENPOSE_FILESTREAM_VIDEO_SAVER_HPP
#define OPENPOSE_FILESTREAM_VIDEO_SAVER_HPP

#include <opencv2/core/core.hpp> // cv::Mat
#include <opencv2/highgui/highgui.hpp> // cv::VideoWriter
#include <openpose/core/common.hpp>

namespace op
{
    class OP_API VideoSaver
    {
    public:
        VideoSaver(const std::string& videoSaverPath, const int cvFourcc, const double fps, const Point<int>& cvSize);

        VideoSaver(const std::vector<std::string>& videoSaverPaths, const int cvFourcc, const double fps, const Point<int>& cvSize);

        bool isOpened();

        void write(const cv::Mat& cvMat);

        void write(const std::vector<cv::Mat>& cvMats);

    private:
        std::vector<cv::VideoWriter> mVideoWriters;

        DELETE_COPY(VideoSaver);
    };
}

#endif // OPENPOSE_FILESTREAM_VIDEO_SAVER_HPP
