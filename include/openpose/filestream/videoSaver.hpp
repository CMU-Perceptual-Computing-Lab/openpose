#ifndef OPENPOSE__FILESTREAM__VIDEO_SAVER_HPP
#define OPENPOSE__FILESTREAM__VIDEO_SAVER_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> // cv::VideoWriter
#include "../utilities/macros.hpp"

namespace op
{
    class VideoSaver
    {
    public:
        VideoSaver(const std::string& videoSaverPath, const int cvFourcc, const double fps, const cv::Size& cvSize);

        VideoSaver(const std::vector<std::string>& videoSaverPaths, const int cvFourcc, const double fps, const cv::Size& cvSize);

        bool isOpened();

        void write(const cv::Mat& cvMat);

        void write(const std::vector<cv::Mat>& cvMats);

    private:
        std::vector<cv::VideoWriter> mVideoWriters;

        DELETE_COPY(VideoSaver);
    };
}

#endif // OPENPOSE__FILESTREAM__VIDEO_SAVER_HPP
