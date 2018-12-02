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
        VideoSaver(const std::string& videoSaverPath, const int cvFourcc, const double fps);

        virtual ~VideoSaver();

        bool isOpened();

        void write(const cv::Mat& cvMat);

        void write(const std::vector<cv::Mat>& cvMats);

    private:
        const std::string mVideoSaverPath;
        const int mCvFourcc;
        const double mFps;
        Point<int> mCvSize;
        bool mVideoStarted;
        cv::VideoWriter mVideoWriter;
        unsigned int mNumberImages;

        DELETE_COPY(VideoSaver);
    };
}

#endif // OPENPOSE_FILESTREAM_VIDEO_SAVER_HPP
