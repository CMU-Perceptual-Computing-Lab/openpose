#include <openpose/filestream/videoSaver.hpp>

namespace op
{
    VideoSaver::VideoSaver(const std::string& videoSaverPath, const int cvFourcc, const double fps, const Point<int>& cvSize) :
        VideoSaver::VideoSaver{std::vector<std::string>{videoSaverPath}, cvFourcc, fps, cvSize}
    {
    }

    VideoSaver::VideoSaver(const std::vector<std::string>& videoSaverPaths, const int cvFourcc, const double fps, const Point<int>& cvSize)
    {
        try
        {
            if (cvSize.x <= 0 || cvSize.y <= 0)
                error("Desired frame size to save frames is <= 0.", __LINE__, __FUNCTION__, __FILE__);

            if (fps <= 0.)
                error("Desired fps to save frames is <= 0.", __LINE__, __FUNCTION__, __FILE__);

            for (const auto& videoSaverPath : videoSaverPaths)
            {
                mVideoWriters.emplace_back(videoSaverPath, cvFourcc, fps, cv::Size{cvSize.x, cvSize.y});

                if (!mVideoWriters.crbegin()->isOpened())
                {
                    const std::string errorMessage{"Video to write frames could not be opened on " + videoSaverPath + ". Please, "
                                                   "check OpenCV is properly compiled with the FFmpeg codecs in order to save video."};
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    bool VideoSaver::isOpened()
    {
        try
        {
            bool opened = (!mVideoWriters.empty());
            for (const auto& videoWriter : mVideoWriters)
            {
                if (!videoWriter.isOpened())
                {
                    opened = false;
                    break;
                }
            }
            return opened;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    void VideoSaver::write(const cv::Mat& cvMat)
    {
        try
        {
            write(std::vector<cv::Mat>{cvMat});
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void VideoSaver::write(const std::vector<cv::Mat>& cvMats)
    {
        try
        {
            if (!isOpened())
                error("Video to write frames is not opened.", __LINE__, __FUNCTION__, __FILE__);

            if (cvMats.size() != mVideoWriters.size())
                error("Size cvMats != size video writers");

            for (auto i = 0 ; i < mVideoWriters.size() ; i++)
                mVideoWriters[i].write(cvMats[i]);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
