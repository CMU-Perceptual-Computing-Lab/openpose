#include <openpose/filestream/videoSaver.hpp>

namespace op
{
    cv::VideoWriter openVideo(const std::string& videoSaverPath, const int cvFourcc, const double fps,
                              const Point<int>& cvSize, const int numberImages)
    {
        try
        {
            return cv::VideoWriter{videoSaverPath, cvFourcc, fps, cv::Size{numberImages*cvSize.x, cvSize.y}};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return cv::VideoWriter{};
        }
    }

    VideoSaver::VideoSaver(const std::string& videoSaverPath, const int cvFourcc, const double fps,
                           const Point<int>& cvSize) :
        mVideoSaverPath{videoSaverPath},
        mCvFourcc{cvFourcc},
        mFps{fps},
        mCvSize{cvSize},
        mNumberImages{1}
    {
        try
        {
            // Sanity checks
            if (cvSize.x <= 0 || cvSize.y <= 0)
                error("Desired frame size to save the video is <= 0.", __LINE__, __FUNCTION__, __FILE__);
            if (fps <= 0.)
                error("Desired fps (frame rate) to save the video is <= 0.", __LINE__, __FUNCTION__, __FILE__);
            // Open video-writter
            mVideoWriter = openVideo(mVideoSaverPath, mCvFourcc, mFps, mCvSize, mNumberImages);
            // Check it was successfully opened
            if (!mVideoWriter.isOpened())
            {
                const std::string errorMessage{"Video to write frames could not be opened as `" + videoSaverPath
                                               + "`. Please, check that:\n\t1. The path ends in `.avi`."
                                               "\n\t2. The parent folder exists.\n\t3. OpenCV is properly"
                                               " compiled with the FFmpeg codecs in order to save video."};
                error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    VideoSaver::~VideoSaver()
    {
    }

    bool VideoSaver::isOpened()
    {
        try
        {
            return mVideoWriter.isOpened();
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
            // Sanity checks
            if (!isOpened())
                error("Video to write frames is not opened.", __LINE__, __FUNCTION__, __FILE__);
            if (cvMats.empty())
                error("The image(s) to be saved cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
            // Re-shape video writer if required
            if (mNumberImages != cvMats.size())
            {
                mNumberImages = (unsigned int)cvMats.size();
                mVideoWriter = openVideo(mVideoSaverPath, mCvFourcc, mFps, mCvSize, mNumberImages);
            }
            // Concat images
            cv::Mat cvOutputData;
            if (cvMats.size() > 1)
                cv::hconcat(cvMats.data(), cvMats.size(), cvOutputData);
            else
                cvOutputData = cvMats.at(0);
            // Save concatenated image
            mVideoWriter.write(cvOutputData);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
