#include <openpose/filestream/videoSaver.hpp>

namespace op
{
    cv::VideoWriter openVideo(const std::string& videoSaverPath, const int cvFourcc, const double fps,
                              const Point<int>& cvSize)
    {
        try
        {
            // Open video
            const cv::VideoWriter videoWriter{
                videoSaverPath, cvFourcc, fps, cv::Size{cvSize.x, cvSize.y}};
            // Check it was successfully opened
            if (!videoWriter.isOpened())
            {
                const std::string errorMessage{
                    "Video to write frames could not be opened as `" + videoSaverPath + "`. Please, check that:"
                    "\n\t1. The path ends in `.avi`.\n\t2. The parent folder exists.\n\t3. OpenCV is properly"
                    " compiled with the FFmpeg codecs in order to save video."
                    "\n\t4. You are not saving in a protected folder. If you desire to save a video in a"
                    " protected folder, use sudo (Ubuntu) or execute the binary file as administrator (Windows)."};
                error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
            }
            // Return video
            return videoWriter;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return cv::VideoWriter{};
        }
    }

    VideoSaver::VideoSaver(const std::string& videoSaverPath, const int cvFourcc, const double fps) :
        mVideoSaverPath{videoSaverPath},
        mCvFourcc{cvFourcc},
        mFps{fps},
        mVideoStarted{false}
    {
        try
        {
            // Sanity check
            if (fps <= 0.)
                error("Desired fps (frame rate) to save the video is <= 0.", __LINE__, __FUNCTION__, __FILE__);
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
            // Sanity check
            if (cvMats.empty())
                error("The image(s) to be saved cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
            for (const auto& cvMat : cvMats)
                if (cvMat.empty())
                    error("The image(s) to be saved cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
            // Open video (1st frame)
            // Done here and not in the constructor to handle cases where the resolution is not known (e.g.,
            // reading images or multiple cameras)
            if (!mVideoStarted)
            {
                mVideoStarted = true;
                const auto cvSize = cvMats.at(0).size();
                mCvSize = Point<int>{(int)cvMats.size()*cvSize.width, cvSize.height};
                mVideoWriter = openVideo(mVideoSaverPath, mCvFourcc, mFps, mCvSize);
            }
            // Sanity check
            if (!isOpened())
                error("Video to write frames is not opened.", __LINE__, __FUNCTION__, __FILE__);
            // Concat images
            cv::Mat cvOutputData;
            if (cvMats.size() > 1)
                cv::hconcat(cvMats.data(), cvMats.size(), cvOutputData);
            else
                cvOutputData = cvMats.at(0);
            // Sanity check
            if (mCvSize.x != cvOutputData.cols || mCvSize.y != cvOutputData.rows)
                error("You selected to write video (`--write_video`), but the frames to be saved have different"
                      " resolution. You can only save frames with the same resolution.",
                      __LINE__, __FUNCTION__, __FILE__);
            // Save concatenated image
            mVideoWriter.write(cvOutputData);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
