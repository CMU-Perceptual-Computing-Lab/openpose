// #include <opencv2/opencv.hpp> // cv::imshow, cv::waitKey, cv::namedWindow, cv::setWindowProperty
#include <opencv2/highgui/highgui.hpp> // cv::imshow, cv::waitKey, cv::namedWindow, cv::setWindowProperty
#include <openpose/gui/frameDisplayer.hpp>

namespace op
{
    FrameDisplayer::FrameDisplayer(const std::string& windowedName, const Point<int>& initialWindowedSize,
                                   const bool fullScreen) :
        mWindowName{windowedName},
        mWindowedSize{initialWindowedSize},
        mFullScreenMode{(fullScreen ? FullScreenMode::FullScreen : FullScreenMode::Windowed)}
    {
        try
        {
            // If initial window size = 0 --> initialize to 640x480
            if (mWindowedSize.x <= 0 || mWindowedSize.y <= 0)
                mWindowedSize = Point<int>{640, 480};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void FrameDisplayer::initializationOnThread()
    {
        try
        {
            setFullScreenMode(mFullScreenMode);

            const cv::Mat blackFrame(mWindowedSize.y, mWindowedSize.x, CV_32FC3, {0,0,0});
            FrameDisplayer::displayFrame(blackFrame);
            // This one will show most probably a white image (I guess the program does not have time to render
            // in 1 msec)
            cv::waitKey(1);
            // // This one will show the desired black image
            // cv::waitKey(1000);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void FrameDisplayer::setFullScreenMode(const FullScreenMode fullScreenMode)
    {
        try
        {
            mFullScreenMode = fullScreenMode;

            // Setting output resolution
            cv::namedWindow(mWindowName, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
            if (mFullScreenMode == FullScreenMode::FullScreen)
                cv::setWindowProperty(mWindowName, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
            else if (mFullScreenMode == FullScreenMode::Windowed)
            {
                cv::resizeWindow(mWindowName, mWindowedSize.x, mWindowedSize.y);
                cv::setWindowProperty(mWindowName, CV_WND_PROP_FULLSCREEN, CV_WINDOW_NORMAL);
            }
            else
                error("Unknown FullScreenMode", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void FrameDisplayer::switchFullScreenMode()
    {
        try
        {
            if (mFullScreenMode == FullScreenMode::FullScreen)
                setFullScreenMode(FullScreenMode::Windowed);
            else if (mFullScreenMode == FullScreenMode::Windowed)
                setFullScreenMode(FullScreenMode::FullScreen);
            else
                error("Unknown FullScreenMode", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void FrameDisplayer::displayFrame(const cv::Mat& frame, const int waitKeyValue)
    {
        try
        {
            // Security check
            if (frame.empty())
                error("Empty frame introduced.", __LINE__, __FUNCTION__, __FILE__);
            // If frame > window size --> Resize window
            if (mWindowedSize.x < frame.cols || mWindowedSize.y < frame.rows)
            {
                mWindowedSize.x = std::max(mWindowedSize.x, frame.cols);
                mWindowedSize.y = std::max(mWindowedSize.y, frame.rows);
                cv::resizeWindow(mWindowName, mWindowedSize.x, mWindowedSize.y);
                // This one will show most probably a white image (I guess the program does not have time to render
                // in 1 msec)
                cv::waitKey(1);
            }
            cv::imshow(mWindowName, frame);
            if (waitKeyValue != -1)
                cv::waitKey(waitKeyValue);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void FrameDisplayer::displayFrame(const std::vector<cv::Mat>& frames, const int waitKeyValue)
    {
        try
        {
            // No frames
            if (frames.empty())
                displayFrame(cv::Mat(), waitKeyValue);
            // 1 frame
            else if (frames.size() == 1u)
                displayFrame(frames[0], waitKeyValue);
            // >= 2 frames
            else
            {
                // Prepare final cvMat
                // Concat (0)
                cv::Mat cvMat = frames[0].clone();
                // Concat (1,size()-1)
                for (auto i = 1u; i < frames.size(); i++)
                    cv::hconcat(cvMat, frames[i], cvMat);
                // Display it
                displayFrame(cvMat, waitKeyValue);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
