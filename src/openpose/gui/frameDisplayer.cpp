#include <opencv2/opencv.hpp> // cv::imshow, cv::waitKey, cv::namedWindow, cv::setWindowProperty
#include <opencv2/highgui/highgui.hpp> // cv::imshow, cv::waitKey, cv::namedWindow, cv::setWindowProperty
#include <openpose/gui/frameDisplayer.hpp>

namespace op
{
    FrameDisplayer::FrameDisplayer(const std::string& windowedName, const Point<int>& initialWindowedSize, const bool fullScreen) :
        mWindowName{windowedName},
        mWindowedSize{initialWindowedSize},
        mGuiDisplayMode{(fullScreen ? GuiDisplayMode::FullScreen : GuiDisplayMode::Windowed)}
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
            setGuiDisplayMode(mGuiDisplayMode);

            const cv::Mat blackFrame(mWindowedSize.y, mWindowedSize.x, CV_32FC3, {0,0,0});
            FrameDisplayer::displayFrame(blackFrame);
            cv::waitKey(1); // This one will show most probably a white image (I guess the program does not have time to render in 1 msec)
            // cv::waitKey(1000); // This one will show the desired black image
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void FrameDisplayer::setGuiDisplayMode(const GuiDisplayMode displayMode)
    {
        try
        {
            mGuiDisplayMode = displayMode;

            // Setting output resolution
            cv::namedWindow(mWindowName, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
            if (mGuiDisplayMode == GuiDisplayMode::FullScreen)
                cv::setWindowProperty(mWindowName, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
            else if (mGuiDisplayMode == GuiDisplayMode::Windowed)
            {
                cv::resizeWindow(mWindowName, mWindowedSize.x, mWindowedSize.y);
                cv::setWindowProperty(mWindowName, CV_WND_PROP_FULLSCREEN, CV_WINDOW_NORMAL);
            }
            else
                error("Unknown GuiDisplayMode", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void FrameDisplayer::switchGuiDisplayMode()
    {
        try
        {
            if (mGuiDisplayMode == GuiDisplayMode::FullScreen)
                setGuiDisplayMode(GuiDisplayMode::Windowed);
            else if (mGuiDisplayMode == GuiDisplayMode::Windowed)
                setGuiDisplayMode(GuiDisplayMode::FullScreen);
            else
                error("Unknown GuiDisplayMode", __LINE__, __FUNCTION__, __FILE__);
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
            // If frame > window size --> Resize window
            if (mWindowedSize.x < frame.cols || mWindowedSize.y < frame.rows)
            {
                mWindowedSize.x = std::max(mWindowedSize.x, frame.cols);
                mWindowedSize.y = std::max(mWindowedSize.y, frame.rows);
                cv::resizeWindow(mWindowName, mWindowedSize.x, mWindowedSize.y);
                cv::waitKey(1); // This one will show most probably a white image (I guess the program does not have time to render in 1 msec)
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
}
