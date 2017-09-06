#include <opencv2/highgui/highgui.hpp> // cv::imshow, cv::waitKey, cv::namedWindow, cv::setWindowProperty
#include <openpose/gui/frameDisplayer.hpp>

namespace op
{
    FrameDisplayer::FrameDisplayer(const Point<int>& windowedSize, const std::string& windowedName, const bool fullScreen) :
        mWindowedSize{windowedSize},
        mWindowName{windowedName},
        mGuiDisplayMode{(fullScreen ? GuiDisplayMode::FullScreen : GuiDisplayMode::Windowed)}
    {
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
