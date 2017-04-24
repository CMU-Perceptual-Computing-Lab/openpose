#ifndef OPENPOSE__GUI__FRAMES_DISPLAY_HPP
#define OPENPOSE__GUI__FRAMES_DISPLAY_HPP

#include <string>
#include <opencv2/core/core.hpp>
#include "enumClasses.hpp"

namespace op
{
    /**
     *  The FrameDisplayer class is the one presenting visually the processed frame to the user.
     */
    class FrameDisplayer
    {
    public:
        /**
         * Constructor of the FrameDisplayer class.
         * @param fullScreen bool from which the FrameDisplayer::GuiDisplayMode property mGuiDisplayMode will be set, i.e. specifying the type of initial display (it can be changed later).
         * @param windowedSize const cv::Size with the windored output resolution (width and height).
         * @param windowedName const std::string value with the opencv resulting display name. Showed at the top-left part of the window.
         */
        FrameDisplayer(const cv::Size& windowedSize, const std::string& windowedName = "OpenPose Display", const bool fullScreen = false);

        /**
         * This function set the new FrameDisplayer::GuiDisplayMode (e.g. full screen).
         * @param displayMode New FrameDisplayer::GuiDisplayMode state.
         */
        void setGuiDisplayMode(const GuiDisplayMode displayMode);

        /**
         * This function switch between full screen and windowed modes (e.g. when double click on video players or Ctrt+Enter are presed).
         */
        void switchGuiDisplayMode();

        /**
         * This function displays an image on the display.
         * @param frame cv::Mat image to display.
         * @param waitKeyValue int value that specifies the argument parameter for cv::waitKey (see OpenCV documentation for more information). Special cases: select -1
         * not to use cv::waitKey or 0 for cv::waitKey(0). OpenCV doc: http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=waitkey
         */
        void displayFrame(const cv::Mat& frame, const int waitKeyValue = -1);

    private:
        const cv::Size mWindowedSize;
        const std::string mWindowName;
        GuiDisplayMode mGuiDisplayMode;
    };
}

#endif // OPENPOSE__GUI__FRAMES_DISPLAY_HPP
