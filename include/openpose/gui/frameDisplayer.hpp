#ifndef OPENPOSE_GUI_FRAMES_DISPLAY_HPP
#define OPENPOSE_GUI_FRAMES_DISPLAY_HPP

#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/common.hpp>
#include <openpose/gui/enumClasses.hpp>

namespace op
{
    /**
     *  The FrameDisplayer class is the one presenting visually the processed frame to the user.
     */
    class OP_API FrameDisplayer
    {
    public:
        /**
         * Constructor of the FrameDisplayer class.
         * @param fullScreen bool from which the FrameDisplayer::GuiDisplayMode property mGuiDisplayMode will be set, i.e. specifying the type of initial display (it can be changed later).
         * @param windowedSize const Point<int> with the windored output resolution (width and height).
         * @param windowedName const std::string value with the opencv resulting display name. Showed at the top-left part of the window.
         */
        FrameDisplayer(const Point<int>& windowedSize, const std::string& windowedName = "OpenPose Display", const bool fullScreen = false);

        // Due to OpenCV visualization issues (all visualization functions must be in the same thread)
        void initializationOnThread();

        /**
         * This function set the new FrameDisplayer::GuiDisplayMode (e.g. full screen).
         * @param displayMode New FrameDisplayer::GuiDisplayMode state.
         */
        void setGuiDisplayMode(const GuiDisplayMode displayMode);

        /**
         * This function switch between full screen and windowed modes (e.g. when double-click on video players or Ctrt+Enter are presed).
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
        const Point<int> mWindowedSize;
        const std::string mWindowName;
        GuiDisplayMode mGuiDisplayMode;
    };
}

#endif // OPENPOSE_GUI_FRAMES_DISPLAY_HPP
