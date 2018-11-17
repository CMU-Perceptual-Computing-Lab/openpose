#ifndef OPENPOSE_WRAPPER_WRAPPER_STRUCT_GUI_HPP
#define OPENPOSE_WRAPPER_WRAPPER_STRUCT_GUI_HPP

#include <openpose/core/common.hpp>
#include <openpose/gui/enumClasses.hpp>

namespace op
{
    /**
     * WrapperStructGui: It controls a small GUI for quick visualization.
     */
    struct OP_API WrapperStructGui
    {
        /**
         * Display mode
         * a) -1 for automatic selection.
         * b) 0 for no display. Useful if there is no X server and/or to slightly speed up the processing if visual
         *    output is not required.
         * c) 2 for 2-D display in the OpenPose small integrated GUI.
         * d) 3 for 3-D display, if `--3d` was enabled.
         * e) 1 for both 2-D and 3-D display.
         */
        DisplayMode displayMode;

        /**
         * Whether to add some information to the frame (number of frame, number people detected, etc.) after it is
         * saved on disk and before it is displayed and/or returned to the user.
         */
        bool guiVerbose;

        /**
         * Whether to display the OpenPose small integrated GUI on fullscreen mode. It can be changed by interacting
         * with the GUI itself.
         */
        bool fullScreen;

        /**
         * Constructor of the struct.
         * It has the recommended and default values we recommend for each element of the struct.
         * Since all the elements of the struct are public, they can also be manually filled.
         */
        WrapperStructGui(
            const DisplayMode displayMode = DisplayMode::NoDisplay, const bool guiVerbose = false,
            const bool fullScreen = false);
    };
}

#endif // OPENPOSE_WRAPPER_WRAPPER_STRUCT_GUI_HPP
