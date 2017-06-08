#ifndef OPENPOSE_GUI_ENUM_CLASSES_HPP
#define OPENPOSE_GUI_ENUM_CLASSES_HPP

namespace op
{
    /** 
     * GUI display modes.
     * An enum class with the different output screen options (e.g. full screen, windored or disabling the display).
     */
    enum class GuiDisplayMode : bool
    {
        FullScreen, /**< Full screen mode. */
        Windowed,   /**< Windowed mode, depending on the frame output size. */
        // NoDisplay,  /**< Not displaying the output. */
    };
}

#endif // OPENPOSE_GUI_ENUM_CLASSES_HPP
