#ifndef OPENPOSE_GUI_ENUM_CLASSES_HPP
#define OPENPOSE_GUI_ENUM_CLASSES_HPP

namespace op
{
    /**
    * GUI display modes.
     * An enum class with the different output screen options. E.g., 2-D, 3-D, all, none.
     */
    enum class DisplayMode : unsigned short
    {
        NoDisplay,  /**< No display. */
        DisplayAll, /**< All (2-D and 3-D/Adam) displays */
        Display2D,  /**< Only 2-D display. */
        Display3D,  /**< Only 3-D display. */
        DisplayAdam /**< Only Adam display. */
    };

    /**
     * Full screen modes.
     * An enum class with the different full screen mode options, i.e., full screen or windored.
     */
    enum class FullScreenMode : bool
    {
        FullScreen, /**< Full screen mode. */
        Windowed,   /**< Windowed mode, depending on the frame output size. */
    };
}

#endif // OPENPOSE_GUI_ENUM_CLASSES_HPP
