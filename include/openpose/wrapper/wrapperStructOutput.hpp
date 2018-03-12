#ifndef OPENPOSE_WRAPPER_WRAPPER_STRUCT_OUTPUT_HPP
#define OPENPOSE_WRAPPER_WRAPPER_STRUCT_OUTPUT_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/enumClasses.hpp>
#include <openpose/gui/enumClasses.hpp>

namespace op
{
    /**
     * WrapperStructOutput: Output (small GUI, writing rendered results and/or pose data, etc.) configuration struct.
     * WrapperStructOutput allows the user to set up the input frames generator.
     */
    struct OP_API WrapperStructOutput
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
         * Pose (x, y, score) locations saving folder location.
         * If it is empty (default), it is disabled.
         * Select format with writeKeypointFormat.
         */
        std::string writeKeypoint;

        /**
         * Data format to save Pose (x, y, score) locations.
         * Options: DataFormat::Json (default), DataFormat::Xml and DataFormat::Yml (equivalent to DataFormat::Yaml)
         * JSON option only available for OpenCV >= 3.0.
         */
        DataFormat writeKeypointFormat;

        /**
         * Directory to write OpenPose output in JSON format.
         * If it is empty (default), it is disabled.
         * It includes:
         *     - `people` field with body, hand, and face pose keypoints in (x, y, score) format.
         *     - `part_candidates` field with body part candidates in (x, y, score) format (if enabled with
         *       `--part_candidates`).
         */
        std::string writeJson;

        /**
         * Pose (x, y, score) locations saving folder location in JSON COCO validation format.
         * If it is empty (default), it is disabled.
         */
        std::string writeCocoJson;

        /**
         * Rendered image saving folder.
         * If it is empty (default), it is disabled.
         */
        std::string writeImages;

        /**
         * Rendered image saving folder format.
         * Check your OpenCV version documentation for a list of compatible formats.
         * E.g. png, jpg, etc.
         * If writeImages is empty (default), it makes no effect.
         */
        std::string writeImagesFormat;

        /**
         * Rendered images saving video path.
         * Please, use *.avi format.
         * If it is empty (default), it is disabled.
         */
        std::string writeVideo;

        /**
         * Rendered heat maps saving folder.
         * In order to save the heatmaps, WrapperStructPose.heatMapTypes must also be filled.
         * If it is empty (default), it is disabled.
         */
        std::string writeHeatMaps;

        /**
         * Heat maps image saving format.
         * Analogous to writeImagesFormat.
         */
        std::string writeHeatMapsFormat;

        /**
         * Frame rate of the recorded video.
         */
        double writeVideoFps;

        /**
         * Constructor of the struct.
         * It has the recommended and default values we recommend for each element of the struct.
         * Since all the elements of the struct are public, they can also be manually filled.
         */
        WrapperStructOutput(const DisplayMode displayMode = DisplayMode::NoDisplay, const bool guiVerbose = false,
                            const bool fullScreen = false, const std::string& writeKeypoint = "",
                            const DataFormat writeKeypointFormat = DataFormat::Xml,
                            const std::string& writeJson = "", const std::string& writeCocoJson = "",
                            const std::string& writeImages = "", const std::string& writeImagesFormat = "",
                            const std::string& writeVideo = "", const double writeVideoFps = 30.,
                            const std::string& writeHeatMaps = "", const std::string& writeHeatMapsFormat = "");
    };
}

#endif // OPENPOSE_WRAPPER_WRAPPER_STRUCT_OUTPUT_HPP
