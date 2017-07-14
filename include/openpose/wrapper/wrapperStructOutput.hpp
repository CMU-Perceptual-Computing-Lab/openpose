#ifndef OPENPOSE_WRAPPER_WRAPPER_STRUCT_OUTPUT_HPP
#define OPENPOSE_WRAPPER_WRAPPER_STRUCT_OUTPUT_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/enumClasses.hpp>

namespace op
{
    /**
     * WrapperStructOutput: Output (small GUI, writing rendered results and/or pose data, etc.) configuration struct.
     * WrapperStructOutput allows the user to set up the input frames generator.
     */
    struct OP_API WrapperStructOutput
    {
        /**
         * Whether to display the OpenPose small integrated GUI.
         */
        bool displayGui;

        /**
         * Whether to add some information to the frame (number of frame, number people detected, etc.) after it is saved on disk
         * and before it is displayed and/or returned to the user.
         */
        bool guiVerbose;

        /**
         * Whether to display the OpenPose small integrated GUI on fullscreen mode. It can be changed by interacting with the GUI itself.
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
         * Pose (x, y, score) locations saving folder location in JSON format (e.g. useful when needed JSON but using OpenCV < 3.0).
         * If it is empty (default), it is disabled.
         */
        std::string writeKeypointJson;

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
         * Constructor of the struct.
         * It has the recommended and default values we recommend for each element of the struct.
         * Since all the elements of the struct are public, they can also be manually filled.
         */
        WrapperStructOutput(const bool displayGui = false, const bool guiVerbose = false, const bool fullScreen = false, const std::string& writeKeypoint = "",
                            const DataFormat writeKeypointFormat = DataFormat::Xml, const std::string& writeKeypointJson = "", const std::string& writeCocoJson = "",
                            const std::string& writeImages = "", const std::string& writeImagesFormat = "", const std::string& writeVideo = "",
                            const std::string& writeHeatMaps = "", const std::string& writeHeatMapsFormat = "");
    };
}

#endif // OPENPOSE_WRAPPER_WRAPPER_STRUCT_OUTPUT_HPP
