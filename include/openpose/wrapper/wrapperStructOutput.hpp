#ifndef OPENPOSE_WRAPPER_WRAPPER_STRUCT_OUTPUT_HPP
#define OPENPOSE_WRAPPER_WRAPPER_STRUCT_OUTPUT_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/enumClasses.hpp>
#include <openpose/gui/enumClasses.hpp>

namespace op
{
    /**
     * WrapperStructOutput: Output ( writing rendered results and/or pose data, etc.) configuration struct.
     */
    struct OP_API WrapperStructOutput
    {
        /**
         * Output verbose in the command line.
         * If -1, it will be disabled (default). If it is a positive integer number, it will print on"
         * the command line every `verbose` frames. If number in the range (0,1), it will print the"
         * progress every `verbose` times the total of frames.
         */
        double verbose;

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
         * Analogous to writeCocoJson but for foot keypoints.
         */
        std::string writeCocoFootJson;

        /**
         * Experimental option (only makes effect on car JSON generation).
         * It selects the COCO variant for cocoJsonSaver.
         */
        int writeCocoJsonVariant;

        /**
         * Rendered image saving folder.
         * If it is empty (default), it is disabled.
         */
        std::string writeImages;

        /**
         * Rendered image saving folder format.
         * Check your OpenCV version documentation for a list of compatible formats.
         * E.g., png, jpg, etc.
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
         * Rendered Adam images saving video path.
         * Please, use *.avi format.
         * If it is empty (default), it is disabled.
         */
        std::string writeVideoAdam;

        /**
         * Path to save a 3-D joint angle BVH file.
         * Please, use *.bvh format.
         * If it is empty (default), it is disabled.
         */
        std::string writeBvh;

        /**
         * Target server IP address for UDP client-server communication.
         */
        std::string udpHost;

        /**
         * Target server IP port for UDP client-server communication.
         */
        std::string udpPort;

        /**
         * Constructor of the struct.
         * It has the recommended and default values we recommend for each element of the struct.
         * Since all the elements of the struct are public, they can also be manually filled.
         */
        WrapperStructOutput(
            const double verbose = -1, const std::string& writeKeypoint = "",
            const DataFormat writeKeypointFormat = DataFormat::Xml, const std::string& writeJson = "",
            const std::string& writeCocoJson = "", const std::string& writeCocoFootJson = "",
            const int writeCocoJsonVariant = 1, const std::string& writeImages = "",
            const std::string& writeImagesFormat = "", const std::string& writeVideo = "",
            const double writeVideoFps = 30., const std::string& writeHeatMaps = "",
            const std::string& writeHeatMapsFormat = "", const std::string& writeVideoAdam = "",
            const std::string& writeBvh = "", const std::string& udpHost = "", const std::string& udpPort = "");
    };
}

#endif // OPENPOSE_WRAPPER_WRAPPER_STRUCT_OUTPUT_HPP
