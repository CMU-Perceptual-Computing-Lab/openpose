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
        String writeKeypoint;

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
        String writeJson;

        /**
         * Pose (x, y, score) locations saving folder location in JSON COCO validation format.
         * If it is empty (default), it is disabled.
         */
        String writeCocoJson;

        /**
         * It selects the COCO variants for cocoJsonSaver.
         * Add 1 for body, add 2 for foot, 4 for face, and/or 8 for hands. Use 0 to use all the possible candidates.
         * E.g., 7 would mean body+foot+face COCO JSON..
         */
        int writeCocoJsonVariants;

        /**
         * Experimental option (only makes effect on car JSON generation).
         * It selects the COCO variant for cocoJsonSaver.
         */
        int writeCocoJsonVariant;

        /**
         * Rendered image saving folder.
         * If it is empty (default), it is disabled.
         */
        String writeImages;

        /**
         * Rendered image saving folder format.
         * Check your OpenCV version documentation for a list of compatible formats.
         * E.g., png, jpg, etc.
         * If writeImages is empty (default), it makes no effect.
         */
        String writeImagesFormat;

        /**
         * Rendered images saving video path.
         * Please, use *.avi format.
         * If it is empty (default), it is disabled.
         */
        String writeVideo;

        /**
         * Frame rate of the recorded video.
         * By default (-1.), it will try to get the input frames producer frame rate (e.g., input video or webcam frame
         * rate). If the input frames producer does not have a set FPS (e.g., image_dir or webcam if OpenCV not
         * compiled with its support), set this value accordingly (e.g., to the frame rate displayed by the OpenPose
         * GUI).
         */
        double writeVideoFps;

        /**
         * Whether to save the output video with audio. The input producer must be a video too.
         */
        bool writeVideoWithAudio;

        /**
         * Rendered heat maps saving folder.
         * In order to save the heatmaps, WrapperStructPose.heatMapTypes must also be filled.
         * If it is empty (default), it is disabled.
         */
        String writeHeatMaps;

        /**
         * Heat maps image saving format.
         * Analogous to writeImagesFormat.
         */
        String writeHeatMapsFormat;

        /**
         * Rendered 3D images saving video path.
         * Please, use *.avi format.
         * If it is empty (default), it is disabled.
         */
        String writeVideo3D;

        /**
         * Rendered Adam images saving video path.
         * Please, use *.avi format.
         * If it is empty (default), it is disabled.
         */
        String writeVideoAdam;

        /**
         * Path to save a 3-D joint angle BVH file.
         * Please, use *.bvh format.
         * If it is empty (default), it is disabled.
         */
        String writeBvh;

        /**
         * Target server IP address for UDP client-server communication.
         */
        String udpHost;

        /**
         * Target server IP port for UDP client-server communication.
         */
        String udpPort;

        /**
         * Constructor of the struct.
         * It has the recommended and default values we recommend for each element of the struct.
         * Since all the elements of the struct are public, they can also be manually filled.
         */
        WrapperStructOutput(
            const double verbose = -1, const String& writeKeypoint = "",
            const DataFormat writeKeypointFormat = DataFormat::Xml, const String& writeJson = "",
            const String& writeCocoJson = "", const int writeCocoJsonVariants = 1,
            const int writeCocoJsonVariant = 1, const String& writeImages = "",
            const String& writeImagesFormat = "png", const String& writeVideo = "",
            const double writeVideoFps = -1., const bool writeVideoWithAudio = false,
            const String& writeHeatMaps = "", const String& writeHeatMapsFormat = "png",
            const String& writeVideo3D = "", const String& writeVideoAdam = "",
            const String& writeBvh = "", const String& udpHost = "",
            const String& udpPort = "8051");
    };
}

#endif // OPENPOSE_WRAPPER_WRAPPER_STRUCT_OUTPUT_HPP
