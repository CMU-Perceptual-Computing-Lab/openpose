#ifndef OPENPOSE_WRAPPER_WRAPPER_STRUCT_INPUT_HPP
#define OPENPOSE_WRAPPER_WRAPPER_STRUCT_INPUT_HPP

#include <limits> // std::numeric_limits
#include <openpose/core/common.hpp>
#include <openpose/producer/producer.hpp>

namespace op
{
    /**
     * WrapperStructInput: Input (images, video, webcam, etc.) configuration struct.
     * WrapperStructInput allows the user to set up the input frames generator.
     */
    struct OP_API WrapperStructInput
    {
        /**
         * Desired type of producer (FlirCamera, ImageDirectory, IPCamera, Video, Webcam, None, etc.).
         * Default: ProducerType::None.
         */
        ProducerType producerType;

        /**
         * Path of the producer (image directory path for ImageDirectory, video path for Video,
         * camera index for Webcam and FlirCamera, URL for IPCamera, etc.).
         * Default: "".
         */
        String producerString;

        /**
         * First image to process.
         * Default: 0.
         */
        unsigned long long frameFirst;

        /**
         * Step or gap across processed frames.
         * Default: 1 (i.e., process all frames).
         * Example: A value of 5 would mean to process frames 0, 5, 10, etc.
         */
        unsigned long long frameStep;

        /**
         * Last image to process.
         * Default: -1 (i.e., process all frames).
         */
        unsigned long long frameLast;

        /**
         * Whether to skip or sleep in order to keep the same FPS as the frames producer.
         */
        bool realTimeProcessing;

        /**
         * Whether to flip (mirror) the image.
         */
        bool frameFlip;

        /**
         * Image rotation.
         * Only 4 possible values: 0 (default, no rotation), 90, 180 or 270 degrees
         */
        int frameRotate;

        /**
         * Whether to re-open the producer if it reaches the end (e.g., video or image directory after the last frame).
         */
        bool framesRepeat;

        /**
         * Camera resolution (only for Webcam and FlirCamera).
         */
        Point<int> cameraResolution;

        /**
         * Directory path for the camera parameters (intrinsic and extrinsic parameters) or optionally XML file
         * full path (if only 1 view).
         */
        String cameraParameterPath;

        /**
         * Whether to undistort the image given the camera parameters.
         */
        bool undistortImage;

        /**
         * Number of camera views.
         * Complementary option for `--image_dir` or `--video`
         * It is -1 for Flir cameras (# cameras detected at runtime), as well as for any other frames source (-1 and 1
         * are equivalent for those). It could be greater than 1 only for prerecorded produced sources, such as video
         * and image directory.
         */
        int numberViews;

        /**
         * Constructor of the struct.
         * It has the recommended and default values we recommend for each element of the struct.
         * Since all the elements of the struct are public, they can also be manually filled.
         */
        WrapperStructInput(
            const ProducerType producerType = ProducerType::None, const String& producerString = "",
            const unsigned long long frameFirst = 0, const unsigned long long frameStep = 1,
            const unsigned long long frameLast = std::numeric_limits<unsigned long long>::max(),
            const bool realTimeProcessing = false, const bool frameFlip = false, const int frameRotate = 0,
            const bool framesRepeat = false, const Point<int>& cameraResolution = Point<int>{-1,-1},
            const String& cameraParameterPath = "models/cameraParameters/",
            const bool undistortImage = false, const int numberViews = -1);
    };
}

#endif // OPENPOSE_WRAPPER_WRAPPER_STRUCT_INPUT_HPP
