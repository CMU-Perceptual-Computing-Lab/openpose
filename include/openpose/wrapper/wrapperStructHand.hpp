#ifndef OPENPOSE_WRAPPER_WRAPPER_STRUCT_HAND_HPP
#define OPENPOSE_WRAPPER_WRAPPER_STRUCT_HAND_HPP

#include <openpose/core/enumClasses.hpp>
#include <openpose/core/point.hpp>
#include <openpose/hand/handParameters.hpp>

namespace op
{
    /**
     * WrapperStructHand: Hand estimation and rendering configuration struct.
     * WrapperStructHand allows the user to set up the hand estimation and rendering parameters that will be used for the OpenPose Wrapper
     * class.
     */
    struct WrapperStructHand
    {
        /**
         * Whether to extract hand.
         */
        bool enable;

        /**
         * CCN (Conv Net) input size.
         * The greater, the slower and more memory it will be needed, but it will potentially increase accuracy.
         * Both width and height must be divisible by 16.
         */
        Point<int> netInputSize;

        /**
         * Whether to perform 1-time keypoint detection (fastest), iterative detection (recommended for images and fast videos, slowest method), or
         * tracking (recommended for video and webcam, in practice as fast as 1-time detection).
         */
        DetectionMode detectionMode;

        /**
         * Whether to render the output (pose locations, body, background or PAF heat maps) with CPU or GPU.
         * Select `None` for no rendering, `Cpu` or `Gpu` por CPU and GPU rendering respectively.
         */
        RenderMode renderMode;

        /**
         * Rendering blending alpha value of the pose point locations with respect to the background image.
         * Value in the range [0, 1]. 0 will only render the background, 1 will fully render the pose.
         */
        float alphaKeypoint;

        /**
         * Rendering blending alpha value of the heat maps (hand part, background or PAF) with respect to the background image.
         * Value in the range [0, 1]. 0 will only render the background, 1 will only render the heat map.
         */
        float alphaHeatMap;

        /**
         * Constructor of the struct.
         * It has the recommended and default values we recommend for each element of the struct.
         * Since all the elements of the struct are public, they can also be manually filled.
         */
        WrapperStructHand(const bool enable = false, const Point<int>& netInputSize = Point<int>{368, 368},
                          const DetectionMode detectionMode = DetectionMode::Fast,
                          const RenderMode renderMode = RenderMode::None,
                          const float alphaKeypoint = HAND_DEFAULT_ALPHA_KEYPOINT,
                          const float alphaHeatMap = HAND_DEFAULT_ALPHA_HEAT_MAP);
    };
}

#endif // OPENPOSE_WRAPPER_WRAPPER_STRUCT_HAND_HPP
