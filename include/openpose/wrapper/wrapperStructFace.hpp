#ifndef OPENPOSE_WRAPPER_WRAPPER_STRUCT_FACE_HPP
#define OPENPOSE_WRAPPER_WRAPPER_STRUCT_FACE_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/enumClasses.hpp>
#include <openpose/face/faceParameters.hpp>

namespace op
{
    /**
     * WrapperStructFace: Face estimation and rendering configuration struct.
     * WrapperStructFace allows the user to set up the face estimation and rendering parameters that will be used for
     * the OpenPose WrapperT template and Wrapper class.
     */
    struct OP_API WrapperStructFace
    {
        /**
         * Whether to extract face.
         */
        bool enable;

        /**
         * CCN (Conv Net) input size.
         * The greater, the slower and more memory it will be needed, but it will potentially increase accuracy.
         * Both width and height must be divisible by 16.
         */
        Point<int> netInputSize;

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
         * Rendering blending alpha value of the heat maps (face part, background or PAF) with respect to the
         * background image.
         * Value in the range [0, 1]. 0 will only render the background, 1 will only render the heat map.
         */
        float alphaHeatMap;

        /**
         * Rendering threshold. Only estimated keypoints whose score confidences are higher than this value will be
         * rendered. Generally, a high threshold (> 0.5) will only render very clear body parts; while small thresholds
         * (~0.1) will also output guessed and occluded keypoints, but also more false positives (i.e., wrong
         * detections).
         */
        float renderThreshold;

        /**
         * Constructor of the struct.
         * It has the recommended and default values we recommend for each element of the struct.
         * Since all the elements of the struct are public, they can also be manually filled.
         */
        WrapperStructFace(
            const bool enable = false, const Point<int>& netInputSize = Point<int>{368, 368},
            const RenderMode renderMode = RenderMode::Gpu, const float alphaKeypoint = FACE_DEFAULT_ALPHA_KEYPOINT,
            const float alphaHeatMap = FACE_DEFAULT_ALPHA_HEAT_MAP, const float renderThreshold = 0.4f);
    };
}

#endif // OPENPOSE_WRAPPER_WRAPPER_STRUCT_FACE_HPP
