#ifndef OPENPOSE_WRAPPER_WRAPPER_STRUCT_HAND_HPP
#define OPENPOSE_WRAPPER_WRAPPER_STRUCT_HAND_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/enumClasses.hpp>
#include <openpose/hand/handParameters.hpp>

namespace op
{
    /**
     * WrapperStructHand: Hand estimation and rendering configuration struct.
     * WrapperStructHand allows the user to set up the hand estimation and rendering parameters that will be used for the OpenPose Wrapper
     * class.
     */
    struct OP_API WrapperStructHand
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
         * Number of scales to process.
         * The greater, the slower and more memory it will be needed, but it will potentially increase accuracy.
         * This parameter is related with scaleRange, such as the final pose estimation will be an average of the predicted results for each scale.
         */
        int scalesNumber;

        /**
         * Total range between smallest and biggest scale. The scales will be centered in ratio 1. E.g. if scaleRange = 0.4 and
         * scalesNumber = 2, then there will be 2 scales, 0.8 and 1.2.
         */
        float scaleRange;

        /**
         * Whether to add tracking between frames. Adding hand tracking might improve hand keypoints detection for webcam (if the frame rate
         * is high enough, i.e. >7 FPS per GPU) and video. This is not person ID tracking, it simply looks for hands in positions at which hands
         * were located in previous frames, but it does not guarantee the same person id among frames.
         */
        bool tracking;

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
         * Rendering threshold. Only estimated keypoints whose score confidences are higher than this value will be rendered. Generally, a
         * high threshold (> 0.5) will only render very clear body parts; while small thresholds (~0.1) will also output guessed and occluded
         * keypoints, but also more false positives (i.e. wrong detections).
         */
        float renderThreshold;

        /**
         * Constructor of the struct.
         * It has the recommended and default values we recommend for each element of the struct.
         * Since all the elements of the struct are public, they can also be manually filled.
         */
        WrapperStructHand(const bool enable = false, const Point<int>& netInputSize = Point<int>{368, 368},
                          const int scalesNumber = 1, const float scaleRange = 0.4f,
                          const bool tracking = false, const RenderMode renderMode = RenderMode::None,
                          const float alphaKeypoint = HAND_DEFAULT_ALPHA_KEYPOINT,
                          const float alphaHeatMap = HAND_DEFAULT_ALPHA_HEAT_MAP,
                          const float renderThreshold = 0.2f);
    };
}

#endif // OPENPOSE_WRAPPER_WRAPPER_STRUCT_HAND_HPP
