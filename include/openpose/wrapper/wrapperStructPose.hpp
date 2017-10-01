#ifndef OPENPOSE_WRAPPER_WRAPPER_STRUCT_POSE_HPP
#define OPENPOSE_WRAPPER_WRAPPER_STRUCT_POSE_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/enumClasses.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/pose/poseParameters.hpp>

namespace op
{
    /**
     * WrapperStructPose: Pose estimation and rendering configuration struct.
     * WrapperStructPose allows the user to set up the pose estimation and rendering parameters that will be used for the OpenPose Wrapper
     * class.
     */
    struct OP_API WrapperStructPose
    {
        /**
         * CCN (Conv Net) input size.
         * The greater, the slower and more memory it will be needed, but it will potentially increase accuracy.
         * Both width and height must be divisible by 16.
         */
        Point<int> netInputSize;

        /**
         * Output size of the final rendered image.
         * It barely affects performance compared to netInputSize.
         * The final Datum.poseKeypoints can be scaled with respect to outputSize if `keypointScale` is set to ScaleMode::OutputResolution, even if the
         * rendering is disabled.
         */
        Point<int> outputSize;

        /**
         * Final scale of the Array<float> Datum.poseKeypoints and the writen pose data.
         * The final Datum.poseKeypoints can be scaled with respect to input size (ScaleMode::InputResolution), net output size (ScaleMode::NetOutputResolution),
         * output rendering size (ScaleMode::OutputResolution), from 0 to 1 (ScaleMode::ZeroToOne), and -1 to 1 (ScaleMode::PlusMinusOne).
         */
        ScaleMode keypointScale;

        /**
         * Number of GPUs processing in parallel.
         * The greater, the faster the algorithm will run, but potentially higher lag will appear (which only affects in real-time webcam scenarios).
         */
        int gpuNumber;

        /**
         * First GPU device.
         * Such as the GPUs used will be the ones in the range: [gpuNumberStart, gpuNumberStart + gpuNumber].
         */
        int gpuNumberStart;

        /**
         * Number of scales to process.
         * The greater, the slower and more memory it will be needed, but it will potentially increase accuracy.
         * This parameter is related with scaleGap, such as the final pose estimation will be an average of the predicted results for each scale.
         */
        int scalesNumber;

        /**
         * Gap between successive scales.
         * The pose estimation will be estimation for the scales in the range [1, 1-scaleGap*scalesNumber], with a gap of scaleGap.
         */
        float scaleGap;

        /**
         * Whether to render the output (pose locations, body, background or PAF heat maps) with CPU or GPU.
         * Select `None` for no rendering, `Cpu` or `Gpu` por CPU and GPU rendering respectively.
         */
        RenderMode renderMode;

        /**
         * Pose model, it affects the number of body parts to render
         * Select PoseModel::COCO_18 for 18 body-part COCO, PoseModel::MPI_15 for 15 body-part MPI, PoseModel::MPI_15_4 for faster version
         * of MPI, etc.).
         */
        PoseModel poseModel;

        /**
         * Whether to blend the final results on top of the original image, or just render them on a flat background.
         */
        bool blendOriginalFrame;

        /**
         * Rendering blending alpha value of the pose point locations with respect to the background image.
         * Value in the range [0, 1]. 0 will only render the background, 1 will fully render the pose.
         */
        float alphaKeypoint;

        /**
         * Rendering blending alpha value of the heat maps (body part, background or PAF) with respect to the background image.
         * Value in the range [0, 1]. 0 will only render the background, 1 will only render the heat map.
         */
        float alphaHeatMap;

        /**
         * Element to initially render.
         * Set 0 for pose, [1, #body parts] for each body part following the order on POSE_BODY_PART_MAPPING on
         * `include/pose/poseParameters.hpp`, #body parts+1 for background, #body parts+2 for all body parts overlapped,
         * #body parts+3 for all PAFs, and [#body parts+4, #body parts+4+#pair pairs] for each PAF following the order on POSE_BODY_PART_PAIRS.
         */
        int defaultPartToRender;

        /**
         * Folder where the pose Caffe models are located.
         */
        std::string modelFolder;

        /**
         * Whether and which heat maps to save on the Array<float> Datum.heatmaps.
         * Use HeatMapType::Parts for body parts, HeatMapType::Background for the background, and HeatMapType::PAFs for the Part Affinity Fields.
         */
        std::vector<HeatMapType> heatMapTypes;

        /**
         * Scale of the Datum.heatmaps.
         * Select ScaleMode::ZeroToOne for range [0,1], ScaleMode::PlusMinusOne for [-1,1] and ScaleMode::UnsignedChar for [0, 255]
         * If heatMapTypes.empty(), then this parameters makes no effect.
         */
        ScaleMode heatMapScale;

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
        WrapperStructPose(const Point<int>& netInputSize = Point<int>{656, 368},
                          const Point<int>& outputSize = Point<int>{1280, 720},
                          const ScaleMode keypointScale = ScaleMode::InputResolution,
                          const int gpuNumber = -1, const int gpuNumberStart = 0, const int scalesNumber = 1,
                          const float scaleGap = 0.15f, const RenderMode renderMode = RenderMode::None,
                          const PoseModel poseModel = PoseModel::COCO_18, const bool blendOriginalFrame = true,
                          const float alphaKeypoint = POSE_DEFAULT_ALPHA_KEYPOINT,
                          const float alphaHeatMap = POSE_DEFAULT_ALPHA_HEAT_MAP,
                          const int defaultPartToRender = 0, const std::string& modelFolder = "models/",
                          const std::vector<HeatMapType>& heatMapTypes = {},
                          const ScaleMode heatMapScale = ScaleMode::ZeroToOne,
                          const float renderThreshold = 0.05f);
    };
}

#endif // OPENPOSE_WRAPPER_WRAPPER_STRUCT_POSE_HPP
