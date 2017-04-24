#ifndef OPENPOSE__POSE__GPU_POSE_RENDER_HPP
#define OPENPOSE__POSE__GPU_POSE_RENDER_HPP

#include <opencv2/core/core.hpp>
#include "enumClasses.hpp"
#include "poseParameters.hpp"

namespace op
{
    void renderPoseGpu(float* framePtr, const PoseModel poseModel, const int numberPeople, const cv::Size& frameSize, const float* const posePtr,
                       const bool googlyEyes = false, const float blendOriginalFrame = true, const float alphaBlending = POSE_DEFAULT_ALPHA_POSE);
    void renderBodyPartGpu(float* frame, const PoseModel poseModel, const cv::Size& frameSize, const float* const heatmap, const cv::Size& heatmapSize,
                           const float scaleToKeepRatio, const int part, const float alphaBlending = POSE_DEFAULT_ALPHA_HEATMAP);
    void renderBodyPartsGpu(float* frame, const PoseModel poseModel, const cv::Size& frameSize, const float* const heatmap, const cv::Size& heatmapSize,
                            const float scaleToKeepRatio, const float alphaBlending = POSE_DEFAULT_ALPHA_HEATMAP);
    void renderPartAffinityFieldGpu(float* framePtr, const PoseModel poseModel, const cv::Size& frameSize, const float* const heatmapPtr,
                                    const cv::Size& heatmapSize, const float scaleToKeepRatio, const int part, const float alphaBlending = POSE_DEFAULT_ALPHA_HEATMAP);
    void renderPartAffinityFieldsGpu(float* framePtr, const PoseModel poseModel, const cv::Size& frameSize, const float* const heatmapPtr,
                                     const cv::Size& heatmapSize, const float scaleToKeepRatio, const float alphaBlending = POSE_DEFAULT_ALPHA_HEATMAP);
}

#endif // OPENPOSE__POSE__GPU_POSE_RENDER_HPP
