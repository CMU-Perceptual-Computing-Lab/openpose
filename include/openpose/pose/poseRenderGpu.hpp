#ifndef OPENPOSE_POSE_GPU_POSE_RENDER_HPP
#define OPENPOSE_POSE_GPU_POSE_RENDER_HPP

#include <openpose/core/point.hpp>
#include "enumClasses.hpp"
#include "poseParameters.hpp"

namespace op
{
    void renderPoseGpu(float* framePtr, const PoseModel poseModel, const int numberPeople, const Point<int>& frameSize, const float* const posePtr,
                       const bool googlyEyes = false, const bool blendOriginalFrame = true, const float alphaBlending = POSE_DEFAULT_ALPHA_KEYPOINT);
    void renderBodyPartGpu(float* frame, const PoseModel poseModel, const Point<int>& frameSize, const float* const heatmap, const Point<int>& heatmapSize,
                           const float scaleToKeepRatio, const int part, const float alphaBlending = POSE_DEFAULT_ALPHA_HEAT_MAP);
    void renderBodyPartsGpu(float* frame, const PoseModel poseModel, const Point<int>& frameSize, const float* const heatmap, const Point<int>& heatmapSize,
                            const float scaleToKeepRatio, const float alphaBlending = POSE_DEFAULT_ALPHA_HEAT_MAP);
    void renderPartAffinityFieldGpu(float* framePtr, const PoseModel poseModel, const Point<int>& frameSize, const float* const heatmapPtr,
                                    const Point<int>& heatmapSize, const float scaleToKeepRatio, const int part, const float alphaBlending = POSE_DEFAULT_ALPHA_HEAT_MAP);
    void renderPartAffinityFieldsGpu(float* framePtr, const PoseModel poseModel, const Point<int>& frameSize, const float* const heatmapPtr,
                                     const Point<int>& heatmapSize, const float scaleToKeepRatio, const float alphaBlending = POSE_DEFAULT_ALPHA_HEAT_MAP);
}

#endif // OPENPOSE_POSE_GPU_POSE_RENDER_HPP
