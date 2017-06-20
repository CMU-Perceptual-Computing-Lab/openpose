#ifndef OPENPOSE_POSE_RENDER_POSE_HPP
#define OPENPOSE_POSE_RENDER_POSE_HPP

#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/array.hpp>
#include <openpose/core/point.hpp>
#include "enumClasses.hpp"
#include "poseParameters.hpp"

namespace op
{
    void renderPoseKeypointsCpu(Array<float>& frameArray, const Array<float>& poseKeypoints, const PoseModel poseModel,
                                const bool blendOriginalFrame = true);

    void renderPoseKeypointsGpu(float* framePtr, const PoseModel poseModel, const int numberPeople, const Point<int>& frameSize,
                                const float* const posePtr, const bool googlyEyes = false, const bool blendOriginalFrame = true,
                                const float alphaBlending = POSE_DEFAULT_ALPHA_KEYPOINT);

    void renderPoseHeatMapGpu(float* frame, const PoseModel poseModel, const Point<int>& frameSize, const float* const heatmap,
                              const Point<int>& heatmapSize, const float scaleToKeepRatio, const int part,
                              const float alphaBlending = POSE_DEFAULT_ALPHA_HEAT_MAP);

    void renderPoseHeatMapsGpu(float* frame, const PoseModel poseModel, const Point<int>& frameSize, const float* const heatmap,
                               const Point<int>& heatmapSize, const float scaleToKeepRatio,
                               const float alphaBlending = POSE_DEFAULT_ALPHA_HEAT_MAP);

    void renderPosePAFGpu(float* framePtr, const PoseModel poseModel, const Point<int>& frameSize, const float* const heatmapPtr,
                          const Point<int>& heatmapSize, const float scaleToKeepRatio, const int part,
                          const float alphaBlending = POSE_DEFAULT_ALPHA_HEAT_MAP);

    void renderPosePAFsGpu(float* framePtr, const PoseModel poseModel, const Point<int>& frameSize, const float* const heatmapPtr,
                           const Point<int>& heatmapSize, const float scaleToKeepRatio,
                           const float alphaBlending = POSE_DEFAULT_ALPHA_HEAT_MAP);
}

#endif // OPENPOSE_POSE_RENDER_POSE_HPP
