#ifndef OPENPOSE_POSE_RENDER_POSE_HPP
#define OPENPOSE_POSE_RENDER_POSE_HPP

#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/common.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/pose/poseParametersRender.hpp>

namespace op
{
    OP_API void renderPoseKeypointsCpu(Array<float>& frameArray, const Array<float>& poseKeypoints,
                                       const PoseModel poseModel, const float renderThreshold,
                                       const bool blendOriginalFrame = true);

    void renderPoseKeypointsGpu(float* framePtr, const PoseModel poseModel, const int numberPeople,
                                const Point<int>& frameSize, const float* const posePtr,
                                const float renderThreshold, const bool googlyEyes = false,
                                const bool blendOriginalFrame = true,
                                const float alphaBlending = POSE_DEFAULT_ALPHA_KEYPOINT);

    void renderPoseHeatMapGpu(float* frame, const Point<int>& frameSize, const float* const heatMapPtr,
                              const Point<int>& heatMapSize, const float scaleToKeepRatio,
                              const unsigned int part,
                              const float alphaBlending = POSE_DEFAULT_ALPHA_HEAT_MAP);

    void renderPoseHeatMapsGpu(float* frame, const PoseModel poseModel, const Point<int>& frameSize,
                               const float* const heatMapPtr, const Point<int>& heatMapSize,
                               const float scaleToKeepRatio,
                               const float alphaBlending = POSE_DEFAULT_ALPHA_HEAT_MAP);

    void renderPosePAFGpu(float* framePtr, const PoseModel poseModel, const Point<int>& frameSize,
                          const float* const heatMapPtr, const Point<int>& heatMapSize,
                          const float scaleToKeepRatio, const int part,
                          const float alphaBlending = POSE_DEFAULT_ALPHA_HEAT_MAP);

    void renderPosePAFsGpu(float* framePtr, const PoseModel poseModel, const Point<int>& frameSize,
                           const float* const heatMapPtr, const Point<int>& heatMapSize,
                           const float scaleToKeepRatio,
                           const float alphaBlending = POSE_DEFAULT_ALPHA_HEAT_MAP);

    void renderPoseDistanceGpu(float* framePtr, const Point<int>& frameSize, const float* const heatMapPtr,
                               const Point<int>& heatMapSize, const float scaleToKeepRatio,
                               const unsigned int part, const float alphaBlending = POSE_DEFAULT_ALPHA_HEAT_MAP);
}

#endif // OPENPOSE_POSE_RENDER_POSE_HPP
