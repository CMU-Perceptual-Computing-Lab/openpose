#ifndef OPENPOSE_POSE_BODY_PARTS_CONNECTOR_HPP
#define OPENPOSE_POSE_BODY_PARTS_CONNECTOR_HPP

#include <openpose/core/common.hpp>
#include <openpose/pose/enumClasses.hpp>

namespace op
{
    template <typename T>
    OP_API void connectBodyPartsCpu(Array<T>& poseKeypoints, const T* const heatMapPtr, const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks,
                                    const int interMinAboveThreshold, const T interThreshold, const int minSubsetCnt, const T minSubsetScore, const T scaleFactor = 1.f);

    template <typename T>
    OP_API void connectBodyPartsGpu(Array<T>& poseKeypoints, T* posePtr, const T* const heatMapPtr, const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
                                    const int maxPeaks, const int interMinAboveThreshold, const T interThreshold, const int minSubsetCnt, const T minSubsetScore, const T scaleFactor = 1.f);
}

#endif // OPENPOSE_POSE_BODY_PARTS_CONNECTOR_HPP
