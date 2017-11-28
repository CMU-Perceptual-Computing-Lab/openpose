#ifndef OPENPOSE_POSE_BODY_PARTS_CONNECTOR_HPP
#define OPENPOSE_POSE_BODY_PARTS_CONNECTOR_HPP

#include <openpose/core/common.hpp>
#include <openpose/pose/enumClasses.hpp>

namespace op
{
    template <typename T>
    OP_API void connectBodyPartsCpu(Array<T>& poseKeypoints, Array<T>& poseScores, const T* const heatMapPtr,
                                    const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
                                    const int maxPeaks, const T interMinAboveThreshold, const T interThreshold,
                                    const int minSubsetCnt, const T minSubsetScore, const T scaleFactor = 1.f);

    template <typename T>
    OP_API void connectBodyPartsGpu(Array<T>& poseKeypoints, Array<T>& poseScores, const T* const heatMapPtr,
                                    const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
                                    const int maxPeaks, const T interMinAboveThreshold, const T interThreshold,
                                    const int minSubsetCnt, const T minSubsetScore, const T scaleFactor = 1.f,
                                    const T* const heatMapGpuPtr = nullptr, const T* const peaksGpuPtr = nullptr);
}

#endif // OPENPOSE_POSE_BODY_PARTS_CONNECTOR_HPP
