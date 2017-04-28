#ifndef OPENPOSE__POSE__BODY_PARTS_CONNECTOR_HPP
#define OPENPOSE__POSE__BODY_PARTS_CONNECTOR_HPP

#include "../core/array.hpp"
#include "enumClasses.hpp"

namespace op
{
    template <typename T>
    void connectBodyPartsCpu(Array<T>& poseKeyPoints, const T* const heatMapPtr, const T* const peaksPtr, const PoseModel poseModel, const cv::Size& heatMapSize, const int maxPeaks,
                             const int interMinAboveThreshold, const T interThreshold, const int minSubsetCnt, const T minSubsetScore, const T scaleFactor = 1.f);

    template <typename T>
    void connectBodyPartsGpu(Array<T>& poseKeyPoints, T* posePtr, const T* const heatMapPtr, const T* const peaksPtr, const PoseModel poseModel, const cv::Size& heatMapSize,
                             const int maxPeaks, const int interMinAboveThreshold, const T interThreshold, const int minSubsetCnt, const T minSubsetScore, const T scaleFactor = 1.f);
}

#endif // OPENPOSE__POSE__BODY_PARTS_CONNECTOR_HPP
