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
    OP_API void connectBodyPartsGpu(Array<T>& poseKeypoints, Array<T>& poseScores, const T* const heatMapGpuPtr,
                                    const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
                                    const int maxPeaks, const T interMinAboveThreshold, const T interThreshold,
                                    const int minSubsetCnt, const T minSubsetScore, const T scaleFactor = 1.f,
                                    Array<T> pairScoresCpu = Array<T>{}, T* pairScoresGpuPtr = nullptr,
                                    const unsigned int* const bodyPartPairsGpuPtr = nullptr,
                                    const unsigned int* const mapIdxGpuPtr = nullptr,
                                    const T* const peaksGpuPtr = nullptr);

    // Private functions used by the 2 above functions
    template <typename T>
    OP_API std::vector<std::pair<std::vector<int>, T>> createPeopleVector(
        const T* const heatMapPtr, const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
        const int maxPeaks, const T interThreshold, const T interMinAboveThreshold,
        const std::vector<unsigned int>& bodyPartPairs, const unsigned int numberBodyParts,
        const unsigned int numberBodyPartPairs, const Array<T>& precomputedPAFs = Array<T>());

    template <typename T>
    OP_API void removePeopleBelowThresholds(std::vector<int>& validSubsetIndexes, int& numberPeople,
                                            const std::vector<std::pair<std::vector<int>, T>>& subsets,
                                            const unsigned int numberBodyParts, const int minSubsetCnt,
                                            const T minSubsetScore, const int maxPeaks);

    template <typename T>
    OP_API void peopleVectorToPeopleArray(Array<T>& poseKeypoints, Array<T>& poseScores, const T scaleFactor,
                                          const std::vector<std::pair<std::vector<int>, T>>& subsets,
                                          const std::vector<int>& validSubsetIndexes, const T* const peaksPtr,
                                          const int numberPeople, const unsigned int numberBodyParts,
                                          const unsigned int numberBodyPartPairs);

    template <typename T>
    OP_API std::vector<std::tuple<T, T, int, int, int>> pafPtrIntoVector(
        const Array<T>& pairScores, const T* const peaksPtr, const int maxPeaks,
        const std::vector<unsigned int>& bodyPartPairs, const unsigned int numberBodyPartPairs);

    template <typename T>
    OP_API std::vector<std::pair<std::vector<int>, T>> pafVectorIntoPeopleVector(
        const std::vector<std::tuple<T, T, int, int, int>>& pairScores, const T* const peaksPtr, const int maxPeaks,
        const std::vector<unsigned int>& bodyPartPairs, const unsigned int numberBodyParts);
}

#endif // OPENPOSE_POSE_BODY_PARTS_CONNECTOR_HPP
