#include <openpose/utilities/cuda.hpp>
#include <openpose/pose/bodyPartConnectorBase.hpp>

namespace op
{
    template <typename T>
    void connectBodyPartsGpu(Array<T>& poseKeypoints, T* posePtr, const T* const heatMapPtr, const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
                             const int maxPeaks, const int interMinAboveThreshold, const T interThreshold, const int minSubsetCnt, const T minSubsetScore, const T scaleFactor)
    {
        try
        {
            UNUSED(poseKeypoints);
            UNUSED(posePtr);
            UNUSED(heatMapPtr);
            UNUSED(peaksPtr);
            UNUSED(poseModel);
            UNUSED(heatMapSize);
            UNUSED(maxPeaks);
            UNUSED(interMinAboveThreshold);
            UNUSED(interThreshold);
            UNUSED(minSubsetCnt);
            UNUSED(minSubsetScore);
            UNUSED(scaleFactor);
            error("GPU version not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void connectBodyPartsGpu(Array<float>& poseKeypoints, float* posePtr, const float* const heatMapPtr, const float* const peaksPtr, const PoseModel poseModel,
                                      const Point<int>& heatMapSize, const int maxPeaks, const int interMinAboveThreshold, const float interThreshold, const int minSubsetCnt,
                                      const float minSubsetScore, const float scaleFactor);
    template void connectBodyPartsGpu(Array<double>& poseKeypoints, double* posePtr, const double* const heatMapPtr, const double* const peaksPtr, const PoseModel poseModel,
                                      const Point<int>& heatMapSize, const int maxPeaks, const int interMinAboveThreshold, const double interThreshold, const int minSubsetCnt,
                                      const double minSubsetScore, const double scaleFactor);
}
