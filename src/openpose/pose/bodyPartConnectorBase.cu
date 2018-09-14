#include <openpose/gpu/cuda.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/pose/bodyPartConnectorBase.hpp>
#include <iostream>

namespace op
{
    const dim3 THREADS_PER_BLOCK{4, 16, 16};

    template<typename T>
    inline __device__ int intRoundGPU(const T a)
    {
        return int(a+T(0.5));
    }

    template <typename T>
    inline __device__  T process(const T* bodyPartA, const T* bodyPartB, const T* mapX, const T* mapY,
                                 const int heatmapWidth, const int heatmapHeight, const T interThreshold,
                                 const T interMinAboveThreshold)
    {
        const auto vectorAToBX = bodyPartB[0] - bodyPartA[0];
        const auto vectorAToBY = bodyPartB[1] - bodyPartA[1];
        const auto vectorAToBMax = max(abs(vectorAToBX), abs(vectorAToBY));
        const auto numberPointsInLine = max(5, min(25, intRoundGPU(sqrt(5*vectorAToBMax))));
        const auto vectorNorm = T(sqrt(vectorAToBX*vectorAToBX + vectorAToBY*vectorAToBY));

        if (vectorNorm > 1e-6)
        {
            const auto sX = bodyPartA[0];
            const auto sY = bodyPartA[1];
            const auto vectorAToBNormX = vectorAToBX/vectorNorm;
            const auto vectorAToBNormY = vectorAToBY/vectorNorm;

            auto sum = 0.;
            auto count = 0;
            const auto vectorAToBXInLine = vectorAToBX/numberPointsInLine;
            const auto vectorAToBYInLine = vectorAToBY/numberPointsInLine;
            for (auto lm = 0; lm < numberPointsInLine; lm++)
            {
                const auto mX = min(heatmapWidth-1, intRoundGPU(sX + lm*vectorAToBXInLine));
                const auto mY = min(heatmapHeight-1, intRoundGPU(sY + lm*vectorAToBYInLine));
                const auto idx = mY * heatmapWidth + mX;
                const auto score = (vectorAToBNormX*mapX[idx] + vectorAToBNormY*mapY[idx]);
                if (score > interThreshold)
                {
                    sum += score;
                    count++;
                }
            }

            // // L2 Hack
            // int l2Dist = (int)sqrt(pow(vectorAToBX,2) + pow(vectorAToBY,2));
            // if (l2Dist <= 2)
            //     count = numberPointsInLine;

            // parts score + connection score
            if (count/(float)numberPointsInLine > interMinAboveThreshold)
                return sum/count;
        }
        return -1;
    }

    template <typename T>
    __global__ void pafScoreKernel(T* finalOutputPtr, const T* const heatMapPtr, const T* const peaksPtr,
                                   const unsigned int* const bodyPartPairsPtr, const unsigned int* const mapIdxPtr,
                                   const unsigned int maxPeaks, const int numberBodyPartPairs,
                                   const int heatmapWidth, const int heatmapHeight, const T interThreshold,
                                   const T interMinAboveThreshold)
    {
        const auto pairIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto peakA = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto peakB = (blockIdx.z * blockDim.z) + threadIdx.z;

        if (pairIndex < numberBodyPartPairs && peakA < maxPeaks && peakB < maxPeaks)
        {
            const auto baseIndex = 2*pairIndex;
            const auto partA = bodyPartPairsPtr[baseIndex];
            const auto partB = bodyPartPairsPtr[baseIndex + 1];

            const T numberPeaksA = peaksPtr[3*partA*(maxPeaks+1)];
            const T numberPeaksB = peaksPtr[3*partB*(maxPeaks+1)];

            const auto outputIndex = (pairIndex*maxPeaks+peakA)*maxPeaks + peakB;
            if (peakA < numberPeaksA && peakB < numberPeaksB)
            {
                const auto mapIdxX = mapIdxPtr[baseIndex];
                const auto mapIdxY = mapIdxPtr[baseIndex + 1];

                const T* const bodyPartA = peaksPtr + (3*(partA*(maxPeaks+1) + peakA+1));
                const T* const bodyPartB = peaksPtr + (3*(partB*(maxPeaks+1) + peakB+1));
                const T* const mapX = heatMapPtr + mapIdxX*heatmapWidth*heatmapHeight;
                const T* const mapY = heatMapPtr + mapIdxY*heatmapWidth*heatmapHeight;
                finalOutputPtr[outputIndex] = process(
                    bodyPartA, bodyPartB, mapX, mapY, heatmapWidth, heatmapHeight, interThreshold,
                    interMinAboveThreshold);
            }
            else
                finalOutputPtr[outputIndex] = -1;
        }
    }

    template <typename T>
    void connectBodyPartsGpu(Array<T>& poseKeypoints, Array<T>& poseScores, const T* const heatMapGpuPtr,
                             const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
                             const int maxPeaks, const T interMinAboveThreshold, const T interThreshold,
                             const int minSubsetCnt, const T minSubsetScore, const T scaleFactor,
                             Array<T> finalOutputCpu, T* finalOutputGpuPtr,
                             const unsigned int* const bodyPartPairsGpuPtr, const unsigned int* const mapIdxGpuPtr,
                             const T* const peaksGpuPtr)
    {
        try
        {
            // Parts Connection
            const auto& bodyPartPairs = getPosePartPairs(poseModel);
            const auto numberBodyParts = getPoseNumberBodyParts(poseModel);
            const auto numberBodyPartPairs = bodyPartPairs.size() / 2;
            const auto subsetCounterIndex = numberBodyParts;
            const auto totalComputations = finalOutputCpu.getVolume();

            if (numberBodyParts == 0)
                error("Invalid value of numberBodyParts, it must be positive, not " + std::to_string(numberBodyParts),
                      __LINE__, __FUNCTION__, __FILE__);
            if (bodyPartPairsGpuPtr == nullptr || mapIdxGpuPtr == nullptr)
                error("The pointers bodyPartPairsGpuPtr and mapIdxGpuPtr cannot be nullptr.",
                      __LINE__, __FUNCTION__, __FILE__);


            // Run Kernel - finalOutputGpu
            const dim3 numBlocks{
                getNumberCudaBlocks(numberBodyPartPairs, THREADS_PER_BLOCK.x),
                getNumberCudaBlocks(maxPeaks, THREADS_PER_BLOCK.y),
                getNumberCudaBlocks(maxPeaks, THREADS_PER_BLOCK.z)};
            pafScoreKernel<<<numBlocks, THREADS_PER_BLOCK>>>(
                finalOutputGpuPtr, heatMapGpuPtr, peaksGpuPtr, bodyPartPairsGpuPtr, mapIdxGpuPtr,
                maxPeaks, numberBodyPartPairs, heatMapSize.x, heatMapSize.y, interThreshold,
                interMinAboveThreshold);
            // finalOutputCpu <-- finalOutputGpu
            cudaMemcpy(finalOutputCpu.getPtr(), finalOutputGpuPtr, totalComputations * sizeof(float),
                       cudaMemcpyDeviceToHost);

            // std::vector<std::pair<std::vector<int>, double>> refers to:
            //     - std::vector<int>: [body parts locations, #body parts found]
            //     - double: subset score
            const T* const tNullptr = nullptr;
            const auto subsets = generateInitialSubsets(
                tNullptr, peaksPtr, poseModel, heatMapSize, maxPeaks, interThreshold, interMinAboveThreshold,
                bodyPartPairs, numberBodyParts, numberBodyPartPairs, subsetCounterIndex, finalOutputCpu);

            // Delete people below the following thresholds:
                // a) minSubsetCnt: removed if less than minSubsetCnt body parts
                // b) minSubsetScore: removed if global score smaller than this
                // c) maxPeaks (POSE_MAX_PEOPLE): keep first maxPeaks people above thresholds
            int numberPeople;
            std::vector<int> validSubsetIndexes;
            validSubsetIndexes.reserve(fastMin((size_t)maxPeaks, subsets.size()));
            removeSubsetsBelowThresholds(validSubsetIndexes, numberPeople, subsets, subsetCounterIndex,
                                         numberBodyParts, minSubsetCnt, minSubsetScore, maxPeaks);

            // Fill and return poseKeypoints
            subsetsToPoseKeypointsAndScores(poseKeypoints, poseScores, scaleFactor, subsets, validSubsetIndexes,
                                            peaksPtr, numberPeople, numberBodyParts, numberBodyPartPairs);

            // Sanity check
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void connectBodyPartsGpu(Array<float>& poseKeypoints, Array<float>& poseScores,
                                      const float* const heatMapGpuPtr, const float* const peaksPtr,
                                      const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks,
                                      const float interMinAboveThreshold, const float interThreshold,
                                      const int minSubsetCnt, const float minSubsetScore, const float scaleFactor,
                                      Array<float> finalOutputCpu, float* finalOutputGpuPtr,
                                      const unsigned int* const bodyPartPairsGpuPtr,
                                      const unsigned int* const mapIdxGpuPtr,
                                      const float* const peaksGpuPtr);
    template void connectBodyPartsGpu(Array<double>& poseKeypoints, Array<double>& poseScores,
                                      const double* const heatMapGpuPtr, const double* const peaksPtr,
                                      const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks,
                                      const double interMinAboveThreshold, const double interThreshold,
                                      const int minSubsetCnt, const double minSubsetScore, const double scaleFactor,
                                      Array<double> finalOutputCpu, double* finalOutputGpuPtr,
                                      const unsigned int* const bodyPartPairsGpuPtr,
                                      const unsigned int* const mapIdxGpuPtr,
                                      const double* const peaksGpuPtr);
}
