#include <openpose/gpu/cuda.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/pose/bodyPartConnectorBase.hpp>
#include <iostream>

namespace op
{
    template<typename T>
    inline __device__ int intRoundGPU(const T a)
    {
        return int(a+0.5f);
    }

    template <typename T>
    inline __device__  T process(const T* bodyPartA, const T* bodyPartB, const T* mapX, const T* mapY, const int heatmapWidth, const int heatmapHeight, const float interThreshold = 0.05, const float interMinAboveThreshold = 0.95, const float renderThreshold = 0.05)
    {
        T finalOutput = -1;
        if(bodyPartA[2] < renderThreshold || bodyPartB[2] < renderThreshold) return finalOutput;

        auto vectorAToBX = bodyPartB[0] - bodyPartA[0];
        auto vectorAToBY = bodyPartB[1] - bodyPartA[1];
        auto vectorAToBMax = max(abs(vectorAToBX), abs(vectorAToBY));
        auto numberPointsInLine = max(5, min(25, intRoundGPU(sqrt(5*vectorAToBMax))));
        auto vectorNorm = T(sqrt(vectorAToBX*vectorAToBX + vectorAToBY*vectorAToBY));

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

            // L2 Hack
            //int l2Dist = (int)sqrt(pow(vectorAToBX,2) + pow(vectorAToBY,2));
            //if(l2Dist <= 2) count = numberPointsInLine;

            // parts score + connection score
            if (count/(float)numberPointsInLine > interMinAboveThreshold)
                finalOutput = sum/count;
        }

        return finalOutput;
    }

    template <typename T>
    __global__ void bpcKernel(T* finalOutputPtr, const T* heatMapPtr, const T* peaksPtrA, const T* peaksPtrB, const unsigned int* bodyPartPairsPtr, const unsigned int* mapIdxPtr, const int POSE_MAX_PEOPLE, const int TOTAL_BODY_PARTS, const int TOTAL_BODY_PAIRS, const int heatmapWidth, const int heatmapHeight)
    {
        const auto i = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto j = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto k = (blockIdx.z * blockDim.z) + threadIdx.z;

        if(i >= TOTAL_BODY_PAIRS) return;

        int partA = bodyPartPairsPtr[i*2];
        int partB = bodyPartPairsPtr[i*2 + 1];
        int mapIdxX = mapIdxPtr[i*2];
        int mapIdxY = mapIdxPtr[i*2 + 1];

        const T* bodyPartA = peaksPtrA + (partA*POSE_MAX_PEOPLE*3 + j*3);
        const T* bodyPartB = peaksPtrB + (partB*POSE_MAX_PEOPLE*3 + k*3);
        const T* mapX = heatMapPtr + mapIdxX*heatmapWidth*heatmapHeight;
        const T* mapY = heatMapPtr + mapIdxY*heatmapWidth*heatmapHeight;

        T finalOutput = process(bodyPartA, bodyPartB, mapX, mapY, heatmapWidth, heatmapHeight);
        finalOutputPtr[i*POSE_MAX_PEOPLE*POSE_MAX_PEOPLE + j*POSE_MAX_PEOPLE + k] = finalOutput;
    }

    template <typename T>
    void connectBodyPartsGpu(Array<T>& poseKeypoints, Array<T>& poseScores, const T* const heatMapPtr,
                             const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
                             const int maxPeaks, const T interMinAboveThreshold, const T interThreshold,
                             const int minSubsetCnt, const T minSubsetScore, const T scaleFactor,
                             const T* const heatMapGpuPtr, const T* const peaksGpuPtr)
    {
        try
        {
            // Parts Connection
            const auto& bodyPartPairs = getPosePartPairs(poseModel);
            const auto& mapIdx_old = getPoseMapIndex(poseModel);
            const auto numberBodyParts = getPoseNumberBodyParts(poseModel);
            const auto numberBodyPartPairs = bodyPartPairs.size() / 2;
            const auto subsetCounterIndex = numberBodyParts;
            auto mapIdx(mapIdx_old);
            for(auto& i : mapIdx) i+=(numberBodyParts+1);

            if (numberBodyParts == 0)
                error("Invalid value of numberBodyParts, it must be positive, not " + std::to_string(numberBodyParts),
                      __LINE__, __FUNCTION__, __FILE__);

            // Upload required data to GPU
            unsigned int* bodyPartPairsGpuPtr;
            cudaMallocHost((void **)&bodyPartPairsGpuPtr, bodyPartPairs.size() * sizeof(unsigned int));
            cudaMemcpy(bodyPartPairsGpuPtr, &bodyPartPairs[0], bodyPartPairs.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
            unsigned int* mapIdxGpuPtr;
            cudaMallocHost((void **)&mapIdxGpuPtr, mapIdx.size() * sizeof(unsigned int));
            cudaMemcpy(mapIdxGpuPtr, &mapIdx[0], mapIdx.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
            T* finalOutputGpuPtr;
            op::Array<T> finalOutputCpu;
            finalOutputCpu.reset({(int)numberBodyPartPairs, (int)POSE_MAX_PEOPLE, (int)POSE_MAX_PEOPLE},-1);
            int totalComputations = numberBodyPartPairs * POSE_MAX_PEOPLE * POSE_MAX_PEOPLE;
            cudaMallocHost((void **)&finalOutputGpuPtr, totalComputations * sizeof(float));

            // Run Kernel
            const dim3 threadsPerBlock{4, 6, 6}; //4 is good for BODY_25, 8 for COCO?
            if(POSE_MAX_PEOPLE % threadsPerBlock.y || POSE_MAX_PEOPLE % threadsPerBlock.z)
                error("Invalid value of POSE_MAX_PEOPLE, it must be multiple of 16" + std::to_string(POSE_MAX_PEOPLE),
                      __LINE__, __FUNCTION__, __FILE__);
            int pairBlocks = std::round((numberBodyPartPairs/threadsPerBlock.x) + 0.5);
            const dim3 numBlocks{pairBlocks, POSE_MAX_PEOPLE / threadsPerBlock.y, POSE_MAX_PEOPLE / threadsPerBlock.z};
            bpcKernel<<<numBlocks, threadsPerBlock>>>(finalOutputGpuPtr, heatMapGpuPtr, peaksGpuPtr, peaksGpuPtr, bodyPartPairsGpuPtr, mapIdxGpuPtr, POSE_MAX_PEOPLE, numberBodyParts, numberBodyPartPairs, heatMapSize.x, heatMapSize.y);
            cudaMemcpy(finalOutputCpu.getPtr(), finalOutputGpuPtr, totalComputations * sizeof(float),cudaMemcpyDeviceToHost);

            // std::vector<std::pair<std::vector<int>, double>> refers to:
            //     - std::vector<int>: [body parts locations, #body parts found]
            //     - double: subset score
            const auto subsets = generateInitialSubsets(
                heatMapPtr, peaksPtr, poseModel, heatMapSize, maxPeaks, interThreshold, interMinAboveThreshold,
                bodyPartPairs, numberBodyParts, numberBodyPartPairs, subsetCounterIndex, finalOutputCpu);

            // Delete people below the following thresholds:
                // a) minSubsetCnt: removed if less than minSubsetCnt body parts
                // b) minSubsetScore: removed if global score smaller than this
                // c) POSE_MAX_PEOPLE: keep first POSE_MAX_PEOPLE people above thresholds
            int numberPeople;
            std::vector<int> validSubsetIndexes;
            validSubsetIndexes.reserve(fastMin((size_t)POSE_MAX_PEOPLE, subsets.size()));
            removeSubsetsBelowThresholds(validSubsetIndexes, numberPeople, subsets, subsetCounterIndex,
                                         numberBodyParts, minSubsetCnt, minSubsetScore, maxPeaks);

            // Fill and return poseKeypoints
            subsetsToPoseKeypointsAndScores(poseKeypoints, poseScores, scaleFactor, subsets, validSubsetIndexes,
                                            peaksPtr, numberPeople, numberBodyParts, numberBodyPartPairs);

            // Differences w.r.t. CPU version for now
            UNUSED(heatMapGpuPtr);
            UNUSED(peaksGpuPtr);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void connectBodyPartsGpu(Array<float>& poseKeypoints, Array<float>& poseScores,
                                      const float* const heatMapPtr, const float* const peaksPtr,
                                      const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks,
                                      const float interMinAboveThreshold, const float interThreshold,
                                      const int minSubsetCnt, const float minSubsetScore, const float scaleFactor,
                                      const float* const heatMapGpuPtr, const float* const peaksGpuPtr);
    template void connectBodyPartsGpu(Array<double>& poseKeypoints, Array<double>& poseScores,
                                      const double* const heatMapPtr, const double* const peaksPtr,
                                      const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks,
                                      const double interMinAboveThreshold, const double interThreshold,
                                      const int minSubsetCnt, const double minSubsetScore, const double scaleFactor,
                                      const double* const heatMapGpuPtr, const double* const peaksGpuPtr);
}
