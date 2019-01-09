#ifdef USE_OPENCL
    #include <openpose/gpu/opencl.hcl>
    #include <openpose/gpu/cl2.hpp>
#endif
#include <openpose/gpu/cuda.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/net/bodyPartConnectorBase.hpp>
#include <iostream>

namespace op
{
    #ifdef USE_OPENCL
        typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, unsigned int, int, int, int, float, float> PAFScoreKernelFunctor;
        const std::string pafScoreKernel = MULTI_LINE_STRING(

            int intRoundGPU(const Type a)
            {
                return (int)(a+0.5);
            }

            Type process(__global const Type* bodyPartA, __global const Type* bodyPartB, __global const Type* mapX, __global const Type* mapY,
                                         const int heatmapWidth, const int heatmapHeight, const Type interThreshold,
                                         const Type interMinAboveThreshold)
            {
                const Type vectorAToBX = bodyPartB[0] - bodyPartA[0];
                const Type vectorAToBY = bodyPartB[1] - bodyPartA[1];
                const Type vectorAToBMax = max(fabs(vectorAToBX), fabs(vectorAToBY));
                const int numberPointsInLine = max(5, min(25, intRoundGPU(sqrt(5*vectorAToBMax))));
                const Type vectorNorm = (Type)(sqrt(vectorAToBX*vectorAToBX + vectorAToBY*vectorAToBY));
                Type rval = -1;

                if (vectorNorm > 1e-6)
                {
                    const Type sX = bodyPartA[0];
                    const Type sY = bodyPartA[1];
                    const Type vectorAToBNormX = vectorAToBX/vectorNorm;
                    const Type vectorAToBNormY = vectorAToBY/vectorNorm;

                    Type sum = (Type)(0.);
                    int count = 0;
                    const Type vectorAToBXInLine = vectorAToBX/numberPointsInLine;
                    const Type vectorAToBYInLine = vectorAToBY/numberPointsInLine;
                    for (int lm = 0; lm < numberPointsInLine; lm++)
                    {
                        const int mX = min(heatmapWidth-1, intRoundGPU(sX + lm*vectorAToBXInLine));
                        const int mY = min(heatmapHeight-1, intRoundGPU(sY + lm*vectorAToBYInLine));
                        const int idx = mY * heatmapWidth + mX;
                        const Type score = (vectorAToBNormX*mapX[idx] + vectorAToBNormY*mapY[idx]);
                        if (score > interThreshold)
                        {
                            sum += score;
                            count++;
                        }
                    }

                    // Return PAF score
                    if (count/(Type)(numberPointsInLine) > interMinAboveThreshold)
                        return (Type)(sum)/(Type)(count);
                    else
                    {
                        // Ideally, if distanceAB = 0, PAF is 0 between A and B, provoking a false negative
                        // To fix it, we consider PAF-connected keypoints very close to have a minimum PAF score, such that:
                        //     1. It will consider very close keypoints (where the PAF is 0)
                        //     2. But it will not automatically connect them (case PAF score = 1), or real PAF might got
                        //        missing
                        const Type l2Dist = sqrt((Type)(vectorAToBX*vectorAToBX + vectorAToBY*vectorAToBY));
                        const Type threshold = sqrt((Type)(heatmapWidth*heatmapHeight))/150; // 3.3 for 368x656, 6.6 for 2x resolution
                        if (l2Dist < threshold)
                            return 0.15;
                    }
                }
                return -1;
            }

            __kernel void pafScoreKernel(__global Type* pairScoresPtr, __global const Type* const heatMapPtr, __global const Type* const peaksPtr,
                                         __global const unsigned int* const bodyPartPairsPtr, __global const unsigned int* const mapIdxPtr,
                                         const unsigned int maxPeaks, const int numberBodyPartPairs,
                                         const int heatmapWidth, const int heatmapHeight, const Type interThreshold,
                                         const Type interMinAboveThreshold)
            {
                int pairIndex = get_global_id(0);
                int peakA = get_global_id(1);
                int peakB = get_global_id(2);

                if (pairIndex < numberBodyPartPairs && peakA < maxPeaks && peakB < maxPeaks)
                {
                    int baseIndex = 2*pairIndex;
                    int partA = bodyPartPairsPtr[baseIndex];
                    int partB = bodyPartPairsPtr[baseIndex + 1];

                    const Type numberPeaksA = peaksPtr[3*partA*(maxPeaks+1)];
                    const Type numberPeaksB = peaksPtr[3*partB*(maxPeaks+1)];

                    const int outputIndex = (pairIndex*maxPeaks+peakA)*maxPeaks + peakB;
                    if (peakA < numberPeaksA && peakB < numberPeaksB)
                    {
                        const int mapIdxX = mapIdxPtr[baseIndex];
                        const int mapIdxY = mapIdxPtr[baseIndex + 1];

                        __global const Type* bodyPartA = peaksPtr + (3*(partA*(maxPeaks+1) + peakA+1));
                        __global const Type* bodyPartB = peaksPtr + (3*(partB*(maxPeaks+1) + peakB+1));
                        __global const Type* mapX = heatMapPtr + mapIdxX*heatmapWidth*heatmapHeight;
                        __global const Type* mapY = heatMapPtr + mapIdxY*heatmapWidth*heatmapHeight;

                        pairScoresPtr[outputIndex] = process(
                            bodyPartA, bodyPartB, mapX, mapY, heatmapWidth, heatmapHeight, interThreshold,
                            interMinAboveThreshold);
                    }
                    else
                        pairScoresPtr[outputIndex] = -1;
                }

            }
        );
    #endif

    template <typename T>
    void connectBodyPartsOcl(Array<T>& poseKeypoints, Array<T>& poseScores, const T* const heatMapGpuPtr,
                             const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
                             const int maxPeaks, const T interMinAboveThreshold, const T interThreshold,
                             const int minSubsetCnt, const T minSubsetScore, const T scaleFactor,
                             const bool maximizePositives, Array<T> pairScoresCpu, T* pairScoresGpuPtr,
                             const unsigned int* const bodyPartPairsGpuPtr, const unsigned int* const mapIdxGpuPtr,
                             const T* const peaksGpuPtr, const int gpuID)
    {
        try
        {
            #ifdef USE_OPENCL
                // Parts Connection
                const auto& bodyPartPairs = getPosePartPairs(poseModel);
                const auto numberBodyParts = getPoseNumberBodyParts(poseModel);
                const auto numberBodyPartPairs = (unsigned int)(bodyPartPairs.size() / 2);
                const auto totalComputations = pairScoresCpu.getVolume();

                if (numberBodyParts == 0)
                    error("Invalid value of numberBodyParts, it must be positive, not " + std::to_string(numberBodyParts),
                          __LINE__, __FUNCTION__, __FILE__);
                if (bodyPartPairsGpuPtr == nullptr || mapIdxGpuPtr == nullptr)
                    error("The pointers bodyPartPairsGpuPtr and mapIdxGpuPtr cannot be nullptr.",
                          __LINE__, __FUNCTION__, __FILE__);

                auto pafScoreKernel = OpenCL::getInstance(gpuID)->getKernelFunctorFromManager
                        <PAFScoreKernelFunctor, T>(
                         "pafScoreKernel", op::pafScoreKernel);

                cl::Buffer pairScoresGpuPtrBuffer = cl::Buffer((cl_mem)(pairScoresGpuPtr), true);
                cl::Buffer heatMapGpuPtrBuffer = cl::Buffer((cl_mem)(heatMapGpuPtr), true);
                cl::Buffer peaksGpuPtrBuffer = cl::Buffer((cl_mem)(peaksGpuPtr), true);
                cl::Buffer bodyPartPairsGpuPtrBuffer = cl::Buffer((cl_mem)(bodyPartPairsGpuPtr), true);
                cl::Buffer mapIdxGpuPtrBuffer = cl::Buffer((cl_mem)(mapIdxGpuPtr), true);

                // PAF Kernel Runtime
                pafScoreKernel(
                    cl::EnqueueArgs(OpenCL::getInstance(gpuID)->getQueue(), cl::NDRange(numberBodyPartPairs,maxPeaks,maxPeaks)),
                    pairScoresGpuPtrBuffer, heatMapGpuPtrBuffer, peaksGpuPtrBuffer, bodyPartPairsGpuPtrBuffer, mapIdxGpuPtrBuffer,
                    maxPeaks, (int)numberBodyPartPairs, heatMapSize.x, heatMapSize.y, interThreshold,
                    interMinAboveThreshold);
                OpenCL::getInstance(gpuID)->getQueue().enqueueReadBuffer(pairScoresGpuPtrBuffer, CL_TRUE, 0,
                                                                          totalComputations * sizeof(T), pairScoresCpu.getPtr());

                // New code
                // Get pair connections and their scores
                const auto pairConnections = pafPtrIntoVector(
                    pairScoresCpu, peaksPtr, maxPeaks, bodyPartPairs, numberBodyPartPairs);
                const auto peopleVector = pafVectorIntoPeopleVector(
                    pairConnections, peaksPtr, maxPeaks, bodyPartPairs, numberBodyParts);

               // // Old code
               // // Get pair connections and their scores
               // // std::vector<std::pair<std::vector<int>, double>> refers to:
               // //     - std::vector<int>: [body parts locations, #body parts found]
               // //     - double: person subset score
               // const T* const tNullptr = nullptr;
               // const auto peopleVector = createPeopleVector(
               //     tNullptr, peaksPtr, poseModel, heatMapSize, maxPeaks, interThreshold, interMinAboveThreshold,
               //     bodyPartPairs, numberBodyParts, numberBodyPartPairs, pairScoresCpu);

                // Delete people below the following thresholds:
                    // a) minSubsetCnt: removed if less than minSubsetCnt body parts
                    // b) minSubsetScore: removed if global score smaller than this
                    // c) maxPeaks (POSE_MAX_PEOPLE): keep first maxPeaks people above thresholds
                int numberPeople;
                std::vector<int> validSubsetIndexes;
                validSubsetIndexes.reserve(fastMin((size_t)maxPeaks, peopleVector.size()));
                removePeopleBelowThresholds(validSubsetIndexes, numberPeople, peopleVector, numberBodyParts, minSubsetCnt,
                                            minSubsetScore, maxPeaks, maximizePositives);

                // Fill and return poseKeypoints
                peopleVectorToPeopleArray(poseKeypoints, poseScores, scaleFactor, peopleVector, validSubsetIndexes,
                                          peaksPtr, numberPeople, numberBodyParts, numberBodyPartPairs);

               // // Sanity check
               // cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #else
                UNUSED(poseKeypoints);
                UNUSED(poseScores);
                UNUSED(heatMapGpuPtr);
                UNUSED(peaksPtr);
                UNUSED(poseModel);
                UNUSED(heatMapSize);
                UNUSED(maxPeaks);
                UNUSED(interMinAboveThreshold);
                UNUSED(interThreshold);
                UNUSED(minSubsetCnt);
                UNUSED(minSubsetScore);
                UNUSED(scaleFactor);
                UNUSED(maximizePositives);
                UNUSED(pairScoresCpu);
                UNUSED(pairScoresGpuPtr);
                UNUSED(bodyPartPairsGpuPtr);
                UNUSED(mapIdxGpuPtr);
                UNUSED(peaksGpuPtr);
                UNUSED(gpuID);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void connectBodyPartsOcl(
        Array<float>& poseKeypoints, Array<float>& poseScores, const float* const heatMapGpuPtr,
        const float* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks,
        const float interMinAboveThreshold, const float interThreshold, const int minSubsetCnt,
        const float minSubsetScore, const float scaleFactor, const bool maximizePositives,
        Array<float> pairScoresCpu, float* pairScoresGpuPtr, const unsigned int* const bodyPartPairsGpuPtr,
        const unsigned int* const mapIdxGpuPtr, const float* const peaksGpuPtr, const int gpuID);
    template void connectBodyPartsOcl(
        Array<double>& poseKeypoints, Array<double>& poseScores, const double* const heatMapGpuPtr,
        const double* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks,
        const double interMinAboveThreshold, const double interThreshold, const int minSubsetCnt,
        const double minSubsetScore, const double scaleFactor, const bool maximizePositives,
        Array<double> pairScoresCpu, double* pairScoresGpuPtr, const unsigned int* const bodyPartPairsGpuPtr,
        const unsigned int* const mapIdxGpuPtr, const double* const peaksGpuPtr, const int gpuID);
}
