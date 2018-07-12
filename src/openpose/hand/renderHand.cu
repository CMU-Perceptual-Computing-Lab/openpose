#include <openpose/gpu/cuda.hpp>
#include <openpose/gpu/cuda.hu>
#include <openpose/hand/handParameters.hpp>
#include <openpose/utilities/render.hu>
#include <openpose/hand/renderHand.hpp>

namespace op
{
    __constant__ const unsigned int PART_PAIRS_GPU[] = {HAND_PAIRS_RENDER_GPU};
    __constant__ const float SCALES[] = {HAND_SCALES_RENDER_GPU};
    __constant__ const float COLORS[] = {HAND_COLORS_RENDER_GPU};

    __global__ void renderHandsParts(float* targetPtr, const int targetWidth, const int targetHeight,
                                     const float* const handsPtr, const int numberHands,
                                     const float threshold, const float alphaColorToAdd)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto globalIdx = threadIdx.y * blockDim.x + threadIdx.x;

        // Shared parameters
        __shared__ float2 sharedMins[HAND_MAX_HANDS];
        __shared__ float2 sharedMaxs[HAND_MAX_HANDS];
        __shared__ float sharedScaleF[HAND_MAX_HANDS];

        // Other parameters
        const auto numberPartPairs = sizeof(PART_PAIRS_GPU) / (2*sizeof(PART_PAIRS_GPU[0]));
        const auto numberScales = sizeof(SCALES) / sizeof(SCALES[0]);
        const auto numberColors = sizeof(COLORS) / (3*sizeof(COLORS[0]));
        const auto radius = fastMin(targetWidth, targetHeight) / 100.f;
        const auto lineWidth = fastMin(targetWidth, targetHeight) / 80.f;

        // Render key points
        renderKeypoints(targetPtr, sharedMaxs, sharedMins, sharedScaleF, globalIdx, x, y, targetWidth, targetHeight,
                        handsPtr, PART_PAIRS_GPU, numberHands, HAND_NUMBER_PARTS, numberPartPairs, COLORS,
                        numberColors, radius, lineWidth, SCALES, numberScales, threshold, alphaColorToAdd);
    }

    void renderHandKeypointsGpu(float* framePtr, const Point<int>& frameSize, const float* const handsPtr,
                                const int numberHands, const float renderThreshold, const float alphaColorToAdd)
    {
        try
        {
            if (numberHands > 0)
            {
                dim3 threadsPerBlock;
                dim3 numBlocks;
                getNumberCudaThreadsAndBlocks(threadsPerBlock, numBlocks, frameSize);
                renderHandsParts<<<threadsPerBlock, numBlocks>>>(framePtr, frameSize.x, frameSize.y, handsPtr,
                                                                 numberHands, renderThreshold, alphaColorToAdd);
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
