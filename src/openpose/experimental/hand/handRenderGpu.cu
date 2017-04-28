#include "openpose/experimental/hand/handParameters.hpp"
#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/utilities/cuda.hpp"
#include "openpose/utilities/cuda.hu"
#include "openpose/utilities/render.hu"
#include "openpose/experimental/hand/handRenderGpu.hpp"

namespace op
{
    const auto THREADS_PER_BLOCK_1D = 32;

    __constant__ const unsigned char PART_PAIRS_GPU[] = HAND_PAIRS_TO_RENDER;
    __constant__ const float RGB_COLORS[] = {
        179.f,    0.f,    0.f,
        204.f,    0.f,    0.f,
        230.f,    0.f,    0.f,
        255.f,    0.f,    0.f,
        143.f,  179.f,    0.f,
        163.f,  204.f,    0.f,
        184.f,  230.f,    0.f,
        204.f,  255.f,    0.f,
          0.f,  179.f,   71.f,
          0.f,  204.f,   82.f,
          0.f,  230.f,   92.f,
          0.f,  255.f,  102.f,
          0.f,   71.f,  179.f,
          0.f,   82.f,  204.f,
          0.f,   92.f,  230.f,
          0.f,  102.f,  255.f,
        143.f,    0.f,  179.f,
        163.f,    0.f,  204.f,
        184.f,    0.f,  230.f,
        204.f,    0.f,  255.f,
        179.f,  179.f,  179.f,
        179.f,  179.f,  179.f,
        179.f,  179.f,  179.f,
        179.f,  179.f,  179.f
    };



    __global__ void renderHandsParts(float* targetPtr, const int targetWidth, const int targetHeight, const float* const handsPtr,
                                     const int numberHands, const float threshold, const float alphaColorToAdd)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto globalIdx = threadIdx.y * blockDim.x + threadIdx.x;

        // Shared parameters
        __shared__ float2 sharedMins[HAND_MAX_NUMBER_HANDS];
        __shared__ float2 sharedMaxs[HAND_MAX_NUMBER_HANDS];
        __shared__ float sharedScaleF[HAND_MAX_NUMBER_HANDS];

        // Other parameters
        const auto numberPartPairs = sizeof(PART_PAIRS_GPU) / (2*sizeof(PART_PAIRS_GPU[0]));
        const auto numberColors = sizeof(RGB_COLORS) / (3*sizeof(RGB_COLORS[0]));
        const auto radius = fastMin(targetWidth, targetHeight) / 100.f;
        const auto stickwidth = fastMin(targetWidth, targetHeight) / 80.f;

        // Render key points
        renderKeyPoints(targetPtr, sharedMaxs, sharedMins, sharedScaleF,
                        globalIdx, x, y, targetWidth, targetHeight, handsPtr, PART_PAIRS_GPU, numberHands,
                        HAND_NUMBER_PARTS, numberPartPairs, RGB_COLORS, numberColors,
                        radius, stickwidth, threshold, alphaColorToAdd);
    }

    void renderHandsGpu(float* framePtr, const cv::Size& frameSize, const float* const handsPtr, const int numberHands, const float alphaColorToAdd)
    {
        try
        {
            if (numberHands > 0)
            {
                const auto threshold = 0.05f;
                dim3 threadsPerBlock = dim3{THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D};
                dim3 numBlocks = dim3{getNumberCudaBlocks(frameSize.width, threadsPerBlock.x), getNumberCudaBlocks(frameSize.height, threadsPerBlock.y)};
                renderHandsParts<<<threadsPerBlock, numBlocks>>>(framePtr, frameSize.width, frameSize.height, handsPtr, numberHands, threshold, alphaColorToAdd);
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
