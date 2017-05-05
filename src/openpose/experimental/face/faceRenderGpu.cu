#include "openpose/experimental/face/faceParameters.hpp"
#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/utilities/cuda.hpp"
#include "openpose/utilities/cuda.hu"
#include "openpose/utilities/render.hu"
#include "openpose/experimental/face/faceRenderGpu.hpp"

namespace op
{
    const dim3 THREADS_PER_BLOCK{128, 128, 1};
    __constant__ const unsigned char PART_PAIRS_GPU[] = FACE_PAIRS_TO_RENDER;
    __constant__ const float RGB_COLORS[] = {
        255.f,    255.f,    255.f,
    };



    __global__ void renderFaceParts(float* targetPtr, const int targetWidth, const int targetHeight, const float* const facePtr,
                                    const int numberFaces, const float threshold, const float alphaColorToAdd)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto globalIdx = threadIdx.y * blockDim.x + threadIdx.x;

        // Shared parameters
        __shared__ float2 sharedMins[FACE_MAX_NUMBER_FACE];
        __shared__ float2 sharedMaxs[FACE_MAX_NUMBER_FACE];
        __shared__ float sharedScaleF[FACE_MAX_NUMBER_FACE];

        // Other parameters
        const auto numberPartPairs = sizeof(PART_PAIRS_GPU) / (2*sizeof(PART_PAIRS_GPU[0]));
        const auto numberColors = sizeof(RGB_COLORS) / (3*sizeof(RGB_COLORS[0]));
        const auto radius = fastMin(targetWidth, targetHeight) / 120.f;
        const auto stickwidth = fastMin(targetWidth, targetHeight) / 250.f;

        // Render key points
        renderKeyPoints(targetPtr, sharedMaxs, sharedMins, sharedScaleF,
                        globalIdx, x, y, targetWidth, targetHeight, facePtr, PART_PAIRS_GPU, numberFaces,
                        FACE_NUMBER_PARTS, numberPartPairs, RGB_COLORS, numberColors,
                        radius, stickwidth, threshold, alphaColorToAdd);
    }

    void renderFaceGpu(float* framePtr, const cv::Size& frameSize, const float* const facePtr, const int numberFaces, const float alphaColorToAdd)
    {
        try
        {
            if (numberFaces > 0)
            {
                const auto threshold = 0.5f;
                const auto numBlocks = getNumberCudaBlocks(frameSize, THREADS_PER_BLOCK);
                renderFaceParts<<<THREADS_PER_BLOCK, numBlocks>>>(framePtr, frameSize.width, frameSize.height, facePtr, numberFaces, threshold, alphaColorToAdd);
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
