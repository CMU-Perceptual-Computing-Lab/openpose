#include <openpose/face/faceParameters.hpp>
#include <openpose/gpu/cuda.hpp>
#include <openpose/gpu/cuda.hu>
#include <openpose/utilities/render.hu>
#include <openpose/face/renderFace.hpp>

namespace op
{
    __constant__ const unsigned int PART_PAIRS_GPU[] = {FACE_PAIRS_RENDER_GPU};
    __constant__ const float SCALES[] = {FACE_SCALES_RENDER_GPU};
    __constant__ const float COLORS[] = {FACE_COLORS_RENDER_GPU};

    __global__ void renderFaceParts(float* targetPtr, const int targetWidth, const int targetHeight,
                                    const float* const facePtr, const int numberPeople,
                                    const float threshold, const float alphaColorToAdd)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto globalIdx = threadIdx.y * blockDim.x + threadIdx.x;

        // Shared parameters
        __shared__ float2 sharedMins[FACE_MAX_FACES];
        __shared__ float2 sharedMaxs[FACE_MAX_FACES];
        __shared__ float sharedScaleF[FACE_MAX_FACES];

        // Other parameters
        const auto numberPartPairs = sizeof(PART_PAIRS_GPU) / (2*sizeof(PART_PAIRS_GPU[0]));
        const auto numberScales = sizeof(SCALES) / sizeof(SCALES[0]);
        const auto numberColors = sizeof(COLORS) / (3*sizeof(COLORS[0]));
        const auto radius = fastMin(targetWidth, targetHeight) / 120.f;
        const auto lineWidth = fastMin(targetWidth, targetHeight) / 250.f;

        // Render key points
        renderKeypoints(targetPtr, sharedMaxs, sharedMins, sharedScaleF, globalIdx, x, y, targetWidth, targetHeight,
                        facePtr, PART_PAIRS_GPU, numberPeople, FACE_NUMBER_PARTS, numberPartPairs, COLORS,
                        numberColors, radius, lineWidth, SCALES, numberScales, threshold, alphaColorToAdd);
    }

    void renderFaceKeypointsGpu(float* framePtr, const Point<int>& frameSize, const float* const facePtr,
                                const int numberPeople, const float renderThreshold, const float alphaColorToAdd)
    {
        try
        {
            if (numberPeople > 0)
            {
                dim3 threadsPerBlock;
                dim3 numBlocks;
                getNumberCudaThreadsAndBlocks(threadsPerBlock, numBlocks, frameSize);
                renderFaceParts<<<threadsPerBlock, numBlocks>>>(framePtr, frameSize.x, frameSize.y, facePtr,
                                                                numberPeople, renderThreshold, alphaColorToAdd);
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
