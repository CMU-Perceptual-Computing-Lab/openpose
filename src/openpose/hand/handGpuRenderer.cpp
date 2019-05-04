#ifdef USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#endif
#include <openpose/gpu/cuda.hpp>
#include <openpose/hand/renderHand.hpp>
#include <openpose/hand/handGpuRenderer.hpp>

namespace op
{
    HandGpuRenderer::HandGpuRenderer(const float renderThreshold, const float alphaKeypoint,
                                     const float alphaHeatMap) :
        GpuRenderer{renderThreshold, alphaKeypoint, alphaHeatMap},
        pGpuHand{nullptr},
        pMaxPtr{nullptr},
        pMinPtr{nullptr},
        pScalePtr{nullptr}
    {
    }

    HandGpuRenderer::~HandGpuRenderer()
    {
        try
        {
            // Free CUDA pointers - Note that if pointers are 0 (i.e., nullptr), no operation is performed.
            #ifdef USE_CUDA
                if (pGpuHand != nullptr)
                    cudaFree(pGpuHand);
                if (pMaxPtr != nullptr)
                    cudaFree(pMaxPtr);
                if (pMinPtr != nullptr)
                    cudaFree(pMinPtr);
                if (pScalePtr != nullptr)
                    cudaFree(pScalePtr);
            #endif
        }
        catch (const std::exception& e)
        {
            errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void HandGpuRenderer::initializationOnThread()
    {
        try
        {
            log("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // GPU memory allocation for rendering
            #ifdef USE_CUDA
                cudaMalloc((void**)(&pGpuHand), HAND_MAX_HANDS * HAND_NUMBER_PARTS * 3 * sizeof(float));
                cudaMalloc((void**)&pMaxPtr, sizeof(float) * 2 * HAND_MAX_HANDS);
                cudaMalloc((void**)&pMinPtr, sizeof(float) * 2 * HAND_MAX_HANDS);
                cudaMalloc((void**)&pScalePtr, sizeof(float) * HAND_MAX_HANDS);
            #endif
            log("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void HandGpuRenderer::renderHandInherited(
        Array<float>& outputData, const std::array<Array<float>, 2>& handKeypoints)
    {
        try
        {
            // GPU rendering
            #ifdef USE_CUDA
                // I prefer std::round(T&) over positiveIntRound(T) for std::atomic
                const auto elementRendered = spElementToRender->load();
                const auto numberPeople = handKeypoints[0].getSize(0);
                const Point<int> frameSize{outputData.getSize(1), outputData.getSize(0)};
                // GPU rendering
                if (numberPeople > 0 && elementRendered == 0)
                {
                    // Draw handKeypoints
                    cpuToGpuMemoryIfNotCopiedYet(outputData.getPtr(), outputData.getVolume());
                    const auto handArea = handKeypoints[0].getSize(1)*handKeypoints[0].getSize(2);
                    const auto handVolume = numberPeople * handArea;
                    cudaMemcpy(pGpuHand, handKeypoints[0].getConstPtr(), handVolume * sizeof(float),
                               cudaMemcpyHostToDevice);
                    cudaMemcpy(pGpuHand + handVolume, handKeypoints[1].getConstPtr(),
                               handVolume * sizeof(float), cudaMemcpyHostToDevice);
                    renderHandKeypointsGpu(
                        *spGpuMemory, pMaxPtr, pMinPtr, pScalePtr, frameSize, pGpuHand, 2 * numberPeople,
                        mRenderThreshold, getAlphaKeypoint());
                    // CUDA check
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                }
                // GPU memory to CPU if last renderer
                gpuToCpuMemoryIfLastRenderer(outputData.getPtr(), outputData.getVolume());
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #else
                UNUSED(outputData);
                UNUSED(handKeypoints);
                error("OpenPose must be compiled with the `USE_CUDA` macro definitions in order to run this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
