#ifndef CPU_ONLY
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#endif
#include <openpose/hand/handParameters.hpp>
#include <openpose/hand/renderHand.hpp>
#include <openpose/utilities/cuda.hpp>
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/hand/handRenderer.hpp>

namespace op
{
    HandRenderer::HandRenderer(const Point<int>& frameSize, const float alphaKeypoint, const float alphaHeatMap, const RenderMode renderMode) :
        Renderer{(unsigned long long)(frameSize.area() * 3), alphaKeypoint, alphaHeatMap},
        mFrameSize{frameSize},
        mRenderMode{renderMode}
    {
    }

    HandRenderer::~HandRenderer()
    {
        try
        {
            // Free CUDA pointers - Note that if pointers are 0 (i.e. nullptr), no operation is performed.
            #ifndef CPU_ONLY
                cudaFree(pGpuHand);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void HandRenderer::initializationOnThread()
    {
        try
        {
            log("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            Renderer::initializationOnThread();
            // GPU memory allocation for rendering
            #ifndef CPU_ONLY
                cudaMalloc((void**)(&pGpuHand), 2*HAND_NUMBER_PARTS * 3 * sizeof(float));
            #endif
            log("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void HandRenderer::renderHand(Array<float>& outputData, const Array<float>& handKeypoints)
    {
        try
        {
            // Security checks
            if (outputData.empty())
                error("Empty Array<float> outputData.", __LINE__, __FUNCTION__, __FILE__);

            // CPU rendering
            if (mRenderMode == RenderMode::Cpu)
                renderHandCpu(outputData, handKeypoints);

            // GPU rendering
            else
                renderHandGpu(outputData, handKeypoints);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void HandRenderer::renderHandCpu(Array<float>& outputData, const Array<float>& handKeypoints)
    {
        try
        {
            renderHandKeypointsCpu(outputData, handKeypoints);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void HandRenderer::renderHandGpu(Array<float>& outputData, const Array<float>& handKeypoints)
    {
        try
        {
            // GPU rendering
            #ifndef CPU_ONLY
                const auto elementRendered = spElementToRender->load(); // I prefer std::round(T&) over intRound(T) for std::atomic
                const auto numberPeople = handKeypoints.getSize(0);
                // GPU rendering
                if (numberPeople > 0 && elementRendered == 0)
                {
                    cpuToGpuMemoryIfNotCopiedYet(outputData.getPtr());
                    // Draw handKeypoints
                    cudaMemcpy(pGpuHand, handKeypoints.getConstPtr(), 2*HAND_NUMBER_PARTS*3 * sizeof(float), cudaMemcpyHostToDevice);
                    renderHandKeypointsGpu(*spGpuMemoryPtr, mFrameSize, pGpuHand, handKeypoints.getSize(0));
                    // CUDA check
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                }
                // GPU memory to CPU if last renderer
                gpuToCpuMemoryIfLastRenderer(outputData.getPtr());
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            // CPU_ONLY mode
            #else
                error("GPU rendering not available if `CPU_ONLY` is set.", __LINE__, __FUNCTION__, __FILE__);
                UNUSED(outputData);
                UNUSED(handKeypoints);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
