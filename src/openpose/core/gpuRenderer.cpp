#ifndef CPU_ONLY
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#endif
#include <openpose/core/gpuRenderer.hpp>

namespace op
{
    void checkAndIncreaseGpuMemory(std::shared_ptr<float*>& gpuMemoryPtr,
                                   std::shared_ptr<std::atomic<unsigned long long>>& currentVolumePtr,
                                   const unsigned long long memoryVolume)
    {
        try
        {
            #ifndef CPU_ONLY
                if (*currentVolumePtr < memoryVolume)
                {
                    *currentVolumePtr = memoryVolume;
                    cudaFree(*gpuMemoryPtr);
                    cudaMalloc((void**)(gpuMemoryPtr.get()), *currentVolumePtr * sizeof(float));
                }
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    GpuRenderer::GpuRenderer(const float renderThreshold, const float alphaKeypoint,
                             const float alphaHeatMap, const bool blendOriginalFrame,
                             const unsigned int elementToRender, const unsigned int numberElementsToRender) :
        Renderer{renderThreshold, alphaKeypoint, alphaHeatMap, blendOriginalFrame, elementToRender,
                 numberElementsToRender},
        spGpuMemory{std::make_shared<float*>()},
        spVolume{std::make_shared<std::atomic<unsigned long long>>(0)},
        mIsFirstRenderer{true},
        mIsLastRenderer{true},
        spGpuMemoryAllocated{std::make_shared<bool>(false)}
    {
    }

    GpuRenderer::~GpuRenderer()
    {
        try
        {
            #ifndef CPU_ONLY
                if (mIsLastRenderer)
                    cudaFree(*spGpuMemory);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::tuple<std::shared_ptr<float*>, std::shared_ptr<bool>, std::shared_ptr<std::atomic<unsigned int>>,
               std::shared_ptr<std::atomic<unsigned long long>>, std::shared_ptr<const unsigned int>>
               GpuRenderer::getSharedParameters()
    {
        try
        {
            mIsLastRenderer = false;
            return std::make_tuple(spGpuMemory, spGpuMemoryAllocated, spElementToRender, spVolume, spNumberElementsToRender);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_tuple(nullptr, nullptr, nullptr, nullptr, nullptr);
        }
    }

    void GpuRenderer::setSharedParametersAndIfLast(const std::tuple<std::shared_ptr<float*>, std::shared_ptr<bool>,
                                                                    std::shared_ptr<std::atomic<unsigned int>>,
                                                                    std::shared_ptr<std::atomic<unsigned long long>>,
                                                                    std::shared_ptr<const unsigned int>>& tuple,
                                                   const bool isLast)
    {
        try
        {
            mIsFirstRenderer = false;
            mIsLastRenderer = isLast;
            spGpuMemory = std::get<0>(tuple);
            spGpuMemoryAllocated = std::get<1>(tuple);
            spElementToRender = std::get<2>(tuple);
            spVolume = std::get<3>(tuple);
            spNumberElementsToRender = std::get<4>(tuple);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void GpuRenderer::cpuToGpuMemoryIfNotCopiedYet(const float* const cpuMemory, const unsigned long long memoryVolume)
    {
        try
        {
            #ifndef CPU_ONLY
                if (!*spGpuMemoryAllocated)
                {
                    checkAndIncreaseGpuMemory(spGpuMemory, spVolume, memoryVolume);
                    cudaMemcpy(*spGpuMemory, cpuMemory, memoryVolume * sizeof(float), cudaMemcpyHostToDevice);
                    *spGpuMemoryAllocated = true;
                }
            #else
                error("GPU rendering not available if `CPU_ONLY` is set.", __LINE__, __FUNCTION__, __FILE__);
                UNUSED(cpuMemory);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void GpuRenderer::gpuToCpuMemoryIfLastRenderer(float* cpuMemory, const unsigned long long memoryVolume)
    {
        try
        {
            #ifndef CPU_ONLY
                if (*spGpuMemoryAllocated && mIsLastRenderer)
                {
                    if (*spVolume < memoryVolume)
                        error("CPU is asking for more memory than it was copied into GPU.",
                              __LINE__, __FUNCTION__, __FILE__);
                    cudaMemcpy(cpuMemory, *spGpuMemory, memoryVolume * sizeof(float), cudaMemcpyDeviceToHost);
                    *spGpuMemoryAllocated = false;
                }
            #else
                error("GPU rendering not available if `CPU_ONLY` is set.", __LINE__, __FUNCTION__, __FILE__);
                UNUSED(cpuMemory);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
