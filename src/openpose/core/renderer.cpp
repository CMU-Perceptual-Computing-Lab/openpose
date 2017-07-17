#ifndef CPU_ONLY
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#endif
#include <openpose/core/renderer.hpp>

namespace op
{
    Renderer::Renderer(const unsigned long long volume, const float alphaKeypoint, const float alphaHeatMap,
                       const unsigned int elementToRender, const unsigned int numberElementsToRender) :
        spGpuMemoryPtr{std::make_shared<float*>()},
        spElementToRender{std::make_shared<std::atomic<unsigned int>>(elementToRender)},
        spNumberElementsToRender{std::make_shared<const unsigned int>(numberElementsToRender)},
        mVolume{volume},
        mAlphaKeypoint{alphaKeypoint},
        mAlphaHeatMap{alphaHeatMap},
        mIsFirstRenderer{true},
        mIsLastRenderer{true},
        spGpuMemoryAllocated{std::make_shared<bool>(false)}
    {
    }

    Renderer::~Renderer()
    {
        try
        {
            #ifndef CPU_ONLY
                if (mIsLastRenderer)
                    cudaFree(*spGpuMemoryPtr);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void Renderer::initializationOnThread()
    {
        try
        {
            #ifndef CPU_ONLY
                if (mIsFirstRenderer)
                    cudaMalloc((void**)(spGpuMemoryPtr.get()), mVolume * sizeof(float));
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void Renderer::increaseElementToRender(const int increment)
    {
        try
        {
            auto elementToRender = (((int)(*spElementToRender) + increment) % (int)(*spNumberElementsToRender));
            // Handling negative increments
            while (elementToRender < 0)
                elementToRender += *spNumberElementsToRender;
            // Update final value
            *spElementToRender = elementToRender;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void Renderer::setElementToRender(const int elementToRender)
    {
        try
        {
            *spElementToRender = elementToRender % *spNumberElementsToRender;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::tuple<std::shared_ptr<float*>, std::shared_ptr<bool>, std::shared_ptr<std::atomic<unsigned int>>,
               std::shared_ptr<const unsigned int>> Renderer::getSharedParameters()
    {
        try
        {
            mIsLastRenderer = false;
            return std::make_tuple(spGpuMemoryPtr, spGpuMemoryAllocated, spElementToRender, spNumberElementsToRender);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_tuple(nullptr, nullptr, nullptr, nullptr);
        }
    }

    void Renderer::setSharedParametersAndIfLast(const std::tuple<std::shared_ptr<float*>, std::shared_ptr<bool>,
                                                                 std::shared_ptr<std::atomic<unsigned int>>,
                                                                 std::shared_ptr<const unsigned int>>& tuple, const bool isLast)
    {
        try
        {
            mIsFirstRenderer = false;
            mIsLastRenderer = isLast;
            spGpuMemoryPtr = std::get<0>(tuple);
            spGpuMemoryAllocated = std::get<1>(tuple);
            spElementToRender = std::get<2>(tuple);
            spNumberElementsToRender = std::get<3>(tuple);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    float Renderer::getAlphaKeypoint() const
    {
        try
        {
            return mAlphaKeypoint;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    void Renderer::setAlphaKeypoint(const float alphaKeypoint)
    {
        try
        {
            mAlphaKeypoint = alphaKeypoint;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    float Renderer::getAlphaHeatMap() const
    {
        try
        {
            return mAlphaHeatMap;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    void Renderer::setAlphaHeatMap(const float alphaHeatMap)
    {
        try
        {
            mAlphaHeatMap = alphaHeatMap;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void Renderer::cpuToGpuMemoryIfNotCopiedYet(const float* const cpuMemory)
    {
        try
        {
            #ifndef CPU_ONLY
                if (!*spGpuMemoryAllocated)
                {
                    cudaMemcpy(*spGpuMemoryPtr, cpuMemory, mVolume * sizeof(float), cudaMemcpyHostToDevice);
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

    void Renderer::gpuToCpuMemoryIfLastRenderer(float* cpuMemory)
    {
        try
        {
            #ifndef CPU_ONLY
                if (*spGpuMemoryAllocated && mIsLastRenderer)
                {
                    cudaMemcpy(cpuMemory, *spGpuMemoryPtr, mVolume * sizeof(float), cudaMemcpyDeviceToHost);
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
