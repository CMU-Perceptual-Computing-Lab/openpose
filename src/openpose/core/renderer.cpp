#include <cuda.h>
#include <cuda_runtime_api.h>
#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/core/renderer.hpp"

namespace op
{
    Renderer::Renderer(const unsigned long long volume) :
        spGpuMemoryPtr{std::make_shared<float*>()},
        mVolume{volume},
        mIsFirstRenderer{true},
        mIsLastRenderer{true},
        spGpuMemoryAllocated{std::make_shared<bool>(false)}
    {
    }

    Renderer::~Renderer()
    {
        try
        {
            if (mIsLastRenderer)
                cudaFree(*spGpuMemoryPtr);
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
            if (mIsFirstRenderer)
                cudaMalloc((void**)(spGpuMemoryPtr.get()), mVolume * sizeof(float));
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::pair<std::shared_ptr<float*>, std::shared_ptr<bool>> Renderer::getGpuMemoryAndSetAsFirst()
    {
        try
        {
            mIsLastRenderer = false;
            return std::make_pair(spGpuMemoryPtr, spGpuMemoryAllocated);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_pair(nullptr, nullptr);
        }
    }

    void Renderer::setGpuMemoryAndSetIfLast(const std::pair<std::shared_ptr<float*>, std::shared_ptr<bool>>& gpuMemory, const bool isLast)
    {
        try
        {
            mIsFirstRenderer = false;
            mIsLastRenderer = isLast;
            spGpuMemoryPtr = gpuMemory.first;
            spGpuMemoryAllocated = gpuMemory.second;
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
            if (!*spGpuMemoryAllocated)
            {
                cudaMemcpy(*spGpuMemoryPtr, cpuMemory, mVolume * sizeof(float), cudaMemcpyHostToDevice);
                *spGpuMemoryAllocated = true;
            }
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
            if (*spGpuMemoryAllocated && mIsLastRenderer)
            {
                cudaMemcpy(cpuMemory, *spGpuMemoryPtr, mVolume * sizeof(float), cudaMemcpyDeviceToHost);
                *spGpuMemoryAllocated = false;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
