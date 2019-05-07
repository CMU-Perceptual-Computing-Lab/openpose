#ifndef OPENPOSE_CORE_GPU_RENDERER_HPP
#define OPENPOSE_CORE_GPU_RENDERER_HPP

#include <atomic>
#include <tuple>
#include <openpose/core/common.hpp>
#include <openpose/core/renderer.hpp>

namespace op
{
    class OP_API GpuRenderer : public Renderer
    {
    public:
        explicit GpuRenderer(
            const float renderThreshold, const float alphaKeypoint, const float alphaHeatMap,
            const bool blendOriginalFrame = true, const unsigned int elementToRender = 0u,
            const unsigned int numberElementsToRender = 0u);

        virtual ~GpuRenderer();

        std::tuple<std::shared_ptr<float*>, std::shared_ptr<bool>, std::shared_ptr<std::atomic<unsigned int>>,
                   std::shared_ptr<unsigned long long>, std::shared_ptr<const unsigned int>>
                   getSharedParameters();

        void setSharedParametersAndIfLast(
            const std::tuple<std::shared_ptr<float*>, std::shared_ptr<bool>, std::shared_ptr<std::atomic<unsigned int>>,
                std::shared_ptr<unsigned long long>, std::shared_ptr<const unsigned int>>& tuple,
            const bool isLast);

        void setSharedParameters(
            const std::tuple<std::shared_ptr<float*>, std::shared_ptr<bool>,
                std::shared_ptr<unsigned long long>>& tuple);

    protected:
        std::shared_ptr<float*> spGpuMemory;

        void cpuToGpuMemoryIfNotCopiedYet(const float* const cpuMemory, const unsigned long long memoryVolume);

        void gpuToCpuMemoryIfLastRenderer(float* cpuMemory, const unsigned long long memoryVolume);

    private:
        std::shared_ptr<unsigned long long> spVolume;
        bool mIsFirstRenderer;
        bool mIsLastRenderer;
        std::shared_ptr<bool> spGpuMemoryAllocated;

        DELETE_COPY(GpuRenderer);
    };
}

#endif // OPENPOSE_CORE_GPU_RENDERER_HPP
