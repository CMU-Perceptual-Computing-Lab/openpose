#ifndef OPENPOSE_CORE_RENDERER_HPP
#define OPENPOSE_CORE_RENDERER_HPP

#include <atomic>
#include <tuple>
#include <openpose/core/common.hpp>

namespace op
{
    class OP_API Renderer
    {
    public:
        explicit Renderer(const unsigned long long volume, const float alphaKeypoint, const float alphaHeatMap,
                          const unsigned int elementToRender = 0u, const unsigned int numberElementsToRender = 0u);

        ~Renderer();

        void initializationOnThread();

        void increaseElementToRender(const int increment);

        void setElementToRender(const int elementToRender);

        std::tuple<std::shared_ptr<float*>, std::shared_ptr<bool>, std::shared_ptr<std::atomic<unsigned int>>,
                   std::shared_ptr<const unsigned int>> getSharedParameters();

        void setSharedParametersAndIfLast(const std::tuple<std::shared_ptr<float*>, std::shared_ptr<bool>,
                                                           std::shared_ptr<std::atomic<unsigned int>>,
                                                           std::shared_ptr<const unsigned int>>& tuple, const bool isLast);

        float getAlphaKeypoint() const;

        void setAlphaKeypoint(const float alphaKeypoint);

        float getAlphaHeatMap() const;

        void setAlphaHeatMap(const float alphaHeatMap);

    protected:
        std::shared_ptr<float*> spGpuMemoryPtr;
        std::shared_ptr<std::atomic<unsigned int>> spElementToRender;
        std::shared_ptr<const unsigned int> spNumberElementsToRender;

        void cpuToGpuMemoryIfNotCopiedYet(const float* const cpuMemory);

        void gpuToCpuMemoryIfLastRenderer(float* cpuMemory);

    private:
        const unsigned long long mVolume;
        float mAlphaKeypoint;
        float mAlphaHeatMap;
        bool mIsFirstRenderer;
        bool mIsLastRenderer;
        std::shared_ptr<bool> spGpuMemoryAllocated;

        DELETE_COPY(Renderer);
    };
}

#endif // OPENPOSE_CORE_RENDERER_HPP
