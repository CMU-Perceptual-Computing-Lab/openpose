#ifndef OPENPOSE__CORE__RENDERER_HPP
#define OPENPOSE__CORE__RENDERER_HPP

#include <memory> // std::shared_ptr
#include "../utilities/macros.hpp"

namespace op
{
    class Renderer
    {
    public:
        explicit Renderer(const unsigned long long volume);

        ~Renderer();

        void initializationOnThread();

        std::pair<std::shared_ptr<float*>, std::shared_ptr<bool>> getGpuMemoryAndSetAsFirst();

        void setGpuMemoryAndSetIfLast(const std::pair<std::shared_ptr<float*>, std::shared_ptr<bool>>& gpuMemory, const bool isLast);

    protected:
        std::shared_ptr<float*> spGpuMemoryPtr;

        void cpuToGpuMemoryIfNotCopiedYet(const float* const cpuMemory);

        void gpuToCpuMemoryIfLastRenderer(float* cpuMemory);

    private:
        const unsigned long long mVolume;
        bool mIsFirstRenderer;
        bool mIsLastRenderer;
        std::shared_ptr<bool> spGpuMemoryAllocated;

        DELETE_COPY(Renderer);
    };
}

#endif // OPENPOSE__CORE__RENDERER_HPP
