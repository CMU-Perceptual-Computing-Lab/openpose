#ifndef OPENPOSE_HAND_HAND_GPU_RENDERER_HPP
#define OPENPOSE_HAND_HAND_GPU_RENDERER_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/gpuRenderer.hpp>
#include <openpose/hand/handParameters.hpp>
#include <openpose/hand/handRenderer.hpp>

namespace op
{
    class OP_API HandGpuRenderer : public GpuRenderer, public HandRenderer
    {
    public:
        HandGpuRenderer(const float renderThreshold,
                        const float alphaKeypoint = HAND_DEFAULT_ALPHA_KEYPOINT,
                        const float alphaHeatMap = HAND_DEFAULT_ALPHA_HEAT_MAP);

        virtual ~HandGpuRenderer();

        void initializationOnThread();

        void renderHandInherited(Array<float>& outputData, const std::array<Array<float>, 2>& handKeypoints);

    private:
        float* pGpuHand; // GPU aux memory
        float* pMaxPtr; // GPU aux memory
        float* pMinPtr; // GPU aux memory
        float* pScalePtr; // GPU aux memory

        DELETE_COPY(HandGpuRenderer);
    };
}

#endif // OPENPOSE_HAND_HAND_GPU_RENDERER_HPP
