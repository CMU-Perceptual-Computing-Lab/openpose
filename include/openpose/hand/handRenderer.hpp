#ifndef OPENPOSE_HAND_HAND_RENDERER_HPP
#define OPENPOSE_HAND_HAND_RENDERER_HPP

#include <array>
#include <openpose/core/array.hpp>
#include <openpose/core/enumClasses.hpp>
#include <openpose/core/point.hpp>
#include <openpose/core/renderer.hpp>
#include <openpose/thread/worker.hpp>
#include "handParameters.hpp"

namespace op
{
    class HandRenderer : public Renderer
    {
    public:
        HandRenderer(const Point<int>& frameSize, const float alphaKeypoint = HAND_DEFAULT_ALPHA_KEYPOINT,
                     const float alphaHeatMap = HAND_DEFAULT_ALPHA_HEAT_MAP, const RenderMode renderMode = RenderMode::Cpu);

        ~HandRenderer();

        void initializationOnThread();

        void renderHand(Array<float>& outputData, const std::array<Array<float>, 2>& handKeypoints);

    private:
        const Point<int> mFrameSize;
        const RenderMode mRenderMode;
        float* pGpuHand; // GPU aux memory

        void renderHandCpu(Array<float>& outputData, const std::array<Array<float>, 2>& handKeypoints) const;

        void renderHandGpu(Array<float>& outputData, const std::array<Array<float>, 2>& handKeypoints);

        DELETE_COPY(HandRenderer);
    };
}

#endif // OPENPOSE_HAND_HAND_RENDERER_HPP
