#ifndef OPENPOSE_HAND_HAND_CPU_RENDERER_HPP
#define OPENPOSE_HAND_HAND_CPU_RENDERER_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/renderer.hpp>
#include <openpose/hand/handParameters.hpp>
#include <openpose/hand/handRenderer.hpp>

namespace op
{
    class OP_API HandCpuRenderer : public Renderer, public HandRenderer
    {
    public:
        HandCpuRenderer(const float renderThreshold, const float alphaKeypoint = HAND_DEFAULT_ALPHA_KEYPOINT,
                        const float alphaHeatMap = HAND_DEFAULT_ALPHA_HEAT_MAP);

        virtual ~HandCpuRenderer();

        void renderHandInherited(Array<float>& outputData, const std::array<Array<float>, 2>& handKeypoints);

        DELETE_COPY(HandCpuRenderer);
    };
}

#endif // OPENPOSE_HAND_HAND_CPU_RENDERER_HPP
