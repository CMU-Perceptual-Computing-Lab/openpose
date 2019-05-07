#ifndef OPENPOSE_POSE_POSE_CPU_RENDERER_HPP
#define OPENPOSE_POSE_POSE_CPU_RENDERER_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/renderer.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/pose/poseParametersRender.hpp>
#include <openpose/pose/poseRenderer.hpp>

namespace op
{
    class OP_API PoseCpuRenderer : public Renderer, public PoseRenderer
    {
    public:
        PoseCpuRenderer(
            const PoseModel poseModel, const float renderThreshold, const bool blendOriginalFrame = true,
            const float alphaKeypoint = POSE_DEFAULT_ALPHA_KEYPOINT,
            const float alphaHeatMap = POSE_DEFAULT_ALPHA_HEAT_MAP, const unsigned int elementToRender = 0u);

        virtual ~PoseCpuRenderer();

        std::pair<int, std::string> renderPose(
            Array<float>& outputData, const Array<float>& poseKeypoints, const float scaleInputToOutput,
            const float scaleNetToOutput = -1.f);

    private:
        DELETE_COPY(PoseCpuRenderer);
    };
}

#endif // OPENPOSE_POSE_POSE_CPU_RENDERER_HPP
