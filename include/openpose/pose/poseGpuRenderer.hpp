#ifndef OPENPOSE_POSE_POSE_GPU_RENDERER_HPP
#define OPENPOSE_POSE_POSE_GPU_RENDERER_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/gpuRenderer.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/pose/poseExtractorNet.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/pose/poseRenderer.hpp>

namespace op
{
    class OP_API PoseGpuRenderer : public GpuRenderer, public PoseRenderer
    {
    public:
        PoseGpuRenderer(const PoseModel poseModel, const std::shared_ptr<PoseExtractorNet>& poseExtractorNet,
                        const float renderThreshold, const bool blendOriginalFrame = true,
                        const float alphaKeypoint = POSE_DEFAULT_ALPHA_KEYPOINT,
                        const float alphaHeatMap = POSE_DEFAULT_ALPHA_HEAT_MAP,
                        const unsigned int elementToRender = 0u);

        ~PoseGpuRenderer();

        void initializationOnThread();

        std::pair<int, std::string> renderPose(Array<float>& outputData, const Array<float>& poseKeypoints,
                                               const float scaleInputToOutput,
                                               const float scaleNetToOutput = -1.f);

    private:
        const std::shared_ptr<PoseExtractorNet> spPoseExtractorNet;
        // Init with thread
        float* pGpuPose; // GPU aux memory

        DELETE_COPY(PoseGpuRenderer);
    };
}

#endif // OPENPOSE_POSE_POSE_GPU_RENDERER_HPP
