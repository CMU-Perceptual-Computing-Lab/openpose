#ifndef OPENPOSE_POSE_POSE_RENDERER_HPP
#define OPENPOSE_POSE_POSE_RENDERER_HPP

#include <memory> // std::shared_ptr
#include <openpose/core/array.hpp>
#include <openpose/core/enumClasses.hpp>
#include <openpose/core/point.hpp>
#include <openpose/core/renderer.hpp>
#include <openpose/utilities/macros.hpp>
#include "poseExtractor.hpp"
#include "poseParameters.hpp"

namespace op
{
    class PoseRenderer : public Renderer
    {
    public:
        explicit PoseRenderer(const Point<int>& heatMapsSize, const Point<int>& outputSize, const PoseModel poseModel,
                              const std::shared_ptr<PoseExtractor>& poseExtractor, const bool blendOriginalFrame = true,
                              const float alphaKeypoint = POSE_DEFAULT_ALPHA_KEYPOINT, const float alphaHeatMap = POSE_DEFAULT_ALPHA_HEAT_MAP,
                              const unsigned int elementToRender = 0u, const RenderMode renderMode = RenderMode::Cpu);

        ~PoseRenderer();

        void initializationOnThread();

        bool getBlendOriginalFrame() const;

        bool getShowGooglyEyes() const;

        void setBlendOriginalFrame(const bool blendOriginalFrame);

        void setShowGooglyEyes(const bool showGooglyEyes);

        std::pair<int, std::string> renderPose(Array<float>& outputData, const Array<float>& poseKeypoints, const float scaleNetToOutput = -1.f);

    private:
        const Point<int> mHeatMapsSize;
        const Point<int> mOutputSize;
        const PoseModel mPoseModel;
        const std::map<unsigned int, std::string> mPartIndexToName;
        const std::shared_ptr<PoseExtractor> spPoseExtractor;
        const RenderMode mRenderMode;
        std::atomic<bool> mBlendOriginalFrame;
        std::atomic<bool> mShowGooglyEyes;
        // Init with thread
        float* pGpuPose; // GPU aux memory

        std::pair<int, std::string> renderPoseCpu(Array<float>& outputData, const Array<float>& poseKeypoints, const float scaleNetToOutput = -1.f);

        std::pair<int, std::string> renderPoseGpu(Array<float>& outputData, const Array<float>& poseKeypoints, const float scaleNetToOutput = -1.f);

        DELETE_COPY(PoseRenderer);
    };
}

#endif // OPENPOSE_POSE_POSE_RENDERER_HPP
