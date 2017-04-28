#ifndef OPENPOSE__POSE__POSE_RENDERER_HPP
#define OPENPOSE__POSE__POSE_RENDERER_HPP

#include <memory> // std::shared_ptr
#include <opencv2/core/core.hpp>
#include "../core/array.hpp"
#include "../core/renderer.hpp"
#include "../utilities/macros.hpp"
#include "poseExtractor.hpp"
#include "poseParameters.hpp"

namespace op
{
    class PoseRenderer : public Renderer
    {
    public:
        explicit PoseRenderer(const cv::Size& heatMapsSize, const cv::Size& outputSize, const PoseModel poseModel, const std::shared_ptr<PoseExtractor>& poseExtractor,
                              const bool blendOriginalFrame = true, const float alphaPose = POSE_DEFAULT_ALPHA_POSE,
                              const float alphaHeatMap = POSE_DEFAULT_ALPHA_HEATMAP, const int elementToRender = 0);

        ~PoseRenderer();

        void initializationOnThread();

        void increaseElementToRender(const int increment);

        void setElementToRender(const int elementToRender);

        float getAlphaPose() const;

        void setAlphaPose(const float alphaPose);

        float getAlphaHeatMap() const;

        void setAlphaHeatMap(const float alphaHeatMap);

        bool getBlendOriginalFrame() const;

        bool getShowGooglyEyes() const;

        void setBlendOriginalFrame(const bool blendOriginalFrame);

        void setShowGooglyEyes(const bool showGooglyEyes);

        std::pair<int, std::string> renderPose(Array<float>& outputData, const Array<float>& poseKeyPoints, const double scaleNetToOutput = -1.);

    private:
        const cv::Size mHeatMapsSize;
        const cv::Size mOutputSize;
        const PoseModel mPoseModel;
        const std::map<unsigned char, std::string> mPartIndexToName;
        const int mNumberElementsToRender;
        const std::shared_ptr<PoseExtractor> spPoseExtractor;
        float mAlphaPose;
        float mAlphaHeatMap;
        std::atomic<bool> mBlendOriginalFrame;
        std::atomic<bool> mShowGooglyEyes;
        std::atomic<int> mElementToRender;
        // Init with thread
        float* pGpuPose;        // GPU aux memory

        DELETE_COPY(PoseRenderer);
    };
}

#endif // OPENPOSE__POSE__POSE_RENDERER_HPP
