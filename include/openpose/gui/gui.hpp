#ifndef OPENPOSE_GUI_GUI_HPP
#define OPENPOSE_GUI_GUI_HPP

#include <atomic>
#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/common.hpp>
#include <openpose/core/renderer.hpp>
#include <openpose/gui/frameDisplayer.hpp>
#include <openpose/pose/poseExtractorNet.hpp>
#include <openpose/face/faceExtractorNet.hpp>
#include <openpose/hand/handExtractorNet.hpp>

namespace op
{
    class OP_API Gui
    {
    public:
        Gui(const Point<int>& outputSize, const bool fullScreen,
            const std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr,
            const std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr = nullptr,
            const std::vector<std::shared_ptr<PoseExtractorNet>>& poseExtractorNets = {},
            const std::vector<std::shared_ptr<FaceExtractorNet>>& faceExtractorNets = {},
            const std::vector<std::shared_ptr<HandExtractorNet>>& handExtractorNets = {},
            const std::vector<std::shared_ptr<Renderer>>& renderers = {},
            const DisplayMode displayMode = DisplayMode::Display2D);

        virtual ~Gui();

        virtual void initializationOnThread();

        void setImage(const cv::Mat& cvMatOutput);

        void setImage(const std::vector<cv::Mat>& cvMatOutputs);

        virtual void update();

    protected:
        std::shared_ptr<std::atomic<bool>> spIsRunning;
        DisplayMode mDisplayMode;
        DisplayMode mDisplayModeOriginal;

    private:
        // Frames display
        FrameDisplayer mFrameDisplayer;
        // Other variables
        std::vector<std::shared_ptr<PoseExtractorNet>> mPoseExtractorNets;
        std::vector<std::shared_ptr<FaceExtractorNet>> mFaceExtractorNets;
        std::vector<std::shared_ptr<HandExtractorNet>> mHandExtractorNets;
        std::vector<std::shared_ptr<Renderer>> mRenderers;
        std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>> spVideoSeek;
    };
}

#endif // OPENPOSE_GUI_GUI_HPP
