#ifndef OPENPOSE__GUI__GUI_HPP
#define OPENPOSE__GUI__GUI_HPP

#include <atomic>
#include <memory> // std::shared_ptr
#include <opencv2/core/core.hpp>
#include "../pose/poseExtractor.hpp"
#include "../pose/poseRenderer.hpp"
#include "enumClasses.hpp"
#include "frameDisplayer.hpp"

namespace op
{
    class Gui
    {
    public:
        Gui(const bool fullScreen, const cv::Size& outputSize, const std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr,
            const std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr = nullptr,
            const std::vector<std::shared_ptr<PoseExtractor>>& poseExtractors = {}, const std::vector<std::shared_ptr<PoseRenderer>>& poseRenderers = {});

        void update(const cv::Mat& cvOutputData = cv::Mat{});

    private:
        // Frames display
        FrameDisplayer mFrameDisplayer;
        // Other variables
        std::vector<std::shared_ptr<PoseExtractor>> mPoseExtractors;
        std::vector<std::shared_ptr<PoseRenderer>> mPoseRenderers;
        std::shared_ptr<std::atomic<bool>> spIsRunning;
        std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>> spVideoSeek;
    };
}

#endif // OPENPOSE__GUI__GUI_HPP
