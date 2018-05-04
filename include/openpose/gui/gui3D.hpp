#ifndef OPENPOSE_GUI_GUI_3D_HPP
#define OPENPOSE_GUI_GUI_3D_HPP

#include <openpose/core/common.hpp>
#include <openpose/gui/enumClasses.hpp>
#include <openpose/gui/gui.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/thread/workerConsumer.hpp>

namespace op
{
    // This worker will do 3-D rendering
    class OP_API Gui3D : public Gui
    {
    public:
        Gui3D(const Point<int>& outputSize, const bool fullScreen,
              const std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr,
              const std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr = nullptr,
              const std::vector<std::shared_ptr<PoseExtractorNet>>& poseExtractorNets = {},
              const std::vector<std::shared_ptr<Renderer>>& renderers = {},
              const PoseModel poseModel = PoseModel::COCO_18,
              const DisplayMode displayMode = DisplayMode::DisplayAll);

        ~Gui3D();

        void initializationOnThread();

        void setKeypoints(const Array<float>& poseKeypoints3D, const Array<float>& faceKeypoints3D,
                          const Array<float>& leftHandKeypoints3D, const Array<float>& rightHandKeypoints3D);

        void update();

    private:
        DisplayMode mDisplayMode;
    };
}

#endif // OPENPOSE_GUI_GUI_3D_HPP
