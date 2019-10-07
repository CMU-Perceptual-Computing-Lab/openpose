#ifndef OPENPOSE_GUI_GUI_3D_HPP
#define OPENPOSE_GUI_GUI_3D_HPP

#include <openpose/core/common.hpp>
#include <openpose/gui/enumClasses.hpp>
#include <openpose/gui/gui.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/thread/workerConsumer.hpp>

namespace op
{
    class OP_API Gui3D : public Gui
    {
    public:
        Gui3D(const Point<int>& outputSize, const bool fullScreen,
              const std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr,
              const std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr = nullptr,
              const std::vector<std::shared_ptr<PoseExtractorNet>>& poseExtractorNets = {},
              const std::vector<std::shared_ptr<FaceExtractorNet>>& faceExtractorNets = {},
              const std::vector<std::shared_ptr<HandExtractorNet>>& handExtractorNets = {},
              const std::vector<std::shared_ptr<Renderer>>& renderers = {},
              const PoseModel poseModel = PoseModel::BODY_25,
              const DisplayMode displayMode = DisplayMode::DisplayAll,
              const bool copyGlToCvMat = false);

        virtual ~Gui3D();

        virtual void initializationOnThread();

        void setKeypoints(const Array<float>& poseKeypoints3D, const Array<float>& faceKeypoints3D,
                          const Array<float>& leftHandKeypoints3D, const Array<float>& rightHandKeypoints3D);

        virtual void update();

        virtual Matrix readCvMat();

    private:
        const bool mCopyGlToCvMat;
    };
}

#endif // OPENPOSE_GUI_GUI_3D_HPP
