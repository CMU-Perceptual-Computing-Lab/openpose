#ifdef USE_3D_ADAM_MODEL
#ifndef OPENPOSE_GUI_GUI_ADAM_HPP
#define OPENPOSE_GUI_GUI_ADAM_HPP

#ifdef USE_3D_ADAM_MODEL
    #include <adam/totalmodel.h>
#endif
#include <openpose/core/common.hpp>
#include <openpose/gui/enumClasses.hpp>
#include <openpose/gui/gui.hpp>

namespace op
{
    // This worker will do 3-D rendering
    class OP_API GuiAdam : public Gui
    {
    public:
        GuiAdam(const Point<int>& outputSize, const bool fullScreen,
                const std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr,
                const std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr = nullptr,
                const std::vector<std::shared_ptr<PoseExtractorNet>>& poseExtractorNets = {},
                const std::vector<std::shared_ptr<FaceExtractorNet>>& faceExtractorNets = {},
                const std::vector<std::shared_ptr<HandExtractorNet>>& handExtractorNets = {},
                const std::vector<std::shared_ptr<Renderer>>& renderers = {},
                const DisplayMode displayMode = DisplayMode::DisplayAll,
                const std::shared_ptr<const TotalModel>& totalModel = nullptr,
                const std::string& adamRenderedVideoPath = "");

        virtual ~GuiAdam();

        virtual void initializationOnThread();

        void generateMesh(const Array<float>& poseKeypoints3D, const Array<float>& faceKeypoints3D,
                          const std::array<Array<float>, 2>& handKeypoints3D,
                          const double* const adamPosePtr,
                          const double* const adamTranslationPtr,
                          const double* const vtVecPtr, const int vtVecRows,
                          const double* const j0VecPtr, const int j0VecRows,
                          const double* const adamFaceCoeffsExpPtr);

        virtual void update();

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplGuiAdam;
        std::shared_ptr<ImplGuiAdam> spImpl;

        // PIMP requires DELETE_COPY & destructor, or extra code
        // http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html
        DELETE_COPY(GuiAdam);
    };
}

#endif // OPENPOSE_GUI_GUI_ADAM_HPP
#endif
