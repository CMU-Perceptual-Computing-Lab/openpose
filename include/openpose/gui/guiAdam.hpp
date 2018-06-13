#ifndef OPENPOSE_GUI_GUI_ADAM_HPP
#define OPENPOSE_GUI_GUI_ADAM_HPP

#ifdef WITH_3D_ADAM_MODEL
    #include <adam/totalmodel.h>
#endif
#include <openpose/core/common.hpp>
#include <openpose/gui/enumClasses.hpp>
#include <openpose/gui/gui.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/thread/workerConsumer.hpp>

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
                const std::vector<std::shared_ptr<Renderer>>& renderers = {},
                const std::shared_ptr<const TotalModel>& totalModel = nullptr
                // , const std::string& adamRenderedVideoPath = ""
                );

        virtual ~GuiAdam();

        virtual void initializationOnThread();

        void generateMesh(const Array<float>& poseKeypoints3D, const Array<float>& faceKeypoints3D,
                          const std::array<Array<float>, 2>& handKeypoints3D,
                          const Eigen::Matrix<double, 62, 3, Eigen::RowMajor>& adamPose,
                          const Eigen::Vector3d& adamTranslation,
                          const Eigen::Matrix<double, Eigen::Dynamic, 1>& vtVec,
                          const Eigen::Matrix<double, Eigen::Dynamic, 1>& j0Vec,
                          const double* const adamFaceCoeffsExp);

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
