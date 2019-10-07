#ifndef OPENPOSE_POSE_W_POSE_RENDERER_HPP
#define OPENPOSE_POSE_W_POSE_RENDERER_HPP

#include <openpose/core/common.hpp>
#include <openpose/pose/poseRenderer.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WPoseRenderer : public Worker<TDatums>
    {
    public:
        explicit WPoseRenderer(const std::shared_ptr<PoseRenderer>& poseRendererSharedPtr);

        virtual ~WPoseRenderer();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<PoseRenderer> spPoseRenderer;

        DELETE_COPY(WPoseRenderer);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WPoseRenderer<TDatums>::WPoseRenderer(const std::shared_ptr<PoseRenderer>& poseRendererSharedPtr) :
        spPoseRenderer{poseRendererSharedPtr}
    {
    }

    template<typename TDatums>
    WPoseRenderer<TDatums>::~WPoseRenderer()
    {
    }

    template<typename TDatums>
    void WPoseRenderer<TDatums>::initializationOnThread()
    {
        try
        {
            spPoseRenderer->initializationOnThread();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums>
    void WPoseRenderer<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Render people pose
                for (auto& tDatumPtr : *tDatums)
                    tDatumPtr->elementRendered = spPoseRenderer->renderPose(
                        tDatumPtr->outputData, tDatumPtr->poseKeypoints, (float)tDatumPtr->scaleInputToOutput,
                        (float)tDatumPtr->scaleNetToOutput);
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            tDatums = nullptr;
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WPoseRenderer);
}

#endif // OPENPOSE_POSE_W_POSE_RENDERER_HPP
