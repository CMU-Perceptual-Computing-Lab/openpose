#ifndef OPENPOSE_HAND_W_HAND_RENDERER_HPP
#define OPENPOSE_HAND_W_HAND_RENDERER_HPP

#include <openpose/core/common.hpp>
#include <openpose/hand/handRenderer.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WHandRenderer : public Worker<TDatums>
    {
    public:
        explicit WHandRenderer(const std::shared_ptr<HandRenderer>& handRenderer);

        virtual ~WHandRenderer();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<HandRenderer> spHandRenderer;

        DELETE_COPY(WHandRenderer);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WHandRenderer<TDatums>::WHandRenderer(const std::shared_ptr<HandRenderer>& handRenderer) :
        spHandRenderer{handRenderer}
    {
    }

    template<typename TDatums>
    WHandRenderer<TDatums>::~WHandRenderer()
    {
    }

    template<typename TDatums>
    void WHandRenderer<TDatums>::initializationOnThread()
    {
        spHandRenderer->initializationOnThread();
    }

    template<typename TDatums>
    void WHandRenderer<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Render people hands
                for (auto& tDatumPtr : *tDatums)
                    spHandRenderer->renderHand(
                        tDatumPtr->outputData, tDatumPtr->handKeypoints, (float)tDatumPtr->scaleInputToOutput);
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

    COMPILE_TEMPLATE_DATUM(WHandRenderer);
}

#endif // OPENPOSE_HAND_W_HAND_RENDERER_HPP
