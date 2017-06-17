#ifndef OPENPOSE_HAND_W_HAND_EXTRACTOR_HPP
#define OPENPOSE_HAND_W_HAND_EXTRACTOR_HPP

#include <memory> // std::shared_ptr
#include <openpose/thread/worker.hpp>
#include "handRenderer.hpp"

namespace op
{
    template<typename TDatums>
    class WHandExtractor : public Worker<TDatums>
    {
    public:
        explicit WHandExtractor(const std::shared_ptr<HandExtractor>& handExtractor);

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<HandExtractor> spHandExtractor;

        DELETE_COPY(WHandExtractor);
    };
}





// Implementation
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/macros.hpp>
#include <openpose/utilities/pointerContainer.hpp>
#include <openpose/utilities/profiler.hpp>
namespace op
{
    template<typename TDatums>
    WHandExtractor<TDatums>::WHandExtractor(const std::shared_ptr<HandExtractor>& handExtractor) :
        spHandExtractor{handExtractor}
    {
    }

    template<typename TDatums>
    void WHandExtractor<TDatums>::initializationOnThread()
    {
        spHandExtractor->initializationOnThread();
    }

    template<typename TDatums>
    void WHandExtractor<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Extract people hands
                for (auto& tDatum : *tDatums)
                {
                    spHandExtractor->forwardPass(tDatum.handRectangles, tDatum.cvInputData, tDatum.scaleInputToOutput);
                    tDatum.handKeypoints = spHandExtractor->getHandKeypoints();
                }
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__, Profiler::DEFAULT_X);
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            tDatums = nullptr;
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WHandExtractor);
}

#endif // OPENPOSE_HAND_W_HAND_EXTRACTOR_HPP
