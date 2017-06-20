#ifndef OPENPOSE_HAND_W_HAND_DETECTOR_UPDATE_HPP
#define OPENPOSE_HAND_W_HAND_DETECTOR_UPDATE_HPP

#include <memory> // std::shared_ptr
#include <openpose/thread/worker.hpp>
#include "handRenderer.hpp"

namespace op
{
    template<typename TDatums>
    class WHandDetectorUpdate : public Worker<TDatums>
    {
    public:
        explicit WHandDetectorUpdate(const std::shared_ptr<HandDetector>& handDetector);

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<HandDetector> spHandDetector;

        DELETE_COPY(WHandDetectorUpdate);
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
    WHandDetectorUpdate<TDatums>::WHandDetectorUpdate(const std::shared_ptr<HandDetector>& handDetector) :
        spHandDetector{handDetector}
    {
    }

    template<typename TDatums>
    void WHandDetectorUpdate<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WHandDetectorUpdate<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Detect people hand
                for (auto& tDatum : *tDatums)
                    spHandDetector->updateTracker(tDatum.poseKeypoints, tDatum.handKeypoints);
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

    COMPILE_TEMPLATE_DATUM(WHandDetectorUpdate);
}

#endif // OPENPOSE_HAND_W_HAND_DETECTOR_UPDATE_HPP
