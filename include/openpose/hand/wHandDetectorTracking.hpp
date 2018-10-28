#ifndef OPENPOSE_HAND_W_HAND_DETECTOR_TRACKING_HPP
#define OPENPOSE_HAND_W_HAND_DETECTOR_TRACKING_HPP

#include <openpose/core/common.hpp>
#include <openpose/hand/handRenderer.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WHandDetectorTracking : public Worker<TDatums>
    {
    public:
        explicit WHandDetectorTracking(const std::shared_ptr<HandDetector>& handDetector);

        virtual ~WHandDetectorTracking();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<HandDetector> spHandDetector;

        DELETE_COPY(WHandDetectorTracking);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WHandDetectorTracking<TDatums>::WHandDetectorTracking(const std::shared_ptr<HandDetector>& handDetector) :
        spHandDetector{handDetector}
    {
    }

    template<typename TDatums>
    WHandDetectorTracking<TDatums>::~WHandDetectorTracking()
    {
    }

    template<typename TDatums>
    void WHandDetectorTracking<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WHandDetectorTracking<TDatums>::work(TDatums& tDatums)
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
                    tDatum.handRectangles = spHandDetector->trackHands(tDatum.poseKeypoints);
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
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

    COMPILE_TEMPLATE_DATUM(WHandDetectorTracking);
}

#endif // OPENPOSE_HAND_W_HAND_DETECTOR_TRACKING_HPP
