#ifndef OPENPOSE_HAND_W_HAND_DETECTOR_HPP
#define OPENPOSE_HAND_W_HAND_DETECTOR_HPP

#include <openpose/core/common.hpp>
#include <openpose/hand/handDetector.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WHandDetector : public Worker<TDatums>
    {
    public:
        explicit WHandDetector(const std::shared_ptr<HandDetector>& handDetector);

        virtual ~WHandDetector();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<HandDetector> spHandDetector;

        DELETE_COPY(WHandDetector);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WHandDetector<TDatums>::WHandDetector(const std::shared_ptr<HandDetector>& handDetector) :
        spHandDetector{handDetector}
    {
    }

    template<typename TDatums>
    WHandDetector<TDatums>::~WHandDetector()
    {
    }

    template<typename TDatums>
    void WHandDetector<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WHandDetector<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Detect people hand
                for (auto& tDatumPtr : *tDatums)
                    tDatumPtr->handRectangles = spHandDetector->detectHands(tDatumPtr->poseKeypoints);
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

    COMPILE_TEMPLATE_DATUM(WHandDetector);
}

#endif // OPENPOSE_HAND_W_HAND_DETECTOR_HPP
