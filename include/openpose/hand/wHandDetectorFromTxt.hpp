#ifndef OPENPOSE_HAND_W_HAND_DETECTOR_FROM_JSON_HPP
#define OPENPOSE_HAND_W_HAND_DETECTOR_FROM_JSON_HPP

#include <openpose/core/common.hpp>
#include <openpose/hand/handDetectorFromTxt.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WHandDetectorFromTxt : public Worker<TDatums>
    {
    public:
        explicit WHandDetectorFromTxt(const std::shared_ptr<HandDetectorFromTxt>& handDetectorFromTxt);

        virtual ~WHandDetectorFromTxt();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<HandDetectorFromTxt> spHandDetectorFromTxt;

        DELETE_COPY(WHandDetectorFromTxt);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WHandDetectorFromTxt<TDatums>::WHandDetectorFromTxt(const std::shared_ptr<HandDetectorFromTxt>& handDetectorFromTxt) :
        spHandDetectorFromTxt{handDetectorFromTxt}
    {
    }

    template<typename TDatums>
    WHandDetectorFromTxt<TDatums>::~WHandDetectorFromTxt()
    {
    }

    template<typename TDatums>
    void WHandDetectorFromTxt<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WHandDetectorFromTxt<TDatums>::work(TDatums& tDatums)
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
                    tDatum.handRectangles = spHandDetectorFromTxt->detectHands();
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

    COMPILE_TEMPLATE_DATUM(WHandDetectorFromTxt);
}

#endif // OPENPOSE_HAND_W_HAND_DETECTOR_FROM_JSON_HPP
