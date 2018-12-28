#ifndef OPENPOSE_THREAD_W_FPS_MAX_HPP
#define OPENPOSE_THREAD_W_FPS_MAX_HPP

#include <thread>
#include <openpose/core/common.hpp>
#include <openpose/thread/worker.hpp>
#include <openpose/utilities/fastMath.hpp>

namespace op
{
    template<typename TDatums>
    class WFpsMax : public Worker<TDatums>
    {
    public:
        explicit WFpsMax(const double fpsMax);

        virtual ~WFpsMax();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        const unsigned long long mNanosecondsToSleep;

        DELETE_COPY(WFpsMax);
    };
}





// Implementation
namespace op
{
    template<typename TDatums>
    WFpsMax<TDatums>::WFpsMax(const double fpsMax) :
        mNanosecondsToSleep{uLongLongRound(1e9/fpsMax)}
    {
    }

    template<typename TDatums>
    WFpsMax<TDatums>::~WFpsMax()
    {
    }

    template<typename TDatums>
    void WFpsMax<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WFpsMax<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            // Debugging log
            dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Profiling speed
            const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
            // tDatums not used --> Avoid warning
            UNUSED(tDatums);
            // Sleep the desired time
            std::this_thread::sleep_for(std::chrono::nanoseconds{mNanosecondsToSleep});
            // Profiling speed
            Profiler::timerEnd(profilerKey);
            Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
            // Debugging log
            dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            this->stop();
            tDatums = nullptr;
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WFpsMax);
}

#endif // OPENPOSE_THREAD_W_FPS_MAX_HPP
