#ifndef OPENPOSE_THREAD_W_ID_GENERATOR_HPP
#define OPENPOSE_THREAD_W_ID_GENERATOR_HPP

#include <queue> // std::priority_queue
#include <openpose/core/common.hpp>
#include <openpose/thread/worker.hpp>
#include <openpose/utilities/pointerContainer.hpp>

namespace op
{
    template<typename TDatums>
    class WIdGenerator : public Worker<TDatums>
    {
    public:
        explicit WIdGenerator();

        virtual ~WIdGenerator();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        unsigned long long mGlobalCounter;

        DELETE_COPY(WIdGenerator);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WIdGenerator<TDatums>::WIdGenerator() :
        mGlobalCounter{0ull}
    {
    }

    template<typename TDatums>
    WIdGenerator<TDatums>::~WIdGenerator()
    {
    }

    template<typename TDatums>
    void WIdGenerator<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WIdGenerator<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Add ID
                for (auto& tDatumPtr : *tDatums)
                    // To avoid overwritting ID if e.g., custom input has already filled it
                    if (tDatumPtr->id == std::numeric_limits<unsigned long long>::max())
                        tDatumPtr->id = mGlobalCounter;
                // Increase ID
                const auto& tDatumPtr = (*tDatums)[0];
                if (tDatumPtr->subId == tDatumPtr->subIdMax)
                    mGlobalCounter++;
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

    COMPILE_TEMPLATE_DATUM(WIdGenerator);
}

#endif // OPENPOSE_THREAD_W_ID_GENERATOR_HPP
