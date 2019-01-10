#ifndef OPENPOSE_PRODUCER_W_DATUM_PRODUCER_HPP
#define OPENPOSE_PRODUCER_W_DATUM_PRODUCER_HPP

#include <limits> // std::numeric_limits
#include <queue> // std::queue
#include <openpose/core/common.hpp>
#include <openpose/producer/datumProducer.hpp>
#include <openpose/thread/workerProducer.hpp>

namespace op
{
    template<typename TDatum>
    class WDatumProducer : public WorkerProducer<std::shared_ptr<std::vector<std::shared_ptr<TDatum>>>>
    {
    public:
        explicit WDatumProducer(const std::shared_ptr<DatumProducer<TDatum>>& datumProducer);

        virtual ~WDatumProducer();

        void initializationOnThread();

        std::shared_ptr<std::vector<std::shared_ptr<TDatum>>> workProducer();

    private:
        std::shared_ptr<DatumProducer<TDatum>> spDatumProducer;
        std::queue<std::shared_ptr<std::vector<std::shared_ptr<TDatum>>>> mQueuedElements;

        DELETE_COPY(WDatumProducer);
    };
}





// Implementation
#include <openpose/core/datum.hpp>
namespace op
{
    template<typename TDatum>
    WDatumProducer<TDatum>::WDatumProducer(
        const std::shared_ptr<DatumProducer<TDatum>>& datumProducer) :
        spDatumProducer{datumProducer}
    {
    }

    template<typename TDatum>
    WDatumProducer<TDatum>::~WDatumProducer()
    {
    }


    template<typename TDatum>
    void WDatumProducer<TDatum>::initializationOnThread()
    {
    }

    template<typename TDatum>
    std::shared_ptr<std::vector<std::shared_ptr<TDatum>>> WDatumProducer<TDatum>::workProducer()
    {
        try
        {
            // Debugging log
            dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Profiling speed
            const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
            // Create and fill final shared pointer
            std::shared_ptr<std::vector<std::shared_ptr<TDatum>>> tDatums;
            // Producer
            if (mQueuedElements.empty())
            {
                bool isRunning;
                std::tie(isRunning, tDatums) = spDatumProducer->checkIfRunningAndGetDatum();
                // Stop Worker if producer finished
                if (!isRunning)
                    this->stop();
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
            // Equivalent to WQueueSplitter
            // Queued elements - Multiple views --> Split views into different share pointers
            if (tDatums != nullptr && tDatums->size() > 1)
            {
                // Add tDatums to mQueuedElements
                for (auto i = 0u ; i < tDatums->size() ; i++)
                {
                    auto& tDatumPtr = (*tDatums)[i];
                    tDatumPtr->subId = i;
                    tDatumPtr->subIdMax = tDatums->size()-1;
                    mQueuedElements.emplace(
                        std::make_shared<std::vector<std::shared_ptr<TDatum>>>(
                            std::vector<std::shared_ptr<TDatum>>{tDatumPtr}));
                }
            }
            // Queued elements - Multiple views --> Return oldest view
            if (!mQueuedElements.empty())
            {
                tDatums = mQueuedElements.front();
                mQueuedElements.pop();
            }
            // Return result
            return tDatums;
        }
        catch (const std::exception& e)
        {
            this->stop();
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    extern template class WDatumProducer<BASE_DATUM>;
}

#endif // OPENPOSE_PRODUCER_W_DATUM_PRODUCER_HPP
