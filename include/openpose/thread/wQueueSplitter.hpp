#ifndef OPENPOSE_THREAD_W_QUEUE_SPLITTER_HPP
#define OPENPOSE_THREAD_W_QUEUE_SPLITTER_HPP

#include <iostream>
#include <queue> // std::queue
#include <openpose/core/common.hpp>
#include <openpose/thread/worker.hpp>
#include <openpose/utilities/pointerContainer.hpp>

namespace op
{
    // Note: The goal of WQueueAssembler and WQueueSplitter is to reduce the latency of OpenPose. E.g., if 4 cameras
    // in stereo mode, without this, OpenPose would have to process all 4 cameras with the same GPU. In this way,
    // this work is parallelized over GPUs (1 view for each).
    // Pros: Latency highly recuded, same speed
    // Cons: Requires these extra 2 classes and proper threads for them
    template<typename TDatums, typename TDatumsNoPtr>
    class WQueueSplitter : public Worker<TDatums>
    {
    public:
        explicit WQueueSplitter();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::queue<TDatums> mQueuedElements;

        DELETE_COPY(WQueueSplitter);
    };
}





// Implementation
#include <chrono>
#include <thread>
namespace op
{
    template<typename TDatums, typename TDatumsNoPtr>
    WQueueSplitter<TDatums, TDatumsNoPtr>::WQueueSplitter()
    {
    }

    template<typename TDatums, typename TDatumsNoPtr>
    void WQueueSplitter<TDatums, TDatumsNoPtr>::initializationOnThread()
    {
    }

    template<typename TDatums, typename TDatumsNoPtr>
    void WQueueSplitter<TDatums, TDatumsNoPtr>::work(TDatums& tDatums)
    {
        try
        {
            // Profiling speed
            const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
            // Input TDatums -> enqueue it
            if (checkNoNullNorEmpty(tDatums))
            {
                // Single view --> Return
                if (mQueuedElements.empty() && tDatums->size() == 1)
                    return;
                // Multiple view --> Split views into different TDatums
                // for (auto& tDatum : *tDatums)
                for (auto i = 0u ; i < tDatums->size() ; i++)
                {
                    auto& tDatum = (*tDatums)[i];
                    tDatum.subId = i;
                    tDatum.subIdMax = tDatums->size()-1;
// std::cout << __LINE__ << " " << tDatum.id << " " << tDatum.subId << " " << tDatum.subIdMax << std::endl;
                    mQueuedElements.emplace(
                        std::make_shared<TDatumsNoPtr>(TDatumsNoPtr{tDatum}));
// std::cout << __LINE__ << " " << mQueuedElements.back()->at(0).id << " " << mQueuedElements.back()->at(0).subId << " " << mQueuedElements.back()->at(0).subIdMax << std::endl << std::endl;
                }
                tDatums = nullptr;
                // Return oldest view
                if (!mQueuedElements.empty())
                {
                    tDatums = mQueuedElements.front();
                    mQueuedElements.pop();
// for (auto& tDatum : *tDatums)
// std::cout << __LINE__ << " " << tDatum.id << " " << tDatum.subId << " " << tDatum.subIdMax << std::endl;
// std::cout << __LINE__ << " " << tDatums->size() << std::endl;
                }
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
            // Sleep if no new tDatums to either pop or push
            else if (mQueuedElements.empty())
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
        catch (const std::exception& e)
        {
            this->stop();
            tDatums = nullptr;
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    extern template class WQueueSplitter<DATUM_BASE, DATUM_BASE_NO_PTR>;
}

#endif // OPENPOSE_THREAD_W_QUEUE_SPLITTER_HPP
