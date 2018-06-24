#ifndef OPENPOSE_THREAD_W_QUEUE_ASSEMBLER_HPP
#define OPENPOSE_THREAD_W_QUEUE_ASSEMBLER_HPP

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
    class WQueueAssembler : public Worker<TDatums>
    {
    public:
        explicit WQueueAssembler();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        TDatums mNextTDatums;

        DELETE_COPY(WQueueAssembler);
    };
}





// Implementation
#include <chrono>
#include <thread>
namespace op
{
    template<typename TDatums, typename TDatumsNoPtr>
    WQueueAssembler<TDatums, TDatumsNoPtr>::WQueueAssembler()
    {
    }

    template<typename TDatums, typename TDatumsNoPtr>
    void WQueueAssembler<TDatums, TDatumsNoPtr>::initializationOnThread()
    {
    }

    template<typename TDatums, typename TDatumsNoPtr>
    void WQueueAssembler<TDatums, TDatumsNoPtr>::work(TDatums& tDatums)
    {
        try
        {
            // Profiling speed
            const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
            // Input TDatums -> enqueue it
            if (checkNoNullNorEmpty(tDatums))
            {
                // Security check
                if (tDatums->size() > 1)
                    error("This function assumes that QueueSplitter was applied in the first place, i.e.,"
                          " that there is only 1 element per TDatums.", __LINE__, __FUNCTION__, __FILE__);
                auto tDatum = (*tDatums)[0];
// std::cout << __LINE__ << " " << tDatum.id << " " << tDatum.subId << " " << tDatum.subIdMax << std::endl;
                // Single view --> Return
                if (mNextTDatums == nullptr && tDatum.subIdMax == 0)
                    return;
                // Multiple view --> Merge views into different TDatums (1st frame)
                if (mNextTDatums == nullptr)
                {
                    mNextTDatums = std::make_shared<TDatumsNoPtr>();
                    mNextTDatums->emplace_back(tDatum);
                    tDatums = nullptr;
                }
                // Multiple view --> Merge views into different TDatums
                else
                {
// std::cout << __LINE__ << " " << tDatum.id << " " << tDatum.subId << " " << tDatum.subIdMax << std::endl;
                    mNextTDatums->emplace_back(tDatum);
// std::cout << __LINE__ << " " << mNextTDatums->back().id << " " << mNextTDatums->back().subId << " " << mNextTDatums->back().subIdMax << std::endl;
for (auto& tDatum : *mNextTDatums)
std::cout << __LINE__ << " " << tDatum.id << " " << tDatum.subId << " " << tDatum.subIdMax << std::endl;
                    // Last view - Return frame
                    if (mNextTDatums->back().subId == mNextTDatums->back().subIdMax)
                    {
                        tDatums = mNextTDatums;
                        mNextTDatums = nullptr;
for (auto& tDatum : *tDatums)
std::cout << __LINE__ << " " << tDatum.id << " " << tDatum.subId << " " << tDatum.subIdMax << std::endl;
                    }
                    // Non-last view - Return nothing
                    else
                    {
                        tDatums = nullptr;
                        return;
                    }
                }
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
            // Sleep if no new tDatums to either pop or push
            else
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
        catch (const std::exception& e)
        {
            this->stop();
            tDatums = nullptr;
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    extern template class WQueueAssembler<DATUM_BASE, DATUM_BASE_NO_PTR>;
}

#endif // OPENPOSE_THREAD_W_QUEUE_ASSEMBLER_HPP
