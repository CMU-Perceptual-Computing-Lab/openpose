#ifndef OPENPOSE_THREAD_W_QUEUE_ASSEMBLER_HPP
#define OPENPOSE_THREAD_W_QUEUE_ASSEMBLER_HPP

#include <queue> // std::queue
#include <openpose/core/common.hpp>
#include <openpose/thread/worker.hpp>
#include <openpose/utilities/pointerContainer.hpp>

namespace op
{
    // Note: The goal of WQueueAssembler and WQueueSplitter (integrated in wDatumProducer) is to reduce the latency
    // of OpenPose. E.g., if 4 cameras in stereo mode, without this, OpenPose would have to process all 4 cameras
    // with the same GPU. In this way, this work is parallelized over GPUs (1 view for each).
    // Pros: Latency highly recuded, same speed
    // Cons: Requires these extra 2 classes and proper threads for them
    template<typename TDatums>
    class WQueueAssembler : public Worker<std::shared_ptr<TDatums>>
    {
    public:
        explicit WQueueAssembler();

        virtual ~WQueueAssembler();

        void initializationOnThread();

        void work(std::shared_ptr<TDatums>& tDatums);

    private:
        std::shared_ptr<TDatums> mNextTDatums;

        DELETE_COPY(WQueueAssembler);
    };
}





// Implementation
namespace op
{
    template<typename TDatums>
    WQueueAssembler<TDatums>::WQueueAssembler()
    {
    }

    template<typename TDatums>
    WQueueAssembler<TDatums>::~WQueueAssembler()
    {
    }

    template<typename TDatums>
    void WQueueAssembler<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WQueueAssembler<TDatums>::work(std::shared_ptr<TDatums>& tDatums)
    {
        try
        {
            // Profiling speed
            const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
            // Input TDatums -> enqueue it
            if (checkNoNullNorEmpty(tDatums))
            {
                // Sanity check
                if (tDatums->size() > 1)
                    error("This function assumes that WQueueSplitter (inside WDatumProducer)"
                          " was applied in the first place, i.e., that there is only 1 element"
                          " per TDatums (size = " + std::to_string(tDatums->size()) + ").",
                          __LINE__, __FUNCTION__, __FILE__);
                auto tDatumPtr = (*tDatums)[0];
                // Single view --> Return
                if (tDatumPtr->subIdMax == 0)
                    return;
                // Multiple view --> Merge views into different TDatums (1st frame)
                if (mNextTDatums == nullptr)
                    mNextTDatums = std::make_shared<TDatums>();
                // Multiple view --> Merge views into different TDatums
                mNextTDatums->emplace_back(tDatumPtr);
                // Last view - Return frame
                if (mNextTDatums->back()->subId == mNextTDatums->back()->subIdMax)
                {
                    tDatums = mNextTDatums;
                    mNextTDatums = nullptr;
                    // Profiling speed
                    Profiler::timerEnd(profilerKey);
                    Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                    // Debugging log
                    dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                }
                // Non-last view - Return nothing
                else
                    tDatums = nullptr;
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

    extern template class WQueueAssembler<BASE_DATUMS>;
}

#endif // OPENPOSE_THREAD_W_QUEUE_ASSEMBLER_HPP
