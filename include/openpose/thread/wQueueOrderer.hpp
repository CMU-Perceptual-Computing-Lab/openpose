#ifndef OPENPOSE_THREAD_W_QUEUE_ORDERER_HPP
#define OPENPOSE_THREAD_W_QUEUE_ORDERER_HPP

#include <queue> // std::priority_queue
#include <openpose/core/common.hpp>
#include <openpose/thread/worker.hpp>
#include <openpose/utilities/pointerContainer.hpp>

namespace op
{
    template<typename TDatums>
    class WQueueOrderer : public Worker<TDatums>
    {
    public:
        explicit WQueueOrderer(const unsigned int maxBufferSize = 64u);

        void initializationOnThread();

        void work(TDatums& tDatums);

        void tryStop();

    private:
        const unsigned int mMaxBufferSize;
        bool mStopWhenEmpty;
        unsigned long long mNextExpectedId;
        std::priority_queue<TDatums, std::vector<TDatums>, PointerContainerGreater<TDatums>> mPriorityQueueBuffer;

        DELETE_COPY(WQueueOrderer);
    };
}





// Implementation
#include <chrono>
#include <thread>
namespace op
{
    template<typename TDatums>
    WQueueOrderer<TDatums>::WQueueOrderer(const unsigned int maxBufferSize) :
        mMaxBufferSize{maxBufferSize},
        mStopWhenEmpty{false},
        mNextExpectedId{0}
    {
    }

    template<typename TDatums>
    void WQueueOrderer<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WQueueOrderer<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            // Profiling speed
            const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
            bool profileSpeed = (tDatums != nullptr);
            // Input TDatum -> enqueue or return it back
            if (checkNoNullNorEmpty(tDatums))
            {
                // T* to T
                auto& tDatumsNoPtr = *tDatums;
                // tDatums is the next expected, update counter
                if (tDatumsNoPtr[0].id == mNextExpectedId)
                    mNextExpectedId++;
                // Else push it to our buffered queue
                else
                {
                    // Enqueue current tDatums
                    mPriorityQueueBuffer.emplace(tDatums);
                    tDatums = nullptr;
                    // Else if buffer full -> remove one tDatums
                    if (mPriorityQueueBuffer.size() > mMaxBufferSize)
                    {
                        tDatums = mPriorityQueueBuffer.top();
                        mPriorityQueueBuffer.pop();
                    }
                }
            }
            // If input TDatum enqueued -> check if previously enqueued next desired frame and pop it
            if (!checkNoNullNorEmpty(tDatums))
            {
                // Retrieve frame if next is desired frame or if we want to stop this worker
                if (!mPriorityQueueBuffer.empty()   &&   ((*mPriorityQueueBuffer.top())[0].id == mNextExpectedId || mStopWhenEmpty))
                {
                    tDatums = { mPriorityQueueBuffer.top() };
                    mPriorityQueueBuffer.pop();
                }
            }
            // If TDatum ready to be returned -> updated next expected id
            if (checkNoNullNorEmpty(tDatums))
            {
                const auto& tDatumsNoPtr = *tDatums;
                mNextExpectedId = tDatumsNoPtr[0].id + 1;
            }
            // Sleep if no new tDatums to either pop
            if (!checkNoNullNorEmpty(tDatums) && mPriorityQueueBuffer.size() < mMaxBufferSize / 2u)
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
            // If TDatum popped and/or pushed
            if (profileSpeed || tDatums != nullptr)
            {
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

    template<typename TDatums>
    void WQueueOrderer<TDatums>::tryStop()
    {
        try
        {
            // Close if all frames were retrieved from the queue
            if (mPriorityQueueBuffer.empty())
                this->stop();
            mStopWhenEmpty = true;

        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WQueueOrderer);
}

#endif // OPENPOSE_THREAD_W_QUEUE_ORDERER_HPP
