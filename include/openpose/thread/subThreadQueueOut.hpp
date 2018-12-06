#ifndef OPENPOSE_THREAD_THREAD_QUEUE_OUT_HPP
#define OPENPOSE_THREAD_THREAD_QUEUE_OUT_HPP

#include <openpose/core/common.hpp>
#include <openpose/thread/queue.hpp>
#include <openpose/thread/thread.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums, typename TWorker = std::shared_ptr<Worker<TDatums>>, typename TQueue = Queue<TDatums>>
    class SubThreadQueueOut : public SubThread<TDatums, TWorker>
    {
    public:
        SubThreadQueueOut(const std::vector<TWorker>& tWorkers, const std::shared_ptr<TQueue>& tQueueOut);

        virtual ~SubThreadQueueOut();

        bool work();

    private:
        std::shared_ptr<TQueue> spTQueueOut;

        DELETE_COPY(SubThreadQueueOut);
    };
}





// Implementation
namespace op
{
    template<typename TDatums, typename TWorker, typename TQueue>
    SubThreadQueueOut<TDatums, TWorker, TQueue>::SubThreadQueueOut(const std::vector<TWorker>& tWorkers,
                   const std::shared_ptr<TQueue>& tQueueOut) :
        SubThread<TDatums, TWorker>{tWorkers},
        spTQueueOut{tQueueOut}
    {
        spTQueueOut->addPusher();
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    SubThreadQueueOut<TDatums, TWorker, TQueue>::~SubThreadQueueOut()
    {
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    bool SubThreadQueueOut<TDatums, TWorker, TQueue>::work()
    {
        try
        {
            // If output queue is closed -> close input queue
            if (!spTQueueOut->isRunning())
                return false;
            else
            {
                // Don't work until next queue is not full
                // This reduces latency to half
                if (!spTQueueOut->isFull())
                {
                    // Process TDatums
                    TDatums tDatums;
                    const auto workersAreRunning = this->workTWorkers(tDatums, true);
                    // Push/emplace tDatums if successfully processed
                    if (workersAreRunning)
                    {
                        if (tDatums != nullptr)
                            spTQueueOut->waitAndEmplace(tDatums);
                    }
                    // Close queue otherwise
                    else
                        spTQueueOut->stopPusher();
                    return workersAreRunning;
                }
                else
                {
                    std::this_thread::sleep_for(std::chrono::microseconds{100});
                    return true;
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            spTQueueOut->stop();
            return false;
        }
    }

    COMPILE_TEMPLATE_DATUM(SubThreadQueueOut);
}

#endif // OPENPOSE_THREAD_THREAD_QUEUE_OUT_HPP
