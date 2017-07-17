#ifndef OPENPOSE_THREAD_THREAD_QUEUE_IN_OUT_HPP
#define OPENPOSE_THREAD_THREAD_QUEUE_IN_OUT_HPP

#include <openpose/core/common.hpp>
#include <openpose/thread/queue.hpp>
#include <openpose/thread/thread.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums, typename TWorker = std::shared_ptr<Worker<TDatums>>, typename TQueue = Queue<TDatums>>
    class SubThreadQueueInOut : public SubThread<TDatums, TWorker>
    {
    public:
        SubThreadQueueInOut(const std::vector<TWorker>& tWorkers, const std::shared_ptr<TQueue>& tQueueIn, const std::shared_ptr<TQueue>& tQueueOut);

        bool work();

    private:
        std::shared_ptr<TQueue> spTQueueIn;
        std::shared_ptr<TQueue> spTQueueOut;

        DELETE_COPY(SubThreadQueueInOut);
    };
}





// Implementation
namespace op
{
    template<typename TDatums, typename TWorker, typename TQueue>
    SubThreadQueueInOut<TDatums, TWorker, TQueue>::SubThreadQueueInOut(const std::vector<TWorker>& tWorkers, const std::shared_ptr<TQueue>& tQueueIn,
                                                                       const std::shared_ptr<TQueue>& tQueueOut) :
        SubThread<TDatums, TWorker>{tWorkers},
        spTQueueIn{tQueueIn},
        spTQueueOut{tQueueOut}
    {
        // spTQueueIn->addPopper();
        spTQueueOut->addPusher();
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    bool SubThreadQueueInOut<TDatums, TWorker, TQueue>::work()
    {
        try
        {
            // If output queue is closed -> close input queue
            if (!spTQueueOut->isRunning())
            {
                spTQueueIn->stop();
                return false;
            }
            // If output queue running -> normal operation
            else
            {
                // Pop TDatums
                TDatums tDatums;
                bool workersAreRunning = spTQueueIn->tryPop(tDatums);
                // Check queue not stopped
                if (!workersAreRunning)
                    workersAreRunning = spTQueueIn->isRunning();
                // Process TDatums
                workersAreRunning = this->workTWorkers(tDatums, workersAreRunning);
                // Push/emplace tDatums if successfully processed
                if (workersAreRunning)
                {
                    if (tDatums != nullptr)
                        spTQueueOut->waitAndEmplace(tDatums);
                }
                // Close both queues otherwise
                else
                {
                    spTQueueIn->stop();
                    spTQueueOut->stopPusher();
                }
                return workersAreRunning;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            spTQueueIn->stop();
            spTQueueOut->stop();
            return false;
        }
    }

    COMPILE_TEMPLATE_DATUM(SubThreadQueueInOut);
}

#endif // OPENPOSE_THREAD_THREAD_QUEUE_IN_OUT_HPP
