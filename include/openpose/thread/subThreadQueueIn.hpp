#ifndef OPENPOSE_THREAD_THREAD_QUEUE_IN_HPP
#define OPENPOSE_THREAD_THREAD_QUEUE_IN_HPP

#include <openpose/core/common.hpp>
#include <openpose/thread/queue.hpp>
#include <openpose/thread/thread.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums, typename TWorker = std::shared_ptr<Worker<TDatums>>, typename TQueue = Queue<TDatums>>
    class SubThreadQueueIn : public SubThread<TDatums, TWorker>
    {
    public:
        SubThreadQueueIn(const std::vector<TWorker>& tWorkers, const std::shared_ptr<TQueue>& tQueueIn);

        bool work();

    private:
        std::shared_ptr<TQueue> spTQueueIn;

        DELETE_COPY(SubThreadQueueIn);
    };
}





// Implementation
namespace op
{
    template<typename TDatums, typename TWorker, typename TQueue>
    SubThreadQueueIn<TDatums, TWorker, TQueue>::SubThreadQueueIn(const std::vector<TWorker>& tWorkers, const std::shared_ptr<TQueue>& tQueueIn) :
        SubThread<TDatums, TWorker>{tWorkers},
        spTQueueIn{tQueueIn}
    {
        // spTQueueIn->addPopper();
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    bool SubThreadQueueIn<TDatums, TWorker, TQueue>::work()
    {
        try
        {
            // Pop TDatums
            TDatums tDatums;
            bool queueIsRunning = spTQueueIn->tryPop(tDatums);
            // Check queue not empty
            if (!queueIsRunning)
                queueIsRunning = spTQueueIn->isRunning();
            // Process TDatums
            const auto workersAreRunning = this->workTWorkers(tDatums, queueIsRunning);
            // Close queue input if all workers closed
            if (!workersAreRunning)
                spTQueueIn->stop();
            return workersAreRunning;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            spTQueueIn->stop();
            return false;
        }
    }

    COMPILE_TEMPLATE_DATUM(SubThreadQueueIn);
}

#endif // OPENPOSE_THREAD_THREAD_QUEUE_IN_HPP
