#ifndef OPENPOSE_THREAD_WORKER_CONSUMER_HPP
#define OPENPOSE_THREAD_WORKER_CONSUMER_HPP

#include <openpose/core/common.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WorkerConsumer : public Worker<TDatums>
    {
    public:
        virtual ~WorkerConsumer();

        void work(TDatums& tDatums);

    protected:
        virtual void workConsumer(const TDatums& tDatums) = 0;
    };
}





// Implementation
namespace op
{
    template<typename TDatums>
    WorkerConsumer<TDatums>::~WorkerConsumer()
    {
    }

    template<typename TDatums>
    void WorkerConsumer<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            workConsumer(tDatums);
        }
        catch (const std::exception& e)
        {
            this->stop();
            errorWorker(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WorkerConsumer);
}

#endif // OPENPOSE_THREAD_WORKER_CONSUMER_HPP
