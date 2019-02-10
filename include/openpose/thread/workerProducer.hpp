#ifndef OPENPOSE_THREAD_WORKER_PRODUCER_HPP
#define OPENPOSE_THREAD_WORKER_PRODUCER_HPP

#include <openpose/core/common.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WorkerProducer : public Worker<TDatums>
    {
    public:
        virtual ~WorkerProducer();

        void work(TDatums& tDatums);

    protected:
        virtual TDatums workProducer() = 0;
    };
}





// Implementation
namespace op
{
    template<typename TDatums>
    WorkerProducer<TDatums>::~WorkerProducer()
    {
    }

    template<typename TDatums>
    void WorkerProducer<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            tDatums = std::move(workProducer());
        }
        catch (const std::exception& e)
        {
            this->stop();
            errorWorker(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WorkerProducer);
}

#endif // OPENPOSE_THREAD_WORKER_PRODUCER_HPP
