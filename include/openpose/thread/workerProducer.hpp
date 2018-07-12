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

        inline void work(TDatums& tDatums)
        {
            tDatums = std::move(workProducer());
        }

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

    COMPILE_TEMPLATE_DATUM(WorkerProducer);
}

#endif // OPENPOSE_THREAD_WORKER_PRODUCER_HPP
