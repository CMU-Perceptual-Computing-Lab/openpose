#ifndef OPENPOSE_THREAD_WORKER_CONSUMER_HPP
#define OPENPOSE_THREAD_WORKER_CONSUMER_HPP

#include "worker.hpp"

namespace op
{
    template<typename TDatums>
    class WorkerConsumer : public Worker<TDatums>
    {
    public:
        virtual ~WorkerConsumer();

        inline void work(TDatums& tDatums)
        {
            workConsumer(tDatums);
        }

    protected:
        virtual void workConsumer(const TDatums& tDatums) = 0;
    };
}





// Implementation
#include <openpose/utilities/macros.hpp>
namespace op
{
    template<typename TDatums>
    WorkerConsumer<TDatums>::~WorkerConsumer()
    {
    }

    COMPILE_TEMPLATE_DATUM(WorkerConsumer);
}

#endif // OPENPOSE_THREAD_WORKER_CONSUMER_HPP
