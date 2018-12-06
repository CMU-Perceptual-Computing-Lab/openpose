#ifndef OPENPOSE_THREAD_THREAD_NO_QUEUE_HPP
#define OPENPOSE_THREAD_THREAD_NO_QUEUE_HPP

#include <openpose/core/common.hpp>
#include <openpose/thread/thread.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums, typename TWorker = std::shared_ptr<Worker<TDatums>>>
    class SubThreadNoQueue : public SubThread<TDatums, TWorker>
    {
    public:
        explicit SubThreadNoQueue(const std::vector<TWorker>& tWorkers);

        virtual ~SubThreadNoQueue();

        bool work();

        DELETE_COPY(SubThreadNoQueue);
    };
}





// Implementation
namespace op
{
    template<typename TDatums, typename TWorker>
    SubThreadNoQueue<TDatums, TWorker>::SubThreadNoQueue(const std::vector<TWorker>& tWorkers) :
        SubThread<TDatums, TWorker>{tWorkers}
    {
    }

    template<typename TDatums, typename TWorker>
    SubThreadNoQueue<TDatums, TWorker>::~SubThreadNoQueue()
    {
    }

    template<typename TDatums, typename TWorker>
    bool SubThreadNoQueue<TDatums, TWorker>::work()
    {
        try
        {
            TDatums tDatums;
            return this->workTWorkers(tDatums, true);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    COMPILE_TEMPLATE_DATUM(SubThreadNoQueue);
}

#endif // OPENPOSE_THREAD_THREAD_NO_QUEUE_HPP
