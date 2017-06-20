#ifndef OPENPOSE_THREAD_THREAD_NO_QUEUE_HPP
#define OPENPOSE_THREAD_THREAD_NO_QUEUE_HPP

#include <vector>
#include "thread.hpp"
#include "worker.hpp"

namespace op
{
    template<typename TDatums, typename TWorker = std::shared_ptr<Worker<TDatums>>>
    class SubThreadNoQueue : public SubThread<TDatums, TWorker>
    {
    public:
        explicit SubThreadNoQueue(const std::vector<TWorker>& tWorkers);

        bool work();

        DELETE_COPY(SubThreadNoQueue);
    };
}





// Implementation
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/macros.hpp>
namespace op
{
    template<typename TDatums, typename TWorker>
    SubThreadNoQueue<TDatums, TWorker>::SubThreadNoQueue(const std::vector<TWorker>& tWorkers) :
        SubThread<TDatums, TWorker>{tWorkers}
    {}

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
