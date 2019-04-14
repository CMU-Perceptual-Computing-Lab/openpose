#include <openpose/thread/headers.hpp>

namespace op
{
    // Queues
    DEFINE_TEMPLATE_DATUM(PriorityQueue);
    DEFINE_TEMPLATE_DATUM(Queue);
    template class OP_API QueueBase<BASE_DATUMS_SH, std::queue<BASE_DATUMS_SH>>;
    template class OP_API QueueBase<
        BASE_DATUMS_SH,
        std::priority_queue<BASE_DATUMS_SH, std::vector<BASE_DATUMS_SH>,
        std::greater<BASE_DATUMS_SH>>>;
    // Subthread
    DEFINE_TEMPLATE_DATUM(SubThread);
    DEFINE_TEMPLATE_DATUM(SubThreadNoQueue);
    DEFINE_TEMPLATE_DATUM(SubThreadQueueIn);
    DEFINE_TEMPLATE_DATUM(SubThreadQueueInOut);
    DEFINE_TEMPLATE_DATUM(SubThreadQueueOut);
    // Thread
    DEFINE_TEMPLATE_DATUM(Thread);
    DEFINE_TEMPLATE_DATUM(ThreadManager);
    // Main workers
    DEFINE_TEMPLATE_DATUM(Worker);
    DEFINE_TEMPLATE_DATUM(WorkerConsumer);
    DEFINE_TEMPLATE_DATUM(WorkerProducer);
    // W-classes
    DEFINE_TEMPLATE_DATUM(WFpsMax);
    DEFINE_TEMPLATE_DATUM(WIdGenerator);
    template class OP_API WQueueAssembler<BASE_DATUMS>;
    DEFINE_TEMPLATE_DATUM(WQueueOrderer);
}
