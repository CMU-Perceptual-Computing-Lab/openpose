#include <openpose/thread/headers.hpp>

namespace op
{
    // Queues
    DEFINE_TEMPLATE_DATUM(PriorityQueue);
    DEFINE_TEMPLATE_DATUM(Queue);
    template class OP_API QueueBase<DATUM_BASE, std::queue<DATUM_BASE>>;
    template class OP_API QueueBase<DATUM_BASE, std::priority_queue<DATUM_BASE, std::vector<DATUM_BASE>, std::greater<DATUM_BASE>>>;
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
    template class OP_API WQueueAssembler<DATUM_BASE, DATUM_BASE_NO_PTR>;
    DEFINE_TEMPLATE_DATUM(WQueueOrderer);
}
