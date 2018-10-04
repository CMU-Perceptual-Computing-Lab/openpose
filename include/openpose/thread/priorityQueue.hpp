#ifndef OPENPOSE_THREAD_PRIORITY_QUEUE_HPP
#define OPENPOSE_THREAD_PRIORITY_QUEUE_HPP 

#include <queue> // std::priority_queue
#include <openpose/core/common.hpp>
#include <openpose/thread/queueBase.hpp>

namespace op
{
    template<typename TDatums, typename TQueue = std::priority_queue<TDatums, std::vector<TDatums>, std::greater<TDatums>>>
    class PriorityQueue : public QueueBase<TDatums, TQueue>
    {
    public:
        explicit PriorityQueue(const long long maxSize = 256);

        virtual ~PriorityQueue();

        TDatums front() const;

    private:
        bool pop(TDatums& tDatums);

        DELETE_COPY(PriorityQueue);
    };
}





// Implementation
#include <type_traits> // std::is_same
namespace op
{
    template<typename TDatums, typename TQueue>
    PriorityQueue<TDatums, TQueue>::PriorityQueue(const long long maxSize) :
        QueueBase<TDatums, TQueue>{maxSize}
    {
        // Check TDatums = underlying value type of TQueue
        typedef typename TQueue::value_type underlyingValueType;
        static_assert(std::is_same<TDatums, underlyingValueType>::value,
                      "Error: The type of the queue must be the same as the type of the container");
    }

    template<typename TDatums, typename TQueue>
    PriorityQueue<TDatums, TQueue>::~PriorityQueue()
    {
    }

    template<typename TDatums, typename TQueue>
    TDatums PriorityQueue<TDatums, TQueue>::front() const
    {
        try
        {
            const std::lock_guard<std::mutex> lock{this->mMutex};
            return this->mTQueue.top();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return TDatums{};
        }
    }

    template<typename TDatums, typename TQueue>
    bool PriorityQueue<TDatums, TQueue>::pop(TDatums& tDatums)
    {
        try
        {
            if (this->mPopIsStopped || this->mTQueue.empty())
                return false;

            tDatums = {std::move(this->mTQueue.top())};
            this->mTQueue.pop();
            this->mConditionVariable.notify_one();
            return true;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    COMPILE_TEMPLATE_DATUM(PriorityQueue);
}

#endif // OPENPOSE_THREAD_PRIORITY_QUEUE_HPP
