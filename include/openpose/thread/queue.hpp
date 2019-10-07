#ifndef OPENPOSE_THREAD_QUEUE_HPP
#define OPENPOSE_THREAD_QUEUE_HPP

#include <queue> // std::queue
#include <openpose/core/common.hpp>
#include <openpose/thread/queueBase.hpp>

namespace op
{
    template<typename TDatums, typename TQueue = std::queue<TDatums>>
    class Queue : public QueueBase<TDatums, TQueue>
    {
    public:
        explicit Queue(const long long maxSize);

        virtual ~Queue();

        TDatums front() const;

    private:
        bool pop(TDatums& tDatums);

        DELETE_COPY(Queue);
    };
}





// Implementation
#include <type_traits> // std::is_same
namespace op
{
    template<typename TDatums, typename TQueue>
    Queue<TDatums, TQueue>::Queue(const long long maxSize) :
        QueueBase<TDatums, TQueue>{maxSize}
    {
        // Check TDatums = underlying value type of TQueue
        typedef typename TQueue::value_type underlyingValueType;
        static_assert(std::is_same<TDatums, underlyingValueType>::value,
                      "Error: The type of the queue must be the same as the type of the container");
    }

    template<typename TDatums, typename TQueue>
    Queue<TDatums, TQueue>::~Queue()
    {
    }

    template<typename TDatums, typename TQueue>
    TDatums Queue<TDatums, TQueue>::front() const
    {
        try
        {
            const std::lock_guard<std::mutex> lock{this->mMutex};
            return this->mTQueue.front();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return TDatums{};
        }
    }

    template<typename TDatums, typename TQueue>
    bool Queue<TDatums, TQueue>::pop(TDatums& tDatums)
    {
        try
        {
            if (this->mPopIsStopped || this->mTQueue.empty())
                return false;

            tDatums = {std::move(this->mTQueue.front())};
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

    COMPILE_TEMPLATE_DATUM(Queue);
}

#endif // OPENPOSE_THREAD_QUEUE_HPP
