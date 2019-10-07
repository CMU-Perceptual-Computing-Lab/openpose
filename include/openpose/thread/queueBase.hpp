#ifndef OPENPOSE_THREAD_QUEUE_BASE_HPP
#define OPENPOSE_THREAD_QUEUE_BASE_HPP

#include <condition_variable>
#include <mutex>
#include <queue> // std::queue & std::priority_queue
#include <openpose/core/common.hpp>

namespace op
{
    template<typename TDatums, typename TQueue>
    class QueueBase
    {
    public:
        explicit QueueBase(const long long maxSize = -1);

        virtual ~QueueBase();

        bool forceEmplace(TDatums& tDatums);

        bool tryEmplace(TDatums& tDatums);

        bool waitAndEmplace(TDatums& tDatums);

        bool forcePush(const TDatums& tDatums);

        bool tryPush(const TDatums& tDatums);

        bool waitAndPush(const TDatums& tDatums);

        bool tryPop(TDatums& tDatums);

        bool tryPop();

        bool waitAndPop(TDatums& tDatums);

        bool waitAndPop();

        bool empty() const;

        void stop();

        void stopPusher();

        void addPopper();

        void addPusher();

        bool isRunning() const;

        bool isFull() const;

        size_t size() const;

        void clear();

        virtual TDatums front() const = 0;

    protected:
        mutable std::mutex mMutex;
        long long mPoppers;
        long long mPushers;
        long long mMaxPoppersPushers;
        bool mPopIsStopped;
        bool mPushIsStopped;
        std::condition_variable mConditionVariable;
        TQueue mTQueue;

        virtual bool pop(TDatums& tDatums) = 0;

        unsigned long long getMaxSize() const;

    private:
        const long long mMaxSize;

        bool emplace(TDatums& tDatums);

        bool push(const TDatums& tDatums);

        bool pop();

        void updateMaxPoppersPushers();

        DELETE_COPY(QueueBase);
    };
}





// Implementation
#include <openpose/core/datum.hpp>
#include <openpose/utilities/fastMath.hpp>
namespace op
{
    template<typename TDatums, typename TQueue>
    QueueBase<TDatums, TQueue>::QueueBase(const long long maxSize) :
        mPoppers{0ll},
        mPushers{0ll},
        mPopIsStopped{false},
        mPushIsStopped{false},
        mMaxSize{maxSize}
    {
    }

    // Virutal destructor
    template<typename TDatums, typename TQueue>
    QueueBase<TDatums, TQueue>::~QueueBase()
    {
        try
        {
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            stop();
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TQueue>
    bool QueueBase<TDatums, TQueue>::forceEmplace(TDatums& tDatums)
    {
        try
        {
            const std::lock_guard<std::mutex> lock{mMutex};
            if (mTQueue.size() >= getMaxSize())
                mTQueue.pop();
            return emplace(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TQueue>
    bool QueueBase<TDatums, TQueue>::tryEmplace(TDatums& tDatums)
    {
        try
        {
            const std::lock_guard<std::mutex> lock{mMutex};
            if (mTQueue.size() >= getMaxSize())
                return false;
            return emplace(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TQueue>
    bool QueueBase<TDatums, TQueue>::waitAndEmplace(TDatums& tDatums)
    {
        try
        {
            std::unique_lock<std::mutex> lock{mMutex};
            mConditionVariable.wait(lock, [this]{return mTQueue.size() < getMaxSize() || mPushIsStopped; });
            return emplace(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TQueue>
    bool QueueBase<TDatums, TQueue>::forcePush(const TDatums& tDatums)
    {
        try
        {
            const std::lock_guard<std::mutex> lock{mMutex};
            if (mTQueue.size() >= getMaxSize())
                mTQueue.pop();
            return push(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TQueue>
    bool QueueBase<TDatums, TQueue>::tryPush(const TDatums& tDatums)
    {
        try
        {
            const std::lock_guard<std::mutex> lock{mMutex};
            if (mTQueue.size() >= getMaxSize())
                return false;
            return push(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TQueue>
    bool QueueBase<TDatums, TQueue>::waitAndPush(const TDatums& tDatums)
    {
        try
        {
            std::unique_lock<std::mutex> lock{mMutex};
            mConditionVariable.wait(lock, [this]{return mTQueue.size() < getMaxSize() || mPushIsStopped; });
            return push(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TQueue>
    bool QueueBase<TDatums, TQueue>::tryPop(TDatums& tDatums)
    {
        try
        {
            const std::lock_guard<std::mutex> lock{mMutex};
            return pop(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TQueue>
    bool QueueBase<TDatums, TQueue>::tryPop()
    {
        try
        {
            const std::lock_guard<std::mutex> lock{mMutex};
            return pop();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TQueue>
    bool QueueBase<TDatums, TQueue>::waitAndPop(TDatums& tDatums)
    {
        try
        {
            std::unique_lock<std::mutex> lock{mMutex};
            mConditionVariable.wait(lock, [this]{return !mTQueue.empty() || mPopIsStopped; });
            return pop(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TQueue>
    bool QueueBase<TDatums, TQueue>::waitAndPop()
    {
        try
        {
            std::unique_lock<std::mutex> lock{mMutex};
            mConditionVariable.wait(lock, [this]{return !mTQueue.empty() || mPopIsStopped; });
            return pop();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TQueue>
    bool QueueBase<TDatums, TQueue>::empty() const
    {
        try
        {
            const std::lock_guard<std::mutex> lock{mMutex};
            return mTQueue.empty();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TQueue>
    void QueueBase<TDatums, TQueue>::stop()
    {
        try
        {
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            const std::lock_guard<std::mutex> lock{mMutex};
            mPopIsStopped = {true};
            mPushIsStopped = {true};
            while (!mTQueue.empty())
                mTQueue.pop();
            mConditionVariable.notify_all();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TQueue>
    void QueueBase<TDatums, TQueue>::stopPusher()
    {
        try
        {
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            const std::lock_guard<std::mutex> lock{mMutex};
            mPushers--;
            if (mPushers == 0)
            {
                mPushIsStopped = {true};
                if (mTQueue.empty())
                    mPopIsStopped = {true};
                mConditionVariable.notify_all();
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TQueue>
    void QueueBase<TDatums, TQueue>::addPopper()
    {
        try
        {
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            const std::lock_guard<std::mutex> lock{mMutex};
            mPoppers++;
            updateMaxPoppersPushers();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TQueue>
    void QueueBase<TDatums, TQueue>::addPusher()
    {
        try
        {
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            const std::lock_guard<std::mutex> lock{mMutex};
            mPushers++;
            updateMaxPoppersPushers();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TQueue>
    bool QueueBase<TDatums, TQueue>::isRunning() const
    {
        try
        {
            const std::lock_guard<std::mutex> lock{mMutex};
            return !(mPushIsStopped && (mPopIsStopped || mTQueue.empty()));
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return true;
        }
    }

    template<typename TDatums, typename TQueue>
    bool QueueBase<TDatums, TQueue>::isFull() const
    {
        try
        {
            // No mutex required because the size() and getMaxSize() are already thread-safe
            return size() == getMaxSize();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TQueue>
    size_t QueueBase<TDatums, TQueue>::size() const
    {
        try
        {
            const std::lock_guard<std::mutex> lock{mMutex};
            return mTQueue.size();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0;
        }
    }

    template<typename TDatums, typename TQueue>
    void QueueBase<TDatums, TQueue>::clear()
    {
        try
        {
            const std::lock_guard<std::mutex> lock{mMutex};
            while (!mTQueue.empty())
                mTQueue.pop();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TQueue>
    unsigned long long QueueBase<TDatums, TQueue>::getMaxSize() const
    {
        try
        {
            return (mMaxSize > 0 ? mMaxSize : fastMax(1ll, mMaxPoppersPushers));
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TQueue>
    bool QueueBase<TDatums, TQueue>::emplace(TDatums& tDatums)
    {
        try
        {
            if (mPushIsStopped)
                return false;

            mTQueue.emplace(tDatums);
            mConditionVariable.notify_all();
            return true;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TQueue>
    bool QueueBase<TDatums, TQueue>::push(const TDatums& tDatums)
    {
        try
        {
            if (mPushIsStopped)
                return false;

            mTQueue.push(tDatums);
            mConditionVariable.notify_all();
            return true;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TQueue>
    bool QueueBase<TDatums, TQueue>::pop()
    {
        try
        {
            if (mPopIsStopped || mTQueue.empty())
                return false;

            mTQueue.pop();
            mConditionVariable.notify_all();
            return true;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TQueue>
    void QueueBase<TDatums, TQueue>::updateMaxPoppersPushers()
    {
        try
        {
            mMaxPoppersPushers = fastMax(mPoppers, mPushers);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    extern template class QueueBase<BASE_DATUMS_SH, std::queue<BASE_DATUMS_SH>>;
    extern template class QueueBase<
        BASE_DATUMS_SH,
        std::priority_queue<BASE_DATUMS_SH, std::vector<BASE_DATUMS_SH>,
        std::greater<BASE_DATUMS_SH>>>;
}

#endif // OPENPOSE_THREAD_QUEUE_BASE_HPP
