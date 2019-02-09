#ifndef OPENPOSE_THREAD_THREAD_MANAGER_HPP
#define OPENPOSE_THREAD_THREAD_MANAGER_HPP

#include <atomic>
#include <set> // std::multiset
#include <tuple>
#include <openpose/core/common.hpp>
#include <openpose/thread/enumClasses.hpp>
#include <openpose/thread/queue.hpp>
#include <openpose/thread/thread.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums, typename TWorker = std::shared_ptr<Worker<TDatums>>, typename TQueue = Queue<TDatums>>
    class ThreadManager
    {
    public:
        // Completely customizable case
        explicit ThreadManager(const ThreadManagerMode threadManagerMode = ThreadManagerMode::Synchronous);

        virtual ~ThreadManager();

        /**
         * It sets the maximum number of elements in the queue.
         * For maximum speed, set to a very large number, but the trade-off would be:
         *  - Latency will hugely increase.
         *  - The program might go out of RAM memory (so the computer might freeze).
         * For minimum latency while keeping an optimal speed, set to -1, that will automatically
         * detect the ideal number based on how many elements are connected to that queue.
         * @param defaultMaxSizeQueues long long element with the maximum number of elements on the queue.
         */
        void setDefaultMaxSizeQueues(const long long defaultMaxSizeQueues = -1);

        void add(const unsigned long long threadId, const std::vector<TWorker>& tWorkers,
                 const unsigned long long queueInId, const unsigned long long queueOutId);

        void add(const unsigned long long threadId, const TWorker& tWorker, const unsigned long long queueInId,
                 const unsigned long long queueOutId);

        void reset();

        void exec();

        void start();

        void stop();

        inline std::shared_ptr<std::atomic<bool>> getIsRunningSharedPtr()
        {
            return spIsRunning;
        }

        inline bool isRunning() const
        {
            return *spIsRunning;
        }

        bool tryEmplace(TDatums& tDatums);

        bool waitAndEmplace(TDatums& tDatums);

        bool tryPush(const TDatums& tDatums);

        bool waitAndPush(const TDatums& tDatums);

        bool tryPop(TDatums& tDatums);

        bool waitAndPop(TDatums& tDatums);

    private:
        const ThreadManagerMode mThreadManagerMode;
        std::shared_ptr<std::atomic<bool>> spIsRunning;
        long long mDefaultMaxSizeQueues;
        std::multiset<std::tuple<unsigned long long, std::vector<TWorker>, unsigned long long, unsigned long long>> mThreadWorkerQueues;
        std::vector<std::shared_ptr<Thread<TDatums, TWorker>>> mThreads;
        std::vector<std::shared_ptr<TQueue>> mTQueues;

        void add(const std::vector<std::tuple<unsigned long long, std::vector<TWorker>, unsigned long long, unsigned long long>>& threadWorkerQueues);

        void add(const std::vector<std::tuple<unsigned long long, TWorker, unsigned long long, unsigned long long>>& threadWorkerQueues);

        void multisetToThreads();

        void checkAndCreateEmptyThreads();

        void checkAndCreateQueues();

        DELETE_COPY(ThreadManager);
    };
}





// Implementation
#include <utility> // std::pair
#include <openpose/utilities/fastMath.hpp>
#include <openpose/thread/subThread.hpp>
#include <openpose/thread/subThreadNoQueue.hpp>
#include <openpose/thread/subThreadQueueIn.hpp>
#include <openpose/thread/subThreadQueueInOut.hpp>
#include <openpose/thread/subThreadQueueOut.hpp>
namespace op
{
    template<typename TDatums, typename TWorker, typename TQueue>
    ThreadManager<TDatums, TWorker, TQueue>::ThreadManager(const ThreadManagerMode threadManagerMode) :
        mThreadManagerMode{threadManagerMode},
        spIsRunning{std::make_shared<std::atomic<bool>>(false)},
        mDefaultMaxSizeQueues{-1ll}
    {
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    ThreadManager<TDatums, TWorker, TQueue>::~ThreadManager()
    {
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void ThreadManager<TDatums, TWorker, TQueue>::setDefaultMaxSizeQueues(const long long defaultMaxSizeQueues)
    {
        try
        {
            mDefaultMaxSizeQueues = {defaultMaxSizeQueues};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void ThreadManager<TDatums, TWorker, TQueue>::add(const unsigned long long threadId,
                                                      const std::vector<TWorker>& tWorkers,
                                                      const unsigned long long queueInId,
                                                      const unsigned long long queueOutId)
    {
        try
        {
            add({std::make_tuple(threadId, tWorkers, queueInId, queueOutId)});
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void ThreadManager<TDatums, TWorker, TQueue>::add(const unsigned long long threadId,
                                                      const TWorker& tWorker,
                                                      const unsigned long long queueInId,
                                                      const unsigned long long queueOutId)
    {
        try
        {
            add({std::make_tuple(threadId, std::vector<TWorker>{tWorker}, queueInId, queueOutId)});
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void ThreadManager<TDatums, TWorker, TQueue>::reset()
    {
        try
        {
            mThreadWorkerQueues.clear();
            mThreads.clear();
            mTQueues.clear();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void ThreadManager<TDatums, TWorker, TQueue>::exec()
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Set threads
            multisetToThreads();
            if (!mThreads.empty())
            {
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Start threads
                for (auto i = 0u; i < mThreads.size() - 1; i++)
                    mThreads.at(i)->startInThread();
                (*mThreads.rbegin())->exec(spIsRunning);
                // Stop threads - It will arrive here when the exec() command has finished
                stop();
            }
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void ThreadManager<TDatums, TWorker, TQueue>::start()
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Set threads
            multisetToThreads();
            // Start threads
            for (auto& thread : mThreads)
                thread->startInThread();
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void ThreadManager<TDatums, TWorker, TQueue>::stop()
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            for (auto& tQueue : mTQueues)
                tQueue->stop();
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            *spIsRunning = false;
            for (auto& thread : mThreads)
                thread->stopAndJoin();
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            checkWorkerErrors();
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    bool ThreadManager<TDatums, TWorker, TQueue>::tryEmplace(TDatums& tDatums)
    {
        try
        {
            if (mThreadManagerMode != ThreadManagerMode::Asynchronous
                && mThreadManagerMode != ThreadManagerMode::AsynchronousIn)
                error("Not available for this ThreadManagerMode.", __LINE__, __FUNCTION__, __FILE__);
            if (mTQueues.empty())
                error("ThreadManager already stopped or not started yet.", __LINE__, __FUNCTION__, __FILE__);
            return mTQueues[0]->tryEmplace(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    bool ThreadManager<TDatums, TWorker, TQueue>::waitAndEmplace(TDatums& tDatums)
    {
        try
        {
            if (mThreadManagerMode != ThreadManagerMode::Asynchronous
                && mThreadManagerMode != ThreadManagerMode::AsynchronousIn)
                error("Not available for this ThreadManagerMode.", __LINE__, __FUNCTION__, __FILE__);
            if (mTQueues.empty())
                error("ThreadManager already stopped or not started yet.", __LINE__, __FUNCTION__, __FILE__);
            return mTQueues[0]->waitAndEmplace(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    bool ThreadManager<TDatums, TWorker, TQueue>::tryPush(const TDatums& tDatums)
    {
        try
        {
            if (mThreadManagerMode != ThreadManagerMode::Asynchronous
                && mThreadManagerMode != ThreadManagerMode::AsynchronousIn)
                error("Not available for this ThreadManagerMode.", __LINE__, __FUNCTION__, __FILE__);
            if (mTQueues.empty())
                error("ThreadManager already stopped or not started yet.", __LINE__, __FUNCTION__, __FILE__);
            return mTQueues[0]->tryPush(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    bool ThreadManager<TDatums, TWorker, TQueue>::waitAndPush(const TDatums& tDatums)
    {
        try
        {
            if (mThreadManagerMode != ThreadManagerMode::Asynchronous
                && mThreadManagerMode != ThreadManagerMode::AsynchronousIn)
                error("Not available for this ThreadManagerMode.", __LINE__, __FUNCTION__, __FILE__);
            if (mTQueues.empty())
                error("ThreadManager already stopped or not started yet.", __LINE__, __FUNCTION__, __FILE__);
            return mTQueues[0]->waitAndPush(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    bool ThreadManager<TDatums, TWorker, TQueue>::tryPop(TDatums& tDatums)
    {
        try
        {
            if (mThreadManagerMode != ThreadManagerMode::Asynchronous
                && mThreadManagerMode != ThreadManagerMode::AsynchronousOut)
                error("Not available for this ThreadManagerMode.", __LINE__, __FUNCTION__, __FILE__);
            if (mTQueues.empty())
                error("ThreadManager already stopped or not started yet.", __LINE__, __FUNCTION__, __FILE__);
            return (*mTQueues.rbegin())->tryPop(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    bool ThreadManager<TDatums, TWorker, TQueue>::waitAndPop(TDatums& tDatums)
    {
        try
        {
            if (mThreadManagerMode != ThreadManagerMode::Asynchronous
                && mThreadManagerMode != ThreadManagerMode::AsynchronousOut)
                error("Not available for this ThreadManagerMode.", __LINE__, __FUNCTION__, __FILE__);
            if (mTQueues.empty())
                error("ThreadManager already stopped or not started yet.", __LINE__, __FUNCTION__, __FILE__);
            return (*mTQueues.rbegin())->waitAndPop(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void ThreadManager<TDatums, TWorker, TQueue>::add(const std::vector<std::tuple<unsigned long long, std::vector<TWorker>,
                                                                                   unsigned long long, unsigned long long>>& threadWorkerQueues)
    {
        try
        {
            for (const auto& threadWorkerQueue : threadWorkerQueues)
                mThreadWorkerQueues.insert(threadWorkerQueue);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void ThreadManager<TDatums, TWorker, TQueue>::add(const std::vector<std::tuple<unsigned long long, TWorker, unsigned long long,
                                                                                   unsigned long long>>& threadWorkerQueues)
    {
        try
        {
            for (const auto& threadWorkerQueue : threadWorkerQueues)
                add({std::make_tuple(std::get<0>(threadWorkerQueue),
                                     std::vector<TWorker>{std::get<1>(threadWorkerQueue)},
                                     std::get<2>(threadWorkerQueue),
                                     std::get<3>(threadWorkerQueue))});
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void ThreadManager<TDatums, TWorker, TQueue>::multisetToThreads()
    {
        try
        {
            if (!mThreadWorkerQueues.empty())
            {
                // This avoids extra std::cout if errors occur on different threads
                setMainThread();

                // Check threads
                checkAndCreateEmptyThreads();

                // Check and create queues
                checkAndCreateQueues();

                // Data
                const auto maxQueueIdSynchronous = mTQueues.size()+1;

                // Set up threads
                for (const auto& threadWorkerQueue : mThreadWorkerQueues)
                {
                    auto& thread = mThreads[std::get<0>(threadWorkerQueue)];
                    const auto& tWorkers = std::get<1>(threadWorkerQueue);
                    const auto queueIn = std::get<2>(threadWorkerQueue);
                    const auto queueOut = std::get<3>(threadWorkerQueue);
                    std::shared_ptr<SubThread<TDatums, TWorker>> subThread;
                    // If AsynchronousIn -> queue indexes are OK
                    if (mThreadManagerMode == ThreadManagerMode::Asynchronous
                        || mThreadManagerMode == ThreadManagerMode::AsynchronousIn)
                    {
                        if (mThreadManagerMode == ThreadManagerMode::AsynchronousIn
                            && queueOut == mTQueues.size())
                            subThread = {std::make_shared<SubThreadQueueIn<TDatums, TWorker, TQueue>>(
                                tWorkers, mTQueues.at(queueIn))};
                        else
                            subThread = {std::make_shared<SubThreadQueueInOut<TDatums, TWorker, TQueue>>(
                                tWorkers, mTQueues.at(queueIn), mTQueues.at(queueOut))};
                    }
                    // If !AsynchronousIn -> queue indexes - 1
                    else if (queueOut != maxQueueIdSynchronous
                        || mThreadManagerMode == ThreadManagerMode::AsynchronousOut)
                    {
                        // Queue in + out
                        if (queueIn != 0)
                            subThread = {std::make_shared<SubThreadQueueInOut<TDatums, TWorker, TQueue>>(
                                tWorkers, mTQueues.at(queueIn-1), mTQueues.at(queueOut-1))};
                        // Case queue out (first TWorker(s))
                        else
                            subThread = {std::make_shared<SubThreadQueueOut<TDatums, TWorker, TQueue>>(
                                tWorkers, mTQueues.at(queueOut-1))};
                    }
                    // Case queue in (last TWorker(s))
                    else if (queueIn != 0) // && queueOut == maxQueueIdSynchronous
                        subThread = {std::make_shared<SubThreadQueueIn<TDatums, TWorker, TQueue>>(
                            tWorkers, mTQueues.at(queueIn-1))};
                    // Case no queue
                    else // if (queueIn == 0 && queueOut == maxQueueIdSynchronous)
                        subThread = {std::make_shared<SubThreadNoQueue<TDatums, TWorker>>(tWorkers)};
                    thread->add(subThread);
                }
            }
            else
                error("Empty, no TWorker(s) added.", __LINE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void ThreadManager<TDatums, TWorker, TQueue>::checkAndCreateEmptyThreads()
    {
        try
        {
            // Check all thread ids from 0-maxThreadId are present
            const auto maxThreadId = std::get<0>(*mThreadWorkerQueues.crbegin());
            auto previousThreadId = std::get<0>(*mThreadWorkerQueues.cbegin());
            for (const auto& threadWorkerQueue : mThreadWorkerQueues)
            {
                const auto currentThreadId = std::get<0>(threadWorkerQueue);
                if (currentThreadId - previousThreadId > 1)
                    error("Missing thread id " + std::to_string(currentThreadId) + " of "
                          + std::to_string(maxThreadId) + ".", __LINE__, __FUNCTION__, __FILE__);
                previousThreadId = currentThreadId;
            }

            // Create Threads
            // #threads = maxThreadId+1
            mThreads.resize(maxThreadId);
            for (auto& thread : mThreads)
                thread = std::make_shared<Thread<TDatums, TWorker>>();
            mThreads.emplace_back(std::make_shared<Thread<TDatums, TWorker>>(spIsRunning));
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void ThreadManager<TDatums, TWorker, TQueue>::checkAndCreateQueues()
    {
        try
        {
            if (!mThreadWorkerQueues.empty())
            {
                // Get max queue id to get queue size
                auto maxQueueId = std::get<3>(*mThreadWorkerQueues.cbegin());
                for (const auto& threadWorkerQueue : mThreadWorkerQueues)
                    maxQueueId = fastMax(
                        maxQueueId, fastMax(std::get<2>(threadWorkerQueue), std::get<3>(threadWorkerQueue)));

                // Check each queue id has at least a worker that uses it as input and another one as output.
                // Special cases:
                std::vector<std::pair<bool, bool>> usedQueueIds(maxQueueId+1, {false, false});
                for (const auto& threadWorkerQueue : mThreadWorkerQueues)
                {
                    usedQueueIds.at(std::get<2>(threadWorkerQueue)).first = true;
                    usedQueueIds.at(std::get<3>(threadWorkerQueue)).second = true;
                }
                // Id 0 must only needs a worker using it as input.
                usedQueueIds.begin()->second = true;
                // Id maxQueueId only needs a worker using it as output.
                usedQueueIds.rbegin()->first = true;
                // Error if missing queue id
                for (auto i = 0ull ; i < usedQueueIds.size() ; i++)
                {
                    if (!usedQueueIds[i].first)
                        error("Missing queue id " + std::to_string(i) + " (of "
                              + std::to_string(maxQueueId) + ") as input.", __LINE__, __FUNCTION__, __FILE__);
                    if (!usedQueueIds[i].second)
                        error("Missing queue id " + std::to_string(i) + " (of "
                              + std::to_string(maxQueueId) + ") as output.", __LINE__, __FUNCTION__, __FILE__);
                }

                // Create Queues
                if (mThreadManagerMode == ThreadManagerMode::Asynchronous)
                    mTQueues.resize(maxQueueId+1);   // First and last one are queues
                else if (mThreadManagerMode == ThreadManagerMode::Synchronous)
                    mTQueues.resize(maxQueueId-1);   // First and last one are not actually queues
                else if (mThreadManagerMode == ThreadManagerMode::AsynchronousIn
                         || mThreadManagerMode == ThreadManagerMode::AsynchronousOut)
                    mTQueues.resize(maxQueueId);   // First or last one is queue
                else
                    error("Unknown ThreadManagerMode", __LINE__, __FUNCTION__, __FILE__);
                for (auto& tQueue : mTQueues)
                    tQueue = std::make_shared<TQueue>(mDefaultMaxSizeQueues);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(ThreadManager);
}

#endif // OPENPOSE_THREAD_THREAD_MANAGER_HPP
