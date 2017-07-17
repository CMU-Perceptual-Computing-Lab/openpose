#ifndef OPENPOSE_THREAD_THREAD_HPP
#define OPENPOSE_THREAD_THREAD_HPP

#include <atomic>
#include <thread>
#include <openpose/core/common.hpp>
#include <openpose/thread/subThread.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums, typename TWorker = std::shared_ptr<Worker<TDatums>>>
    class Thread
    {
    public:
        explicit Thread(const std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr = nullptr);

        // Move constructor
        Thread(Thread&& t);

        // Move assignment
        Thread& operator=(Thread&& t);

        // Destructor
        virtual ~Thread();

        void add(const std::vector<std::shared_ptr<SubThread<TDatums, TWorker>>>& subThreads);

        void add(const std::shared_ptr<SubThread<TDatums, TWorker>>& subThread);

        void exec(const std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr);

        void startInThread();

        void stopAndJoin();

        inline bool isRunning() const
        {
            return *spIsRunning;
        }

    private:
        std::shared_ptr<std::atomic<bool>> spIsRunning;
        std::vector<std::shared_ptr<SubThread<TDatums, TWorker>>> mSubThreads;
        std::thread mThread;

        void initializationOnThread();

        void threadFunction();

        void stop();

        void join();

        DELETE_COPY(Thread);
    };
}





// Implementation
namespace op
{
    template<typename TDatums, typename TWorker>
    Thread<TDatums, TWorker>::Thread(const std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr) :
        spIsRunning{(isRunningSharedPtr != nullptr ? isRunningSharedPtr : std::make_shared<std::atomic<bool>>(false))}
    {
    }

    template<typename TDatums, typename TWorker>
    Thread<TDatums, TWorker>::Thread(Thread<TDatums, TWorker>&& t) :
        spIsRunning{std::make_shared<std::atomic<bool>>(t.spIsRunning->load())}
    {
        std::swap(mSubThreads, t.mSubThreads);
        std::swap(mThread, t.mThread);
    }

    template<typename TDatums, typename TWorker>
    Thread<TDatums, TWorker>& Thread<TDatums, TWorker>::operator=(Thread<TDatums, TWorker>&& t)
    {
        std::swap(mSubThreads, t.mSubThreads);
        std::swap(mThread, t.mThread);
        spIsRunning = {std::make_shared<std::atomic<bool>>(t.spIsRunning->load())};
        return *this;
    }

    template<typename TDatums, typename TWorker>
    Thread<TDatums, TWorker>::~Thread()
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            stopAndJoin();
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker>
    void Thread<TDatums, TWorker>::add(const std::vector<std::shared_ptr<SubThread<TDatums, TWorker>>>& subThreads)
    {
        for (const auto& subThread : subThreads)
            mSubThreads.emplace_back(subThread);
    }

    template<typename TDatums, typename TWorker>
    void Thread<TDatums, TWorker>::add(const std::shared_ptr<SubThread<TDatums, TWorker>>& subThread)
    {
        add(std::vector<std::shared_ptr<SubThread<TDatums, TWorker>>>{subThread});
    }

    template<typename TDatums, typename TWorker>
    void Thread<TDatums, TWorker>::exec(const std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr)
    {
        try
        {
            stopAndJoin();
            spIsRunning = isRunningSharedPtr;
            *spIsRunning = {true};
            threadFunction();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker>
    void Thread<TDatums, TWorker>::startInThread()
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            stopAndJoin();
            *spIsRunning = {true};
            mThread = {std::thread{&Thread::threadFunction, this}};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker>
    void Thread<TDatums, TWorker>::stopAndJoin()
    {
        try
        {
            stop();
            join();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker>
    void Thread<TDatums, TWorker>::initializationOnThread()
    {
        try
        {
            for (auto& subThread : mSubThreads)
                subThread->initializationOnThread();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker>
    void Thread<TDatums, TWorker>::threadFunction()
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            initializationOnThread();

            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            while (isRunning())
            {
                bool allSubThreadsClosed = true;
                for (auto& subThread : mSubThreads)
                    allSubThreadsClosed &= !subThread->work();

                if (allSubThreadsClosed)
                {
                    log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    stop();
                    break;
                }
            }
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker>
    void Thread<TDatums, TWorker>::stop()
    {
        try
        {
            *spIsRunning = {false};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker>
    void Thread<TDatums, TWorker>::join()
    {
        try
        {
            if (mThread.joinable())
                mThread.join();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(Thread);
}

#endif // OPENPOSE_THREAD_THREAD_HPP
