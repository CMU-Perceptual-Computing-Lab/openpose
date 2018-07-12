#ifndef OPENPOSE_THREAD_SUB_THREAD_HPP
#define OPENPOSE_THREAD_SUB_THREAD_HPP

#include <openpose/core/common.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums, typename TWorker = std::shared_ptr<Worker<TDatums>>>
    class SubThread
    {
    public:
        explicit SubThread(const std::vector<TWorker>& tWorkers);

        // Destructor
        virtual ~SubThread();

        void initializationOnThread();

        virtual bool work() = 0;

    protected:
        inline size_t getTWorkersSize() const
        {
            return mTWorkers.size();
        }

        bool workTWorkers(TDatums& tDatums, const bool inputIsRunning);

    private:
        std::vector<TWorker> mTWorkers;

        DELETE_COPY(SubThread);
    };
}





// Implementation
namespace op
{
    template<typename TDatums, typename TWorker>
    SubThread<TDatums, TWorker>::SubThread(const std::vector<TWorker>& tWorkers) :
        mTWorkers{tWorkers}
    {
    }

    template<typename TDatums, typename TWorker>
    SubThread<TDatums, TWorker>::~SubThread()
    {
    }

    template<typename TDatums, typename TWorker>
    bool SubThread<TDatums, TWorker>::workTWorkers(TDatums& tDatums, const bool inputIsRunning)
    {
        try
        {
            // If !inputIsRunning -> try to close TWorkers
            if (!inputIsRunning)
            {
                for (auto& tWorkers : mTWorkers)
                {
                    tWorkers->tryStop();
                    if (tWorkers->isRunning())
                        break;
                }
            }

            // If (at least) last TWorker still working -> make TWorkers work
            if ((*mTWorkers.crbegin())->isRunning())
            {
                // Iterate over all workers and check whether some of them stopped
                auto allRunning = true;
                auto lastOneStopped = false;
                for (auto& worker : mTWorkers)
                {
                    if (lastOneStopped)
                        worker->tryStop();

                    if (!worker->checkAndWork(tDatums))
                    {
                        allRunning = false;
                        lastOneStopped = true;
                    }
                    else
                        lastOneStopped = false;
                }

                if (allRunning)
                    return true;
                else
                {
                    // If last one still running -> try to stop workers
                    // If last one stopped -> return false
                    auto lastRunning = (*mTWorkers.crbegin())->isRunning();
                    if (lastRunning)
                    {
                        // Check last one that stopped
                        auto lastIndexNotRunning = 0ull;
                        for (auto i = mTWorkers.size() - 1 ; i > 0 ; i--)
                        {
                            if (!mTWorkers[i]->checkAndWork(tDatums))
                            {
                                lastIndexNotRunning = i;
                                break;
                            }
                        }

                        // Stop workers before last index stopped
                        for (auto i = 0ull; i < lastIndexNotRunning ; i++)
                            mTWorkers[i]->stop();

                        // Try stopping workers after last index stopped
                        lastRunning = false;
                        for (auto i = lastIndexNotRunning+1; i < mTWorkers.size() ; i++)
                        {
                            mTWorkers[i]->tryStop();
                            if (mTWorkers[i]->isRunning())
                            {
                                lastRunning = true;
                                break;
                            }
                        }
                    }
                    return lastRunning;
                }
            }
            else
                return false;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TWorker>
    void SubThread<TDatums, TWorker>::initializationOnThread()
    {
        try
        {
            for (auto& tWorker : mTWorkers)
                tWorker->initializationOnThread();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(SubThread);
}

#endif // OPENPOSE_THREAD_SUB_THREAD_HPP
