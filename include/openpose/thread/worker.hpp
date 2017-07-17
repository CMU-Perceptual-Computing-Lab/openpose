#ifndef OPENPOSE_THREAD_WORKER_HPP
#define OPENPOSE_THREAD_WORKER_HPP

#include <openpose/core/common.hpp>

namespace op
{
    template<typename TDatums>
    class Worker
    {
    public:
        Worker();

        virtual ~Worker();

        virtual void initializationOnThread() = 0;

        bool checkAndWork(TDatums& tDatums);

        inline bool isRunning() const
        {
            return mIsRunning;
        }

        inline void stop()
        {
            mIsRunning = false;
        }

        // Virtual in case some function needs spetial stopping (e.g. buffers might not stop inmediately and need a few iterations)
        inline virtual void tryStop()
        {
            stop();
        }

    protected:
        virtual void work(TDatums& tDatums) = 0;

    private:
        bool mIsRunning;

        DELETE_COPY(Worker);
    };
}





// Implementation
namespace op
{
    template<typename TDatums>
    Worker<TDatums>::Worker() :
        mIsRunning{true}
    {
    }

    template<typename TDatums>
    Worker<TDatums>::~Worker()
    {
    }

    template<typename TDatums>
    bool Worker<TDatums>::checkAndWork(TDatums& tDatums)
    {
        if (mIsRunning)
            work(tDatums);
        return mIsRunning;
    }

    COMPILE_TEMPLATE_DATUM(Worker);
}

#endif // OPENPOSE_THREAD_WORKER_HPP
