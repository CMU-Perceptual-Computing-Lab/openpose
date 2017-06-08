#ifndef OPENPOSE_THREAD_W_ID_GENERATOR_HPP
#define OPENPOSE_THREAD_W_ID_GENERATOR_HPP

#include <queue> // std::priority_queue
#include "worker.hpp"

namespace op
{
    template<typename TDatums>
    class WIdGenerator : public Worker<TDatums>
    {
    public:
        explicit WIdGenerator();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        unsigned long long mGlobalCounter;

        DELETE_COPY(WIdGenerator);
    };
}





// Implementation
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/macros.hpp>
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WIdGenerator<TDatums>::WIdGenerator() :
        mGlobalCounter{0ull}
    {
    }

    template<typename TDatums>
    void WIdGenerator<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WIdGenerator<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Add ID
                for (auto& tDatum : *tDatums)
                    tDatum.id = mGlobalCounter;
                // Increase ID
                mGlobalCounter++;
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            tDatums = nullptr;
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WIdGenerator);
}

#endif // OPENPOSE_THREAD_W_ID_GENERATOR_HPP
