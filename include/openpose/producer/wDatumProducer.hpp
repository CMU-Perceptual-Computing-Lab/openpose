#ifndef OPENPOSE__PRODUCER__W_DATUM_PRODUCER_HPP
#define OPENPOSE__PRODUCER__W_DATUM_PRODUCER_HPP

#include <limits> // std::numeric_limits
#include <memory> // std::shared_ptr
#include "../thread/workerProducer.hpp"
#include "datumProducer.hpp"

namespace op
{
    template<typename TDatums, typename TDatumsNoPtr>
    class WDatumProducer : public WorkerProducer<TDatums>
    {
    public:
        explicit WDatumProducer(const std::shared_ptr<DatumProducer<TDatumsNoPtr>>& datumProducer);

        void initializationOnThread();

        TDatums workProducer();

    private:
        std::shared_ptr<DatumProducer<TDatumsNoPtr>> spDatumProducer;

        DELETE_COPY(WDatumProducer);
    };
}





// Implementation
#include <vector>
#include "../utilities/errorAndLog.hpp"
#include "../utilities/macros.hpp"
#include "../utilities/profiler.hpp"
#include "../core/datum.hpp"
namespace op
{
    template<typename TDatums, typename TDatumsNoPtr>
    WDatumProducer<TDatums, TDatumsNoPtr>::WDatumProducer(const std::shared_ptr<DatumProducer<TDatumsNoPtr>>& datumProducer) :
        spDatumProducer{datumProducer}
    {
    }

    template<typename TDatums, typename TDatumsNoPtr>
    void WDatumProducer<TDatums, TDatumsNoPtr>::initializationOnThread()
    {
    }

    template<typename TDatums, typename TDatumsNoPtr>
    TDatums WDatumProducer<TDatums, TDatumsNoPtr>::workProducer()
    {
        try
        {
            // Debugging log
            dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Profiling speed
            const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
            // Create and fill TDatums
            const auto isRunningAndTDatums = spDatumProducer->checkIfRunningAndGetDatum();
            // Stop Worker if producer finished
            if (!isRunningAndTDatums.first)
                this->stop();
            // Profiling speed
            Profiler::timerEnd(profilerKey);
            Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__, 1000);
            // Debugging log
            dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Return TDatums
            return isRunningAndTDatums.second;
        }
        catch (const std::exception& e)
        {
            this->stop();
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return TDatums{};
        }
    }

    extern template class WDatumProducer<DATUM_BASE, DATUM_BASE_NO_PTR>;
}

#endif // OPENPOSE__PRODUCER__W_DATUM_PRODUCER_HPP
