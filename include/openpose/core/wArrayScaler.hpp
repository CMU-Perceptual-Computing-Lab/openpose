#ifndef OPENPOSE__CORE__W_ARRAYS_SCALER_HPP
#define OPENPOSE__CORE__W_ARRAYS_SCALER_HPP

#include "../thread/worker.hpp"
#include "arrayScaler.hpp"
#include "scalePose.hpp"

namespace op
{
    template<typename TDatums>
    class WArrayScaler : public Worker<TDatums>
    {
    public:
        explicit WArrayScaler(const std::shared_ptr<ArrayScaler>& arrayScaler);

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<ArrayScaler> spArrayScaler;
    };
}





// Implementation
#include "../utilities/errorAndLog.hpp"
#include "../utilities/macros.hpp"
#include "../utilities/pointerContainer.hpp"
#include "../utilities/profiler.hpp"
namespace op
{
    template<typename TDatums>
    WArrayScaler<TDatums>::WArrayScaler(const std::shared_ptr<ArrayScaler>& arrayScaler) :
        spArrayScaler{arrayScaler}
    {
    }

    template<typename TDatums>
    void WArrayScaler<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WArrayScaler<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Rescale pose data
                for (auto& tDatum : *tDatums)
                {
                    std::vector<Array<float>> arrays{tDatum.pose, tDatum.hands};
                    spArrayScaler->scale(arrays, tDatum.scaleInputToOutput, tDatum.scaleNetToOutput, tDatum.cvInputData.size());
                }
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__, 1000);
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            tDatums = nullptr;
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WArrayScaler);
}

#endif // OPENPOSE__CORE__W_ARRAYS_SCALER_HPP
