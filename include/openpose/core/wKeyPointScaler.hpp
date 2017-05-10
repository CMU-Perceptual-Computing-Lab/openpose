#ifndef OPENPOSE__CORE__W_KEY_POINT_SCALER_HPP
#define OPENPOSE__CORE__W_KEY_POINT_SCALER_HPP

#include "../thread/worker.hpp"
#include "keyPointScaler.hpp"
#include "scaleKeyPoints.hpp"

namespace op
{
    template<typename TDatums>
    class WKeyPointScaler : public Worker<TDatums>
    {
    public:
        explicit WKeyPointScaler(const std::shared_ptr<KeyPointScaler>& keyPointScaler);

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<KeyPointScaler> spKeyPointScaler;
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
    WKeyPointScaler<TDatums>::WKeyPointScaler(const std::shared_ptr<KeyPointScaler>& keyPointScaler) :
        spKeyPointScaler{keyPointScaler}
    {
    }

    template<typename TDatums>
    void WKeyPointScaler<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WKeyPointScaler<TDatums>::work(TDatums& tDatums)
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
                    std::vector<Array<float>> arraysToScale{tDatum.poseKeyPoints, tDatum.handKeyPoints, tDatum.faceKeyPoints};
                    spKeyPointScaler->scale(arraysToScale, tDatum.scaleInputToOutput, tDatum.scaleNetToOutput, tDatum.cvInputData.size());
                }
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__, Profiler::DEFAULT_X);
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

    COMPILE_TEMPLATE_DATUM(WKeyPointScaler);
}

#endif // OPENPOSE__CORE__W_KEY_POINT_SCALER_HPP
