#ifndef OPENPOSE__CORE__W_CV_MAT_TO_OP_INPUT_HPP
#define OPENPOSE__CORE__W_CV_MAT_TO_OP_INPUT_HPP

#include <memory> // std::shared_ptr
#include <opencv2/core/core.hpp>
#include "../thread/worker.hpp"
#include "cvMatToOpInput.hpp"

namespace op
{
    template<typename TDatums>
    class WCvMatToOpInput : public Worker<TDatums>
    {
    public:
        explicit WCvMatToOpInput(const std::shared_ptr<CvMatToOpInput>& cvMatToOpInput);

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        const std::shared_ptr<CvMatToOpInput> spCvMatToOpInput;

        DELETE_COPY(WCvMatToOpInput);
    };
}





// Implementation
#include "../utilities/errorAndLog.hpp"
#include "../utilities/macros.hpp"
#include "../utilities/openCv.hpp"
#include "../utilities/pointerContainer.hpp"
#include "../utilities/profiler.hpp"
namespace op
{
    template<typename TDatums>
    WCvMatToOpInput<TDatums>::WCvMatToOpInput(const std::shared_ptr<CvMatToOpInput>& cvMatToOpInput) :
        spCvMatToOpInput{cvMatToOpInput}
    {
    }

    template<typename TDatums>
    void WCvMatToOpInput<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WCvMatToOpInput<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // cv::Mat -> float*
                for (auto& tDatum : *tDatums)
                    tDatum.inputNetData = spCvMatToOpInput->format(tDatum.cvInputData);
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

    COMPILE_TEMPLATE_DATUM(WCvMatToOpInput);
}

#endif // OPENPOSE__CORE__W_CV_MAT_TO_OP_INPUT_HPP
