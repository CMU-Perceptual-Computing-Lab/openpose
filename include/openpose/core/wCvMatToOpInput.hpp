#ifndef OPENPOSE_CORE_W_CV_MAT_TO_OP_INPUT_HPP
#define OPENPOSE_CORE_W_CV_MAT_TO_OP_INPUT_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/cvMatToOpInput.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WCvMatToOpInput : public Worker<TDatums>
    {
    public:
        explicit WCvMatToOpInput(const std::shared_ptr<CvMatToOpInput>& cvMatToOpInput);

        virtual ~WCvMatToOpInput();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        const std::shared_ptr<CvMatToOpInput> spCvMatToOpInput;

        DELETE_COPY(WCvMatToOpInput);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WCvMatToOpInput<TDatums>::WCvMatToOpInput(const std::shared_ptr<CvMatToOpInput>& cvMatToOpInput) :
        spCvMatToOpInput{cvMatToOpInput}
    {
    }

    template<typename TDatums>
    WCvMatToOpInput<TDatums>::~WCvMatToOpInput()
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
                    tDatum.inputNetData = spCvMatToOpInput->createArray(tDatum.cvInputData,
                                                                        tDatum.scaleInputToNetInputs,
                                                                        tDatum.netInputSizes);
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
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

#endif // OPENPOSE_CORE_W_CV_MAT_TO_OP_INPUT_HPP
