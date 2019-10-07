#ifndef OPENPOSE_CORE_W_OP_OUTPUT_TO_CV_MAT_HPP
#define OPENPOSE_CORE_W_OP_OUTPUT_TO_CV_MAT_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/opOutputToCvMat.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WOpOutputToCvMat : public Worker<TDatums>
    {
    public:
        explicit WOpOutputToCvMat(const std::shared_ptr<OpOutputToCvMat>& opOutputToCvMat);

        virtual ~WOpOutputToCvMat();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        const std::shared_ptr<OpOutputToCvMat> spOpOutputToCvMat;

        DELETE_COPY(WOpOutputToCvMat);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WOpOutputToCvMat<TDatums>::WOpOutputToCvMat(const std::shared_ptr<OpOutputToCvMat>& opOutputToCvMat) :
        spOpOutputToCvMat{opOutputToCvMat}
    {
    }

    template<typename TDatums>
    WOpOutputToCvMat<TDatums>::~WOpOutputToCvMat()
    {
    }

    template<typename TDatums>
    void WOpOutputToCvMat<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WOpOutputToCvMat<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // float* -> cv::Mat
                for (auto& tDatumPtr : *tDatums)
                    tDatumPtr->cvOutputData = spOpOutputToCvMat->formatToCvMat(tDatumPtr->outputData);
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            tDatums = nullptr;
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WOpOutputToCvMat);
}

#endif // OPENPOSE_CORE_W_OP_OUTPUT_TO_CV_MAT_HPP
