#ifndef OPENPOSE_EXAMPLES_TUTORIAL_W_USER_POST_PROCESSING_HPP
#define OPENPOSE_EXAMPLES_TUTORIAL_W_USER_POST_PROCESSING_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/cvMatToOpInput.hpp>
#include <openpose/thread/worker.hpp>
#include "userPostProcessing.hpp"

namespace op
{
    template<typename TDatums>
    class WUserPostProcessing : public Worker<TDatums>
    {
    public:
        WUserPostProcessing(const std::shared_ptr<UserPostProcessing>& userPostProcessing);

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<UserPostProcessing> spUserPostProcessing;

        DELETE_COPY(WUserPostProcessing);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WUserPostProcessing<TDatums>::WUserPostProcessing(const std::shared_ptr<UserPostProcessing>& userPostProcessing) :
        spUserPostProcessing{userPostProcessing}
    {
    }

    template<typename TDatums>
    void WUserPostProcessing<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WUserPostProcessing<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                for (auto& datum : *tDatums)
                {
                    // THESE 2 ARE THE ONLY LINES THAT THE USER MUST MODIFY ON THIS HPP FILE, by using the proper
                    // function and datum elements
                    cv::Mat cvOutputData = OP_OP2CVMAT(datum->cvOutputData);
                    spUserPostProcessing->doSomething(cvOutputData, cvOutputData);
                }
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

    // COMPILE_TEMPLATE_DATUM(WUserPostProcessing);
}

#endif // OPENPOSE_EXAMPLES_TUTORIAL_W_USER_POST_PROCESSING_HPP
