#ifndef OPENPOSE_CORE_W_VERBOSE_PRINTER_HPP
#define OPENPOSE_CORE_W_VERBOSE_PRINTER_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/verbosePrinter.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WVerbosePrinter : public Worker<TDatums>
    {
    public:
        explicit WVerbosePrinter(const std::shared_ptr<VerbosePrinter>& verbosePrinter);

        virtual ~WVerbosePrinter();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        const std::shared_ptr<VerbosePrinter> spVerbosePrinter;

        DELETE_COPY(WVerbosePrinter);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WVerbosePrinter<TDatums>::WVerbosePrinter(
        const std::shared_ptr<VerbosePrinter>& verbosePrinter) :
        spVerbosePrinter{verbosePrinter}
    {
    }

    template<typename TDatums>
    WVerbosePrinter<TDatums>::~WVerbosePrinter()
    {
    }

    template<typename TDatums>
    void WVerbosePrinter<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WVerbosePrinter<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Print verbose
                if (checkNoNullNorEmpty(tDatums))
                {
                    const auto tDatumPtr = (*tDatums)[0];
                    spVerbosePrinter->printVerbose(tDatumPtr->frameNumber);
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

    COMPILE_TEMPLATE_DATUM(WVerbosePrinter);
}

#endif // OPENPOSE_CORE_W_VERBOSE_PRINTER_HPP
