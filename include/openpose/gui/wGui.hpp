#ifndef OPENPOSE__GUI__W_GUI_HPP
#define OPENPOSE__GUI__W_GUI_HPP

#include <memory> // std::shared_ptr
#include <opencv2/core/core.hpp>
#include "../thread/workerConsumer.hpp"
#include "enumClasses.hpp"
#include "gui.hpp"

namespace op
{
    template<typename TDatums>
    class WGui : public WorkerConsumer<TDatums>
    {
    public:
        explicit WGui(const std::shared_ptr<Gui>& gui);

        void initializationOnThread();

        void workConsumer(const TDatums& tDatums);

    private:
        std::shared_ptr<Gui> spGui;

        DELETE_COPY(WGui);
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
    WGui<TDatums>::WGui(const std::shared_ptr<Gui>& gui) :
        spGui{gui}
    {
    }

    template<typename TDatums>
    void WGui<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WGui<TDatums>::workConsumer(const TDatums& tDatums)
    {
        try
        {
            if (tDatums != nullptr)
            {
                // Check tDatums->size() == 1
                if (tDatums->size() > 1)
                    error("Only implemented for tDatums->size() == 1", __LINE__, __FUNCTION__, __FILE__);
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // T* to T
                auto& tDatumsNoPtr = *tDatums;
                // Refresh GUI
                const auto cvOutputData = (!tDatumsNoPtr.empty() ? tDatumsNoPtr[0].cvOutputData : cv::Mat{});
                spGui->update(cvOutputData);
                // Profiling speed
                if (!tDatumsNoPtr.empty())
                {
                    Profiler::timerEnd(profilerKey);
                    Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__, 1000);
                }
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WGui);
}

#endif // OPENPOSE__GUI__W_GUI_HPP
