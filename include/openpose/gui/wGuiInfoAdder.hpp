#ifndef OPENPOSE_GUI_W_ADD_GUI_INFO_HPP
#define OPENPOSE_GUI_W_ADD_GUI_INFO_HPP

#include "guiInfoAdder.hpp"
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WGuiInfoAdder : public Worker<TDatums>
    {
    public:
        explicit WGuiInfoAdder(const std::shared_ptr<GuiInfoAdder>& guiInfoAdder);

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<GuiInfoAdder> spGuiInfoAdder;

        DELETE_COPY(WGuiInfoAdder);
    };
}





// Implementation
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/core/macros.hpp>
#include <openpose/utilities/pointerContainer.hpp>
#include <openpose/utilities/profiler.hpp>
namespace op
{
    template<typename TDatums>
    WGuiInfoAdder<TDatums>::WGuiInfoAdder(const std::shared_ptr<GuiInfoAdder>& guiInfoAdder) :
        spGuiInfoAdder{guiInfoAdder}
    {
    }

    template<typename TDatums>
    void WGuiInfoAdder<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WGuiInfoAdder<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Add GUI components to frame
                for (auto& tDatum : *tDatums)
                    spGuiInfoAdder->addInfo(tDatum.cvOutputData, tDatum.poseKeypoints, tDatum.id, tDatum.elementRendered.second);
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

    COMPILE_TEMPLATE_DATUM(WGuiInfoAdder);
}

#endif // OPENPOSE_GUI_W_ADD_GUI_INFO_HPP
