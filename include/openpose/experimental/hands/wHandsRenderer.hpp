#ifndef OPENPOSE__HANDS__W_HANDS_RENDERER_HPP
#define OPENPOSE__HANDS__W_HANDS_RENDERER_HPP

#include <memory> // std::shared_ptr
#include "../../thread/worker.hpp"
#include "handsRenderer.hpp"

namespace op
{
    namespace experimental
    {
        template<typename TDatums>
        class WHandsRenderer : public Worker<TDatums>
        {
        public:
            explicit WHandsRenderer(const std::shared_ptr<HandsRenderer>& handsRenderer);

            void initializationOnThread();

            void work(TDatums& tDatums);

        private:
            std::shared_ptr<HandsRenderer> spHandsRenderer;

            DELETE_COPY(WHandsRenderer);
        };
    }
}





// Implementation
#include "../../utilities/errorAndLog.hpp"
#include "../../utilities/macros.hpp"
#include "../../utilities/pointerContainer.hpp"
#include "../../utilities/profiler.hpp"
namespace op
{
    namespace experimental
    {
        template<typename TDatums>
        WHandsRenderer<TDatums>::WHandsRenderer(const std::shared_ptr<HandsRenderer>& handsRenderer) :
            spHandsRenderer{handsRenderer}
        {
        }

        template<typename TDatums>
        void WHandsRenderer<TDatums>::initializationOnThread()
        {
            spHandsRenderer->initializationOnThread();
        }

        template<typename TDatums>
        void WHandsRenderer<TDatums>::work(TDatums& tDatums)
        {
            try
            {
                if (checkNoNullNorEmpty(tDatums))
                {
                    // Debugging log
                    dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    // Profiling speed
                    const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                    // Render people hands
                    for (auto& tDatum : *tDatums)
                        spHandsRenderer->renderHands(tDatum.outputData, tDatum.hands);
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

        COMPILE_TEMPLATE_DATUM(WHandsRenderer);
    }
}

#endif // OPENPOSE__HANDS__W_HANDS_RENDERER_HPP
