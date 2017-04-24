#ifndef OPENPOSE__HANDS__W_HANDS_EXTRACTOR_HPP
#define OPENPOSE__HANDS__W_HANDS_EXTRACTOR_HPP

#include <memory> // std::shared_ptr
#include "../../thread/worker.hpp"
#include "handsRenderer.hpp"

namespace op
{
    namespace experimental
    {
        template<typename TDatums>
        class WHandsExtractor : public Worker<TDatums>
        {
        public:
            explicit WHandsExtractor(const std::shared_ptr<HandsExtractor>& handsExtractor);

            void initializationOnThread();

            void work(TDatums& tDatums);

        private:
            std::shared_ptr<HandsExtractor> spHandsExtractor;

            DELETE_COPY(WHandsExtractor);
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
        WHandsExtractor<TDatums>::WHandsExtractor(const std::shared_ptr<HandsExtractor>& handsExtractor) :
            spHandsExtractor{handsExtractor}
        {
        }

        template<typename TDatums>
        void WHandsExtractor<TDatums>::initializationOnThread()
        {
            spHandsExtractor->initializationOnThread();
        }

        template<typename TDatums>
        void WHandsExtractor<TDatums>::work(TDatums& tDatums)
        {
            try
            {
                if (checkNoNullNorEmpty(tDatums))
                {
                    // Debugging log
                    dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    // Profiling speed
                    const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                    // Extract people hands
                    for (auto& tDatum : *tDatums)
                    {
                        spHandsExtractor->forwardPass(tDatum.pose, tDatum.cvInputData);
                        tDatum.hands = spHandsExtractor->getHands();
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

        COMPILE_TEMPLATE_DATUM(WHandsExtractor);
    }
}

#endif // OPENPOSE__HANDS__W_HANDS_EXTRACTOR_HPP
