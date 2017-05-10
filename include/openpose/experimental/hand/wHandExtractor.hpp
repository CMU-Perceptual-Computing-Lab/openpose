#ifndef OPENPOSE__HAND__W_HAND_EXTRACTOR_HPP
#define OPENPOSE__HAND__W_HAND_EXTRACTOR_HPP

#include <memory> // std::shared_ptr
#include "../../thread/worker.hpp"
#include "handRenderer.hpp"

namespace op
{
    namespace experimental
    {
        template<typename TDatums>
        class WHandExtractor : public Worker<TDatums>
        {
        public:
            explicit WHandExtractor(const std::shared_ptr<HandExtractor>& handExtractor);

            void initializationOnThread();

            void work(TDatums& tDatums);

        private:
            std::shared_ptr<HandExtractor> spHandExtractor;

            DELETE_COPY(WHandExtractor);
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
        WHandExtractor<TDatums>::WHandExtractor(const std::shared_ptr<HandExtractor>& handExtractor) :
            spHandExtractor{handExtractor}
        {
        }

        template<typename TDatums>
        void WHandExtractor<TDatums>::initializationOnThread()
        {
            spHandExtractor->initializationOnThread();
        }

        template<typename TDatums>
        void WHandExtractor<TDatums>::work(TDatums& tDatums)
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
                        spHandExtractor->forwardPass(tDatum.poseKeyPoints, tDatum.cvInputData);
                        tDatum.handKeyPoints = spHandExtractor->getHandKeyPoints();
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

        COMPILE_TEMPLATE_DATUM(WHandExtractor);
    }
}

#endif // OPENPOSE__HAND__W_HAND_EXTRACTOR_HPP
