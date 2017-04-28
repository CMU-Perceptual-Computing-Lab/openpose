#ifndef OPENPOSE__POSE__W_POSE_EXTRACTOR_HPP
#define OPENPOSE__POSE__W_POSE_EXTRACTOR_HPP

#include <memory> // std::shared_ptr
#include "../thread/worker.hpp"
#include "poseExtractor.hpp"

namespace op
{
    template<typename TDatums>
    class WPoseExtractor : public Worker<TDatums>
    {
    public:
        explicit WPoseExtractor(const std::shared_ptr<PoseExtractor>& poseExtractorSharedPtr);

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<PoseExtractor> spPoseExtractor;

        DELETE_COPY(WPoseExtractor);
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
    WPoseExtractor<TDatums>::WPoseExtractor(const std::shared_ptr<PoseExtractor>& poseExtractorSharedPtr) :
        spPoseExtractor{poseExtractorSharedPtr}
    {
    }

    template<typename TDatums>
    void WPoseExtractor<TDatums>::initializationOnThread()
    {
        spPoseExtractor->initializationOnThread();
    }

    template<typename TDatums>
    void WPoseExtractor<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Extract people pose
                for (auto& tDatum : *tDatums)
                {
                    spPoseExtractor->forwardPass(tDatum.inputNetData, tDatum.cvInputData.size());
                    tDatum.poseHeatMaps = spPoseExtractor->getHeatMaps();
                    tDatum.poseKeyPoints = spPoseExtractor->getPoseKeyPoints();
                    tDatum.scaleNetToOutput = spPoseExtractor->getScaleNetToOutput();
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

    COMPILE_TEMPLATE_DATUM(WPoseExtractor);
}

#endif // OPENPOSE__POSE__W_POSE_EXTRACTOR_HPP
