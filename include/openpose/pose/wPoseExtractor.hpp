#ifndef OPENPOSE_POSE_W_POSE_EXTRACTOR_HPP
#define OPENPOSE_POSE_W_POSE_EXTRACTOR_HPP

#include <openpose/core/common.hpp>
#include <openpose/pose/poseExtractor.hpp>
#include <openpose/thread/worker.hpp>

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
#include <openpose/utilities/pointerContainer.hpp>
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
                    spPoseExtractor->forwardPass(tDatum.inputNetData, Point<int>{tDatum.cvInputData.cols, tDatum.cvInputData.rows}, tDatum.scaleRatios);
                    tDatum.poseHeatMaps = spPoseExtractor->getHeatMaps();
                    tDatum.poseKeypoints = spPoseExtractor->getPoseKeypoints();
                    tDatum.scaleNetToOutput = spPoseExtractor->getScaleNetToOutput();
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

    COMPILE_TEMPLATE_DATUM(WPoseExtractor);
}

#endif // OPENPOSE_POSE_W_POSE_EXTRACTOR_HPP
