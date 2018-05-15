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
                for (auto i = 0u ; i < tDatums->size() ; i++)
                // for (auto& tDatum : *tDatums)
                {
                    auto& tDatum = (*tDatums)[i];
                    // OpenPose net forward pass
                    spPoseExtractor->forwardPass(tDatum.inputNetData,
                                                 Point<int>{tDatum.cvInputData.cols, tDatum.cvInputData.rows},
                                                 tDatum.scaleInputToNetInputs, tDatum.id);
                    // OpenPose keypoint detector
                    tDatum.poseCandidates = spPoseExtractor->getCandidatesCopy();
                    tDatum.poseHeatMaps = spPoseExtractor->getHeatMapsCopy();
                    tDatum.poseKeypoints = spPoseExtractor->getPoseKeypoints().clone();
                    tDatum.poseScores = spPoseExtractor->getPoseScores().clone();
                    tDatum.scaleNetToOutput = spPoseExtractor->getScaleNetToOutput();
                    // Keep desired top N people
                    spPoseExtractor->keepTopPeople(tDatum.poseKeypoints, tDatum.poseScores);
                    // ID extractor (experimental)
                    tDatum.poseIds = spPoseExtractor->extractIdsLockThread(tDatum.poseKeypoints, tDatum.cvInputData,
                                                                           i, tDatum.id);
                    // Tracking (experimental)
                    spPoseExtractor->trackLockThread(tDatum.poseKeypoints, tDatum.poseIds, tDatum.cvInputData, i,
                                                     tDatum.id);
                }
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
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
