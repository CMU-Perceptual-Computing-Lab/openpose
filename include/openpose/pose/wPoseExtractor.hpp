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

        virtual ~WPoseExtractor();

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
    WPoseExtractor<TDatums>::~WPoseExtractor()
    {
    }

    template<typename TDatums>
    void WPoseExtractor<TDatums>::initializationOnThread()
    {
        try
        {
            spPoseExtractor->initializationOnThread();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums>
    void WPoseExtractor<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Extract people pose
                for (auto i = 0u ; i < tDatums->size() ; i++)
                // for (auto& tDatum : *tDatums)
                {
                    auto& tDatumPtr = (*tDatums)[i];
                    // OpenPose net forward pass
                    spPoseExtractor->forwardPass(
                        tDatumPtr->inputNetData, Point<int>{tDatumPtr->cvInputData.cols(), tDatumPtr->cvInputData.rows()},
                        tDatumPtr->scaleInputToNetInputs, tDatumPtr->poseNetOutput, tDatumPtr->id);
                    // OpenPose keypoint detector
                    tDatumPtr->poseCandidates = spPoseExtractor->getCandidatesCopy();
                    tDatumPtr->poseHeatMaps = spPoseExtractor->getHeatMapsCopy();
                    tDatumPtr->poseKeypoints = spPoseExtractor->getPoseKeypoints().clone();
                    tDatumPtr->poseScores = spPoseExtractor->getPoseScores().clone();
                    tDatumPtr->scaleNetToOutput = spPoseExtractor->getScaleNetToOutput();
                    // Keep desired top N people
                    spPoseExtractor->keepTopPeople(tDatumPtr->poseKeypoints, tDatumPtr->poseScores);
                    // ID extractor (experimental)
                    tDatumPtr->poseIds = spPoseExtractor->extractIdsLockThread(
                        tDatumPtr->poseKeypoints, tDatumPtr->cvInputData, i, tDatumPtr->id);
                    // Tracking (experimental)
                    spPoseExtractor->trackLockThread(
                        tDatumPtr->poseKeypoints, tDatumPtr->poseIds, tDatumPtr->cvInputData, i, tDatumPtr->id);
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

    COMPILE_TEMPLATE_DATUM(WPoseExtractor);
}

#endif // OPENPOSE_POSE_W_POSE_EXTRACTOR_HPP
