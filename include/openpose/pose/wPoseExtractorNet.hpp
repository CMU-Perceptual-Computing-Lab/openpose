#ifndef OPENPOSE_POSE_W_POSE_EXTRACTOR_NET_HPP
#define OPENPOSE_POSE_W_POSE_EXTRACTOR_NET_HPP

#include <openpose/core/common.hpp>
#include <openpose/pose/poseExtractorNet.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WPoseExtractorNet : public Worker<TDatums>
    {
    public:
        explicit WPoseExtractorNet(const std::shared_ptr<PoseExtractorNet>& poseExtractorSharedPtr);

        virtual ~WPoseExtractorNet();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<PoseExtractorNet> spPoseExtractorNet;

        DELETE_COPY(WPoseExtractorNet);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WPoseExtractorNet<TDatums>::WPoseExtractorNet(const std::shared_ptr<PoseExtractorNet>& poseExtractorSharedPtr) :
        spPoseExtractorNet{poseExtractorSharedPtr}
    {
    }

    template<typename TDatums>
    WPoseExtractorNet<TDatums>::~WPoseExtractorNet()
    {
    }

    template<typename TDatums>
    void WPoseExtractorNet<TDatums>::initializationOnThread()
    {
        try
        {
            spPoseExtractorNet->initializationOnThread();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums>
    void WPoseExtractorNet<TDatums>::work(TDatums& tDatums)
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
                for (auto& tDatumPtr : *tDatums)
                {
                    spPoseExtractorNet->forwardPass(
                        tDatumPtr->inputNetData, Point<int>{tDatumPtr->cvInputData.cols, tDatumPtr->cvInputData.rows},
                        tDatumPtr->scaleInputToNetInputs, tDatumPtr->poseNetOutput);
                    tDatumPtr->poseCandidates = spPoseExtractorNet->getCandidatesCopy();
                    tDatumPtr->poseHeatMaps = spPoseExtractorNet->getHeatMapsCopy();
                    tDatumPtr->poseKeypoints = spPoseExtractorNet->getPoseKeypoints().clone();
                    tDatumPtr->poseScores = spPoseExtractorNet->getPoseScores().clone();
                    tDatumPtr->scaleNetToOutput = spPoseExtractorNet->getScaleNetToOutput();
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

    COMPILE_TEMPLATE_DATUM(WPoseExtractorNet);
}

#endif // OPENPOSE_POSE_W_POSE_EXTRACTOR_NET_HPP
