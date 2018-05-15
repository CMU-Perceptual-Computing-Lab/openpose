#ifndef OPENPOSE_POSE_POSE_EXTRACTOR_HPP
#define OPENPOSE_POSE_POSE_EXTRACTOR_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/enumClasses.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/pose/poseExtractorNet.hpp>
#include <openpose/experimental/tracking/personIdExtractor.hpp>

namespace op
{
    class OP_API PoseExtractor
    {
    public:
        PoseExtractor(const std::shared_ptr<PoseExtractorNet>& poseExtractorNet,
                      const std::shared_ptr<PersonIdExtractor>& personIdExtractor,
                      const int numberPeopleMax = -1);

        virtual ~PoseExtractor();

        void initializationOnThread();

        void forwardPass(const std::vector<Array<float>>& inputNetData,
                         const Point<int>& inputDataSize,
                         const std::vector<double>& scaleRatios);

        // PoseExtractorNet functions
        Array<float> getHeatMapsCopy() const;

        std::vector<std::vector<std::array<float,3>>> getCandidatesCopy() const;

        Array<float> getPoseKeypoints() const;

        Array<float> getPoseScores() const;

        float getScaleNetToOutput() const;

        // PersonIdExtractor functions
        // Not thread-safe
        Array<long long> extractIds(const Array<float>& poseKeypoints, const cv::Mat& cvMatInput,
                                    const unsigned long long imageIndex = 0ull);

        // Same than extractIds but thread-safe
        Array<long long> extractIdsLockThread(const Array<float>& poseKeypoints, const cv::Mat& cvMatInput,
                                              const unsigned long long imageIndex,
                                              const long long frameId);

    private:
        const int mNumberPeopleMax;
        const std::shared_ptr<PoseExtractorNet> spPoseExtractorNet;
        const std::shared_ptr<PersonIdExtractor> spPersonIdExtractor;

        DELETE_COPY(PoseExtractor);
    };
}

#endif // OPENPOSE_POSE_POSE_EXTRACTOR_HPP
