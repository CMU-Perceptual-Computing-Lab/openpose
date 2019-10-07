#ifndef OPENPOSE_POSE_POSE_EXTRACTOR_HPP
#define OPENPOSE_POSE_POSE_EXTRACTOR_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/enumClasses.hpp>
#include <openpose/core/keepTopNPeople.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/pose/poseExtractorNet.hpp>
#include <openpose/tracking/personIdExtractor.hpp>
#include <openpose/tracking/personTracker.hpp>

namespace op
{
    class OP_API PoseExtractor
    {
    public:
        PoseExtractor(const std::shared_ptr<PoseExtractorNet>& poseExtractorNet,
                      const std::shared_ptr<KeepTopNPeople>& keepTopNPeople = nullptr,
                      const std::shared_ptr<PersonIdExtractor>& personIdExtractor = nullptr,
                      const std::shared_ptr<std::vector<std::shared_ptr<PersonTracker>>>& personTracker = {},
                      const int numberPeopleMax = -1, const int tracking = -1);

        virtual ~PoseExtractor();

        void initializationOnThread();

        void forwardPass(const std::vector<Array<float>>& inputNetData,
                         const Point<int>& inputDataSize,
                         const std::vector<double>& scaleRatios,
                         const Array<float>& poseNetOutput = Array<float>{},
                         const long long frameId = -1ll);

        // PoseExtractorNet functions
        Array<float> getHeatMapsCopy() const;

        std::vector<std::vector<std::array<float, 3>>> getCandidatesCopy() const;

        Array<float> getPoseKeypoints() const;

        Array<float> getPoseScores() const;

        float getScaleNetToOutput() const;

        // KeepTopNPeople functions
        void keepTopPeople(Array<float>& poseKeypoints, const Array<float>& poseScores) const;

        // PersonIdExtractor functions
        // Not thread-safe
        Array<long long> extractIds(const Array<float>& poseKeypoints, const Matrix& cvMatInput,
                                    const unsigned long long imageIndex = 0ull);

        // Same than extractIds but thread-safe
        Array<long long> extractIdsLockThread(const Array<float>& poseKeypoints, const Matrix& cvMatInput,
                                              const unsigned long long imageIndex,
                                              const long long frameId);

        // PersonTracker functions
        void track(Array<float>& poseKeypoints, Array<long long>& poseIds,
                   const Matrix& cvMatInput, const unsigned long long imageViewIndex = 0ull);

        void trackLockThread(Array<float>& poseKeypoints, Array<long long>& poseIds,
                             const Matrix& cvMatInput,
                             const unsigned long long imageViewIndex,
                             const long long frameId);

    private:
        const int mNumberPeopleMax;
        const int mTracking;
        const std::shared_ptr<PoseExtractorNet> spPoseExtractorNet;
        const std::shared_ptr<KeepTopNPeople> spKeepTopNPeople;
        const std::shared_ptr<PersonIdExtractor> spPersonIdExtractor;
        const std::shared_ptr<std::vector<std::shared_ptr<PersonTracker>>> spPersonTrackers;

        DELETE_COPY(PoseExtractor);
    };
}

#endif // OPENPOSE_POSE_POSE_EXTRACTOR_HPP
