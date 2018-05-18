#ifndef OPENPOSE_TRACKING_PERSON_TRACKER_HPP
#define OPENPOSE_TRACKING_PERSON_TRACKER_HPP

#include <atomic>
#include <unordered_map>
#include <openpose/core/common.hpp>

namespace op
{
    struct PersonTrackerEntry
    {
        std::vector<cv::Point2f> keypoints;
        std::vector<cv::Point2f> lastKeypoints;
        std::vector<char> status;
        std::vector<cv::Point2f> getPredicted() const
        {
            std::vector<cv::Point2f> predictedKeypoints(keypoints);
            if (!lastKeypoints.size())
                return predictedKeypoints;
            for (size_t i=0; i<keypoints.size(); i++)
            {
                predictedKeypoints[i] = cv::Point2f{predictedKeypoints[i].x + (keypoints[i].x-lastKeypoints[i].x),
                                                    predictedKeypoints[i].y + (keypoints[i].y-lastKeypoints[i].y)};
            }
            return predictedKeypoints;
        }
    };

    class OP_API PersonTracker
    {

    public:
        PersonTracker(const bool mergeResults, const int levels = 3, const int patchSize = 31,
                      const float confidenceThreshold = 0.05f, const bool trackVelocity = false,
                      const bool scaleVarying = false, const float rescale = 640);

        virtual ~PersonTracker();

        void track(Array<float>& poseKeypoints, Array<long long>& poseIds, const cv::Mat& cvMatInput);

        void trackLockThread(Array<float>& poseKeypoints, Array<long long>& poseIds, const cv::Mat& cvMatInput,
                             const long long frameId);

        bool getMergeResults() const;

    private:
        const bool mMergeResults;
        const int mLevels;
        const int mPatchSize;
        const bool mTrackVelocity;
        const float mConfidenceThreshold;
        const bool mScaleVarying;
        const float mRescale;

        cv::Mat mImagePrevious;
        std::vector<cv::Mat> mPyramidImagesPrevious;
        std::unordered_map<int, PersonTrackerEntry> mPersonEntries;
        Array<long long> mLastPoseIds;

        // Thread-safe variables
        std::atomic<long long> mLastFrameId;

        DELETE_COPY(PersonTracker);
    };
}

#endif // OPENPOSE_TRACKING_PERSON_TRACKER_HPP
