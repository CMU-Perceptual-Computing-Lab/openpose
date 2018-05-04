#ifndef OPENPOSE_TRACKING_PERSON_TRACKER_HPP
#define OPENPOSE_TRACKING_PERSON_TRACKER_HPP

#include <atomic>
#include <openpose/core/common.hpp>
#include <openpose/experimental/tracking/personTracker.hpp>

namespace op
{
    class OP_API PersonTracker
    {

    public:
        PersonTracker(const bool mergeResults);

        virtual ~PersonTracker();

        void track(Array<float>& poseKeypoints, const cv::Mat& cvMatInput, const Array<long long>& poseIds,
                   const unsigned long long imageViewIndex = 0ull);

        void trackLockThread(Array<float>& poseKeypoints, const cv::Mat& cvMatInput, const Array<long long>& poseIds,
                             const unsigned long long imageViewIndex, const long long frameId);

    private:
        const bool mMergeResults;

        // Thread-safe variables
        std::atomic<long long> mLastFrameId;

        DELETE_COPY(PersonTracker);
    };
}

#endif // OPENPOSE_TRACKING_PERSON_TRACKER_HPP
