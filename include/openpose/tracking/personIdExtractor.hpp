#ifndef OPENPOSE_TRACKING_PERSON_ID_EXTRACTOR_HPP
#define OPENPOSE_TRACKING_PERSON_ID_EXTRACTOR_HPP

#include <atomic>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <openpose/core/common.hpp>

namespace op
{
    struct PersonEntry
    {
        long long counterLastDetection;
        std::vector<cv::Point2f> keypoints;
        std::vector<char> status;
        /*
        PersonEntry(long long _last_frame, 
                    std::vector<cv::Point2f> _keypoints,
                    std::vector<char> _active):
                    last_frame(_last_frame), keypoints(_keypoints),
                    active(_active)
                    {}
        */
    };
    class OP_API PersonIdExtractor
    {

    public:
        PersonIdExtractor(const float confidenceThreshold = 0.1f, const float inlierRatioThreshold = 0.5f,
                          const float distanceThreshold = 30.f, const int numberFramesToDeletePerson = 10);

        virtual ~PersonIdExtractor();

        Array<long long> extractIds(const Array<float>& poseKeypoints, const cv::Mat& cvMatInput,
                                    const unsigned long long imageViewIndex = 0ull);

        Array<long long> extractIdsLockThread(const Array<float>& poseKeypoints, const cv::Mat& cvMatInput,
                                              const unsigned long long imageViewIndex,
                                              const long long frameId);

    private:
        const float mConfidenceThreshold;
        const float mInlierRatioThreshold;
        const float mDistanceThreshold;
        const int mNumberFramesToDeletePerson;
        long long mNextPersonId;
        cv::Mat mImagePrevious;
        std::vector<cv::Mat> mPyramidImagesPrevious;
        std::unordered_map<int, PersonEntry> mPersonEntries;
        // Thread-safe variables
        std::atomic<long long> mLastFrameId;

        DELETE_COPY(PersonIdExtractor);
    };
}

#endif // OPENPOSE_TRACKING_PERSON_ID_EXTRACTOR_HPP
