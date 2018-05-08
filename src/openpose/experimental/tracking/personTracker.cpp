#include <thread>
#include <openpose/experimental/tracking/personTracker.hpp>

namespace op
{

    PersonTracker::PersonTracker(const bool mergeResults) :
        mMergeResults{mergeResults},
        mLastFrameId{-1ll}
    {
        // try
        // {
        //     error("PersonTracker (`tracking` flag) buggy and not working yet, but we are working on it!"
        //           " Coming soon!", __LINE__, __FUNCTION__, __FILE__);
        // }
        // catch (const std::exception& e)
        // {
        //     error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        // }
    }

    PersonTracker::~PersonTracker()
    {
    }

    void PersonTracker::track(Array<float>& poseKeypoints, Array<long long>& poseIds,
                              const cv::Mat& cvMatInput)
    {
        try
        {
            // Note: This case: if mMergeResults == false --> Run LK tracker ONLY IF poseKeypoints.empty() doesn't consider the case
            // of poseKeypoints being empty BECAUSE there were no people on the image

            // if mMergeResults == true --> Combine OP + LK tracker
            // if mMergeResults == false --> Run LK tracker ONLY IF poseKeypoints.empty()
            UNUSED(poseKeypoints);
            UNUSED(cvMatInput);
            UNUSED(poseIds);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PersonTracker::trackLockThread(Array<float>& poseKeypoints, Array<long long>& poseIds, 
                                        const cv::Mat& cvMatInput, const long long frameId)
    {
        try
        {
            // Wait for desired order
            while (mLastFrameId < frameId - 1)
                std::this_thread::sleep_for(std::chrono::microseconds{100});
            // Extract IDs
            track(poseKeypoints, poseIds, cvMatInput);
            // Update last frame id
            mLastFrameId = frameId;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
