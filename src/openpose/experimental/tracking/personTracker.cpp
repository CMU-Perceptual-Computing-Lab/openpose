#include <thread>
#include <openpose/experimental/tracking/personTracker.hpp>

namespace op
{

    PersonTracker::PersonTracker(const bool mergeResults) :
        mMergeResults{mergeResults},
        mLastFrameId{-1ll}
    {
        try
        {
            error("PersonTracker (`tracking` flag) buggy and not working yet, but we are working on it!"
                  " Coming soon!", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    PersonTracker::~PersonTracker()
    {
    }

    void PersonTracker::track(Array<float>& poseKeypoints, const cv::Mat& cvMatInput,
                              const Array<long long>& poseIds, const unsigned long long imageViewIndex)
    {
        try
        {
            // if mergeResults == true --> Combine OP + LK tracker
            // if mergeResults == false --> Run LK tracker ONLY IF poseKeypoints.empty()
            // imageViewIndex has camera view index (for 3D, i.e. index 2 means that there are at least
            //     3 cameras and this is camera index 2)
            UNUSED(poseKeypoints);
            UNUSED(cvMatInput);
            UNUSED(poseIds);
            UNUSED(imageViewIndex);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PersonTracker::trackLockThread(Array<float>& poseKeypoints, const cv::Mat& cvMatInput,
                                        const Array<long long>& poseIds, const unsigned long long imageViewIndex,
                                        const long long frameId)
    {
        try
        {
            // Wait for desired order
            while (mLastFrameId < frameId - 1)
                std::this_thread::sleep_for(std::chrono::microseconds{100});
            // Extract IDs
            track(poseKeypoints, cvMatInput, poseIds, imageViewIndex);
            // Update last frame id
            mLastFrameId = frameId;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
