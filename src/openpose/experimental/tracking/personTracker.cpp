#include <thread>
#include <openpose/experimental/tracking/personTracker.hpp>

namespace op
{

    PersonTracker::PersonTracker(const bool mergeResults, const int levels,
                                 const int patchSize, const bool trackVelocity) :
        mMergeResults{mergeResults},
        mLevels{levels},
        mPatchSize{patchSize},
        mTrackVelocity{trackVelocity},
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

    void personEntriesFromOP(std::unordered_map<int, PersonTrackerEntry>& personEntries, Array<float>& poseKeypoints)
    {
        personEntries.clear();
        wip
    }

    void PersonTracker::track(Array<float>& poseKeypoints, Array<long long>& poseIds,
                              const cv::Mat& cvMatInput)
    {
        try
        {

            // Sanity Checks
            if(!poseKeypoints.empty() && !poseIds.empty())
                if(poseKeypoints.getSize()[0] != poseIds.getSize()[0])
                     error("poseKeypoints and poseIds should have the same number of people", __LINE__, __FUNCTION__, __FILE__);

            // First frame
            if(mImagePrevious.empty()){
                // Create mPersonEntries

            }


            /*
             * 1. Get poseKeypoints for all people - Checks
             * 2. If last image is empty or mPersonEntries is empty (and poseKeypoints and poseIds has data or crash it)
             *      Create mPersonEntries referencing poseIds
             *      Initialize LK points
             * 3. If poseKeypoints is not empty and poseIds has data
             *      1. Reference poseIds against internal mPersonEntries
             *      2. CRUD where required
             *      3. If mergeResults:
             *            Run LK update on mPersonEntries
             *            smooth out with poseKeypoints
             *         else
             *            replace with poseKeypoints
             * 4. Else if poseKeypoints is empty
             *      1. Run LK update on mPersonEntries
             *      2. replace with poseKeypoints
             */


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
