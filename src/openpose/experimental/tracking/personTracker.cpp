#include <thread>
#include <openpose/experimental/tracking/personTracker.hpp>
#include <iostream>

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

    void personEntriesFromOP(std::unordered_map<int, PersonTrackerEntry>& personEntries, const Array<float>& poseKeypoints, const Array<long long>& poseIds)
    {
        personEntries.clear();
        for(int i=0; i<poseKeypoints.getSize()[0]; i++)
        {
            auto id = poseIds[i];
            personEntries[id] = PersonTrackerEntry();
            personEntries[id].keypoints.resize(poseKeypoints.getSize()[1]);
            personEntries[id].status.resize(poseKeypoints.getSize()[1]);
            for(int j=0; j<poseKeypoints.getSize()[1]; j++)
            {
                personEntries[id].keypoints[j].x = poseKeypoints[
                        i*poseKeypoints.getSize()[1]*poseKeypoints.getSize()[2] +
                        j*poseKeypoints.getSize()[2] + 0];
                personEntries[id].keypoints[j].y = poseKeypoints[
                        i*poseKeypoints.getSize()[1]*poseKeypoints.getSize()[2] +
                        j*poseKeypoints.getSize()[2] + 1];
                float prob = poseKeypoints[
                        i*poseKeypoints.getSize()[1]*poseKeypoints.getSize()[2] +
                        j*poseKeypoints.getSize()[2] + 2];
                if(prob < 0.05) personEntries[id].status[j] = 0;
                else personEntries[id].status[j] = 1;
            }
        }
    }

    void OPFromPersonEntries(Array<float>& poseKeypoints, std::unordered_map<int, PersonTrackerEntry>& personEntries)
    {
        if (!personEntries.size())
            return;
        int dims[] = { (int)personEntries.size(), (int)personEntries.begin()->second.keypoints.size(), 3 };
        cv::Mat opArrayMat(3,dims,CV_32FC1);
        int i=0;
        for (auto& kv : personEntries)
        {
            const PersonTrackerEntry& pe = kv.second;
            for (int j=0; j<dims[1]; j++)
            {
                opArrayMat.at<float>(i*dims[1]*dims[2] + j*dims[2] + 0) = pe.keypoints[j].x;
                opArrayMat.at<float>(i*dims[1]*dims[2] + j*dims[2] + 1) = pe.keypoints[j].y;
                opArrayMat.at<float>(i*dims[1]*dims[2] + j*dims[2] + 2) = !(int)pe.status[j];
                if (pe.keypoints[j].x == 0 && pe.keypoints[j].y == 0)
                    opArrayMat.at<float>(i*dims[1]*dims[2] + j*dims[2] + 2) = 0;
            }
            i++;
        }
        poseKeypoints.setFrom(opArrayMat);
    }

    void PersonTracker::track(Array<float>& poseKeypoints, Array<long long>& poseIds,
                              const cv::Mat& cvMatInput)
    {
        try
        {
            std::cout << poseKeypoints.getSize()[0] << std::endl;
            std::cout << mMergeResults << std::endl;
            std::cout << "---" << std::endl;

            // Sanity Checks
            if(!poseKeypoints.empty() && !poseIds.empty())
                if(poseKeypoints.getSize()[0] != poseIds.getSize()[0])
                     error("poseKeypoints and poseIds should have the same number of people", __LINE__, __FUNCTION__, __FILE__);

//            // First frame
//            if(mImagePrevious.empty()){
//                // Create mPersonEntries
//                personEntriesFromOP(mPersonEntries, poseKeypoints, poseIds);


//            }


            // REMEMBER TO FLIP THE LK VALUES!!

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
