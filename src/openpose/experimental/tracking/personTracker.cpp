#include <thread>
#include <openpose/experimental/tracking/personTracker.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <openpose/experimental/tracking/pyramidalLK.hpp>

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

    void updateLK(std::unordered_map<int,PersonTrackerEntry>& personEntries,
                  std::vector<cv::Mat>& pyramidImagesPrevious, std::vector<cv::Mat>& pyramidImagesCurrent,
                  const cv::Mat& imagePrevious, const cv::Mat& imageCurrent,
                  const int levels, const int patchSize, const bool trackVelocity)
    {
        try
        {
            // Inefficient version, do it per person
            for(auto& kv : personEntries)
            {
                PersonTrackerEntry newPersonEntry;
                PersonTrackerEntry& oldPersonEntry = kv.second;
                if (trackVelocity)
                {
                    newPersonEntry.keypoints = oldPersonEntry.getPredicted();
                    pyramidalLKOcv(oldPersonEntry.keypoints, newPersonEntry.keypoints, pyramidImagesPrevious, pyramidImagesCurrent,
                                   oldPersonEntry.status, imagePrevious, imageCurrent, levels, patchSize, true);
                }
                else
                    pyramidalLKOcv(oldPersonEntry.keypoints, newPersonEntry.keypoints, pyramidImagesPrevious, pyramidImagesCurrent,
                                   oldPersonEntry.status, imagePrevious, imageCurrent, levels, patchSize, false);

                newPersonEntry.lastKeypoints = oldPersonEntry.keypoints;
                newPersonEntry.status = oldPersonEntry.status;
                oldPersonEntry = newPersonEntry;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void vizPersonEntries(cv::Mat& debugImage, std::unordered_map<int, PersonTrackerEntry>& personEntries, bool mTrackVelocity)
    {
        for (auto& kv : personEntries)
        {
            PersonTrackerEntry& pe = kv.second;
            std::vector<cv::Point2f> predictedKeypoints = pe.getPredicted();
            for (size_t i=0; i<pe.keypoints.size(); i++)
            {
                cv::circle(debugImage, pe.keypoints[i], 3, cv::Scalar(255,0,0),CV_FILLED);
                cv::putText(debugImage, std::to_string((int)pe.status[i]), pe.keypoints[i],
                            cv::FONT_HERSHEY_DUPLEX, 0.4, cv::Scalar(0,0,255),1);

                if (pe.lastKeypoints.size())
                {
                    cv::line(debugImage, pe.keypoints[i], pe.lastKeypoints[i],cv::Scalar(255,0,0));
                    cv::circle(debugImage, pe.lastKeypoints[i], 3, cv::Scalar(255,255,0),CV_FILLED);
                }
                if (predictedKeypoints.size() && mTrackVelocity)
                {
                    cv::line(debugImage, pe.keypoints[i], predictedKeypoints[i],cv::Scalar(255,0,0));
                    cv::circle(debugImage, predictedKeypoints[i], 3, cv::Scalar(255,0,255),CV_FILLED);
                }
            }
        }
    }

    void personEntriesFromOP(std::unordered_map<int, PersonTrackerEntry>& personEntries,
                             const Array<float>& poseKeypoints, const Array<long long>& poseIds,
                             float confidenceThreshold)
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
                if(prob < confidenceThreshold) personEntries[id].status[j] = 0;
                else personEntries[id].status[j] = 1;
            }
        }
    }

    void syncPersonEntriesWithOP(std::unordered_map<int, PersonTrackerEntry>& personEntries,
                                 const Array<float>& poseKeypoints, const Array<long long>& poseIds,
                                 float confidenceThreshold, bool mergeResults)
    {
        if(poseIds.empty()) return;

        // Delete
        for (auto kv = personEntries.cbegin(); kv != personEntries.cend() /* not hoisted */; /* no increment */)
        {
            bool exists = false;
            for(int i=0; i<poseIds.getSize()[0]; i++)
            {
                auto id = poseIds[i];
                if(id == kv->first) exists = true;
            }
            if(!exists) personEntries.erase(kv++);
            else ++kv;
        }

        // Update or Add
        for(int i=0; i<poseIds.getSize()[0]; i++){
            auto id = poseIds[i];

            // Update
            if(personEntries.count(id) && mergeResults){

                PersonTrackerEntry& personEntry = personEntries[id];
                for(int j=0; j<poseKeypoints.getSize()[1]; j++)
                {
                    float x = poseKeypoints[
                            i*poseKeypoints.getSize()[1]*poseKeypoints.getSize()[2] +
                            j*poseKeypoints.getSize()[2] + 0];
                    float y = poseKeypoints[
                            i*poseKeypoints.getSize()[1]*poseKeypoints.getSize()[2] +
                            j*poseKeypoints.getSize()[2] + 1];
                    float prob = poseKeypoints[
                            i*poseKeypoints.getSize()[1]*poseKeypoints.getSize()[2] +
                            j*poseKeypoints.getSize()[2] + 2];
                    cv::Point lkPoint = personEntry.keypoints[j];
                    cv::Point opPoint = cv::Point(x,y);

                    if(prob < confidenceThreshold) personEntries[id].status[j] = 0;
                    else
                    {
                        personEntries[id].status[j] = 1;
                        float distance = sqrt(pow(lkPoint.x-opPoint.x,2)+pow(lkPoint.y-opPoint.y,2));
                        if (distance < 5)
                            personEntries[id].keypoints[j] = lkPoint;
                        else if (distance < 10)
                            personEntries[id].keypoints[j] = cv::Point((lkPoint.x+opPoint.x)/2.,(lkPoint.y+opPoint.y)/2.);
                        else
                            personEntries[id].keypoints[j] = opPoint;
                    }
                }

            }
            // Add
            else{

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
                    if(prob < confidenceThreshold) personEntries[id].status[j] = 0;
                    else personEntries[id].status[j] = 1;
                }

            }
        }

        // Sanity Check Start
        if((int)personEntries.size() != poseIds.getSize()[0])
        {
            // Print
            for(auto& kv : personEntries) std::cout << kv.first << " ";
            std::cout << std::endl;
            for(int i=0; i<poseIds.getSize()[0]; i++) std::cout << poseIds.at(i) << " ";
            std::cout << std::endl;
            std::cout << "---" << std::endl;
            op::error("Size Mismatch. THere is an error in your poseId formatting");
        }
    }

    void opFromPersonEntries(Array<float>& poseKeypoints,
                             const std::unordered_map<int, PersonTrackerEntry>& personEntries,
                             const Array<long long>& poseIds)
    {
        if (!personEntries.size() || poseIds.empty())
            return;
        int dims[] = { (int)personEntries.size(), (int)personEntries.begin()->second.keypoints.size(), 3 };
        cv::Mat opArrayMat(3,dims,CV_32FC1);
        for(int i=0; i<poseIds.getSize()[0]; i++){
            int id = poseIds[i];
            const PersonTrackerEntry& pe = personEntries.at(id);
            for (int j=0; j<dims[1]; j++)
            {
                opArrayMat.at<float>(i*dims[1]*dims[2] + j*dims[2] + 0) = pe.keypoints[j].x;
                opArrayMat.at<float>(i*dims[1]*dims[2] + j*dims[2] + 1) = pe.keypoints[j].y;
                opArrayMat.at<float>(i*dims[1]*dims[2] + j*dims[2] + 2) = (int)pe.status[j];
                if (pe.keypoints[j].x == 0 && pe.keypoints[j].y == 0)
                    opArrayMat.at<float>(i*dims[1]*dims[2] + j*dims[2] + 2) = 0;
            }
        }
        poseKeypoints.setFrom(opArrayMat);
    }

    void PersonTracker::track(Array<float>& poseKeypoints, Array<long long>& poseIds,
                              const cv::Mat& cvMatInput)
    {
        try
        {
            bool mergeResults = mMergeResults;
            mergeResults = true;

            // Sanity Checks
            if(poseKeypoints.getSize(0) != poseIds.getSize(0))
                 error("poseKeypoints and poseIds should have the same number of people", __LINE__, __FUNCTION__, __FILE__);

            // First frame
            if(mImagePrevious.empty())
            {
                // Create mPersonEntries
                personEntriesFromOP(mPersonEntries, poseKeypoints, poseIds, mConfidenceThreshold);
                // Capture current frame as floating point
                cvMatInput.convertTo(mImagePrevious, CV_32F);
                // Save Last Ids
                mLastPoseIds = poseIds;
            }
            else
            {
                // Update LK
                bool newOPData = !poseKeypoints.empty() && !poseIds.empty();
                if((newOPData && mergeResults) || (!newOPData))
                {
                    cv::Mat imageCurrent;
                    std::vector<cv::Mat> pyramidImagesCurrent;
                    cvMatInput.convertTo(imageCurrent, CV_32F);
                    updateLK(mPersonEntries, mPyramidImagesPrevious, pyramidImagesCurrent, mImagePrevious,
                             imageCurrent, mLevels, mPatchSize, mTrackVelocity);
                    mImagePrevious = imageCurrent;
                    mPyramidImagesPrevious = pyramidImagesCurrent;
                }

                // There is new OP Data
                if(newOPData)
                {
                    mLastPoseIds = poseIds;
                    syncPersonEntriesWithOP(mPersonEntries, poseKeypoints, mLastPoseIds, mConfidenceThreshold, mergeResults);
                    opFromPersonEntries(poseKeypoints, mPersonEntries, mLastPoseIds);
                }
                // There is no new OP Data
                else
                {
                    opFromPersonEntries(poseKeypoints, mPersonEntries, mLastPoseIds);
                    poseIds = mLastPoseIds;
                }
            }

            //cv::Mat debugImage = cvMatInput.clone();
            //vizPersonEntries(debugImage, mPersonEntries, mTrackVelocity);
            //cv::imshow("win", debugImage);
            //cv::waitKey(15);

            /*
             * 1. Get poseKeypoints for all people - Checks
             * 2. If last image is empty or mPersonEntries is empty (and poseKeypoints and poseIds has data or crash it)
             *      Create mPersonEntries referencing poseIds
             *      Initialize LK points
             * 3. If poseKeypoints is not empty and poseIds has data
             *      1. Update LK
             *      2. CRUD/Sync - Check mergeResults flag to smooth or not
             * 4. If poseKeypoints is empty
             *      1. Update LK
             *      2. replace poseKeypoints
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
