#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>
#include <openpose/experimental/tracking/personTracker.hpp>
#include <openpose/experimental/tracking/pyramidalLK.hpp>

namespace op
{
    int roundUp(int numToRound, int multiple)
    {
        if (multiple == 0)
            return numToRound;

        int remainder = numToRound % multiple;
        if (remainder == 0)
            return numToRound;

        return numToRound + multiple - remainder;
    }

    float computePersonScale(const PersonTrackerEntry& personEntry, const cv::Mat& imageCurrent)
    {
        int layerCount = 0;
        if( personEntry.status[0] || personEntry.status[14] ||
            personEntry.status[15] || personEntry.status[16] || personEntry.status[17])
            layerCount++;
        if( personEntry.status[2] || personEntry.status[3] || personEntry.status[4] ||
            personEntry.status[5] || personEntry.status[6] || personEntry.status[7])
            layerCount++;
        if( personEntry.status[8] || personEntry.status[11])
            layerCount++;
        if( personEntry.status[9] || personEntry.status[10] ||
            personEntry.status[12] || personEntry.status[13])
            layerCount++;

        float minX = imageCurrent.size().width, maxX = 0, minY = imageCurrent.size().height, maxY = 0;
        int totalKp = 0;
        for(size_t i=0; i<personEntry.keypoints.size(); i++)
        {
            if(!personEntry.status[i]) continue;
            auto kp = personEntry.keypoints[i];
            if(kp.x < minX) minX = kp.x;
            if(kp.x > maxX) maxX = kp.x;
            if(kp.y < minY) minY = kp.y;
            if(kp.y > maxY) maxY = kp.y;
            totalKp++;
        }
        float xDist = (maxX - minX);
        float yDist = (maxY - minY);
        float maxDist;
        if(xDist > yDist) maxDist = (xDist)*(4/layerCount);
        if(yDist > xDist) maxDist = (yDist)*(4/layerCount);
        float lkSize = roundUp(maxDist / 10., 3);
        //std::cout << lkSize << std::endl;
        return lkSize;
    }

    void updateLK(std::unordered_map<int,PersonTrackerEntry>& personEntries,
                  std::vector<cv::Mat>& pyramidImagesPrevious, std::vector<cv::Mat>& pyramidImagesCurrent,
                  const cv::Mat& imagePrevious, const cv::Mat& imageCurrent,
                  const int levels, const int patchSize, const bool trackVelocity, const bool scaleVarying)
    {
        try
        {
            // Inefficient version, do it per person
            for (auto& kv : personEntries)
            {
                PersonTrackerEntry newPersonEntry;
                PersonTrackerEntry& oldPersonEntry = kv.second;
                int lkSize = patchSize;
                if(scaleVarying)
                {
                    pyramidImagesPrevious.clear();
                    lkSize = computePersonScale(oldPersonEntry, imageCurrent);
                }
                if (trackVelocity)
                {
                    newPersonEntry.keypoints = oldPersonEntry.getPredicted();
                    pyramidalLKOcv(oldPersonEntry.keypoints, newPersonEntry.keypoints, pyramidImagesPrevious,
                                   pyramidImagesCurrent, oldPersonEntry.status, imagePrevious, imageCurrent, levels,
                                   patchSize, true);
                }
                else
                    pyramidalLKOcv(oldPersonEntry.keypoints, newPersonEntry.keypoints, pyramidImagesPrevious,
                                   pyramidImagesCurrent, oldPersonEntry.status, imagePrevious, imageCurrent, levels,
                                   lkSize, false);

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

    // void vizPersonEntries(cv::Mat& debugImage, const std::unordered_map<int, PersonTrackerEntry>& personEntries,
    //                       const bool mTrackVelocity)
    // {
    //     try
    //     {
    //         for (auto& kv : personEntries)
    //         {
    //             const PersonTrackerEntry& pe = kv.second;
    //             const std::vector<cv::Point2f> predictedKeypoints = pe.getPredicted();
    //             for (size_t i=0; i<pe.keypoints.size(); i++)
    //             {
    //                 cv::circle(debugImage, pe.keypoints[i], 3, cv::Scalar(255,0,0),CV_FILLED);
    //                 cv::putText(debugImage, std::to_string((int)pe.status[i]), pe.keypoints[i],
    //                             cv::FONT_HERSHEY_DUPLEX, 0.4, cv::Scalar(0,0,255),1);

    //                 if (pe.lastKeypoints.size())
    //                 {
    //                     cv::line(debugImage, pe.keypoints[i], pe.lastKeypoints[i],cv::Scalar(255,0,0));
    //                     cv::circle(debugImage, pe.lastKeypoints[i], 3, cv::Scalar(255,255,0),CV_FILLED);
    //                 }
    //                 if (predictedKeypoints.size() && mTrackVelocity)
    //                 {
    //                     cv::line(debugImage, pe.keypoints[i], predictedKeypoints[i],cv::Scalar(255,0,0));
    //                     cv::circle(debugImage, predictedKeypoints[i], 3, cv::Scalar(255,0,255),CV_FILLED);
    //                 }
    //             }
    //         }
    //     }
    //     catch (const std::exception& e)
    //     {
    //         error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    //     }
    // }

    void personEntriesFromOP(std::unordered_map<int, PersonTrackerEntry>& personEntries,
                             const Array<float>& poseKeypoints, const Array<long long>& poseIds,
                             float confidenceThreshold)
    {
        try
        {
            personEntries.clear();
            for (int i=0; i<poseKeypoints.getSize()[0]; i++)
            {
                const auto id = poseIds[i];
                personEntries[id] = PersonTrackerEntry();
                personEntries[id].keypoints.resize(poseKeypoints.getSize()[1]);
                personEntries[id].status.resize(poseKeypoints.getSize()[1]);
                for (int j=0; j<poseKeypoints.getSize()[1]; j++)
                {
                    personEntries[id].keypoints[j].x = poseKeypoints[
                            i*poseKeypoints.getSize()[1]*poseKeypoints.getSize()[2] +
                            j*poseKeypoints.getSize()[2] + 0];
                    personEntries[id].keypoints[j].y = poseKeypoints[
                            i*poseKeypoints.getSize()[1]*poseKeypoints.getSize()[2] +
                            j*poseKeypoints.getSize()[2] + 1];
                    const float prob = poseKeypoints[
                            i*poseKeypoints.getSize()[1]*poseKeypoints.getSize()[2] +
                            j*poseKeypoints.getSize()[2] + 2];
                    if (prob < confidenceThreshold)
                        personEntries[id].status[j] = 0;
                    else
                        personEntries[id].status[j] = 1;
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void syncPersonEntriesWithOP(std::unordered_map<int, PersonTrackerEntry>& personEntries,
                                 const Array<float>& poseKeypoints, const Array<long long>& poseIds,
                                 float confidenceThreshold, bool mergeResults)
    {
        try
        {
            if (poseIds.empty())
                return;

            // Delete
            for (auto kv = personEntries.cbegin(); kv != personEntries.cend() /* not hoisted */; /* no increment */)
            {
                bool exists = false;
                for (int i=0; i<poseIds.getSize()[0]; i++)
                {
                    const auto id = poseIds[i];
                    if (id == kv->first)
                        exists = true;
                }
                if (!exists)
                    personEntries.erase(kv++);
                else
                    ++kv;
            }

            // Update or Add
            for (int i=0; i<poseIds.getSize()[0]; i++)
            {
                const auto id = poseIds[i];

                // Update
                if (personEntries.count(id) && mergeResults){

                    PersonTrackerEntry& personEntry = personEntries[id];
                    for (int j=0; j<poseKeypoints.getSize()[1]; j++)
                    {
                        const float x = poseKeypoints[
                                i*poseKeypoints.getSize()[1]*poseKeypoints.getSize()[2] +
                                j*poseKeypoints.getSize()[2] + 0];
                        const float y = poseKeypoints[
                                i*poseKeypoints.getSize()[1]*poseKeypoints.getSize()[2] +
                                j*poseKeypoints.getSize()[2] + 1];
                        const float prob = poseKeypoints[
                                i*poseKeypoints.getSize()[1]*poseKeypoints.getSize()[2] +
                                j*poseKeypoints.getSize()[2] + 2];
                        const cv::Point lkPoint = personEntry.keypoints[j];
                        const cv::Point opPoint = cv::Point(x,y);

                        if (prob < confidenceThreshold)
                            personEntries[id].status[j] = 0;
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
                else
                {
                    personEntries[id] = PersonTrackerEntry();
                    personEntries[id].keypoints.resize(poseKeypoints.getSize()[1]);
                    personEntries[id].status.resize(poseKeypoints.getSize()[1]);
                    for (int j=0; j<poseKeypoints.getSize()[1]; j++)
                    {
                        personEntries[id].keypoints[j].x = poseKeypoints[
                                i*poseKeypoints.getSize()[1]*poseKeypoints.getSize()[2] +
                                j*poseKeypoints.getSize()[2] + 0];
                        personEntries[id].keypoints[j].y = poseKeypoints[
                                i*poseKeypoints.getSize()[1]*poseKeypoints.getSize()[2] +
                                j*poseKeypoints.getSize()[2] + 1];
                        const float prob = poseKeypoints[
                                i*poseKeypoints.getSize()[1]*poseKeypoints.getSize()[2] +
                                j*poseKeypoints.getSize()[2] + 2];
                        if (prob < confidenceThreshold)
                            personEntries[id].status[j] = 0;
                        else
                            personEntries[id].status[j] = 1;
                    }

                }
            }

            // Sanity Check Start
            if ((int)personEntries.size() != poseIds.getSize()[0])
            {
                // Print
                for (auto& kv : personEntries)
                    std::cout << kv.first << " ";
                std::cout << std::endl;
                for (int i=0; i<poseIds.getSize()[0]; i++)
                    std::cout << poseIds.at(i) << " ";
                std::cout << std::endl;
                std::cout << "---" << std::endl;
                error("Size Mismatch. THere is an error in your poseId formatting.",
                      __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void opFromPersonEntries(Array<float>& poseKeypoints,
                             const std::unordered_map<int, PersonTrackerEntry>& personEntries,
                             const Array<long long>& poseIds)
    {
        try
        {
            if (!personEntries.size() || poseIds.empty())
                return;
            int dims[] = { (int)personEntries.size(), (int)personEntries.begin()->second.keypoints.size(), 3 };
            cv::Mat opArrayMat(3,dims,CV_32FC1);
            for (auto i=0; i<poseIds.getSize()[0]; i++)
            {
                const int id = poseIds[i];
                const PersonTrackerEntry& pe = personEntries.at(id);
                for (int j=0; j<dims[1]; j++)
                {
                    const auto baseIndex = i*dims[1]*dims[2] + j*dims[2];
                    opArrayMat.at<float>(baseIndex + 0) = pe.keypoints[j].x;
                    opArrayMat.at<float>(baseIndex + 1) = pe.keypoints[j].y;
                    opArrayMat.at<float>(baseIndex + 2) = (int)pe.status[j];
                    if (pe.keypoints[j].x == 0 && pe.keypoints[j].y == 0)
                        opArrayMat.at<float>(baseIndex + 2) = 0;
                }
            }
            poseKeypoints.setFrom(opArrayMat);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void scaleKeypoints(std::unordered_map<int, PersonTrackerEntry>& personEntries, float xScale, float yScale)
    {
        for (auto& kv : personEntries)
        {
            for(size_t i=0; i<kv.second.keypoints.size(); i++)
            {
                kv.second.keypoints[i].x *= xScale;
                kv.second.keypoints[i].y *= yScale;
            }
        }
    }

    PersonTracker::PersonTracker(const bool mergeResults, const int levels,
                                 const int patchSize, const float confidenceThreshold,
                                 const bool trackVelocity, const bool scaleVarying,
                                 const float rescale) :
        mMergeResults{mergeResults},
        mLevels{levels},
        mPatchSize{patchSize},
        mTrackVelocity{trackVelocity},
        mConfidenceThreshold{confidenceThreshold},
        mScaleVarying{scaleVarying},
        mRescale{rescale},
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
            /*
             * 1. Get poseKeypoints for all people - Checks
             * 2. If last image is empty or mPersonEntries is empty (and poseKeypoints and poseIds has data or crash it)
             *      Create mPersonEntries referencing poseIds
             *      Initialize LK points
             * 3. If poseKeypoints is not empty and poseIds has data
             *      1. Update LK
             *      2. CRUD/Sync - Check mMergeResults flag to smooth or not
             * 4. If poseKeypoints is empty
             *      1. Update LK
             *      2. replace poseKeypoints
             */

            // TODO: This case: if mMergeResults == false --> Run LK tracker ONLY IF poseKeypoints.empty() doesn't consider the case
            // of poseKeypoints being empty BECAUSE there were no people on the image

            // if mMergeResults == true --> Combine OP + LK tracker
            // if mMergeResults == false --> Run LK tracker ONLY IF poseKeypoints.empty()

            bool mergeResults = mMergeResults;
            mergeResults = true;

            // Sanity Checks
            if (poseKeypoints.getSize(0) != poseIds.getSize(0))
                 error("poseKeypoints and poseIds should have the same number of people",
                       __LINE__, __FUNCTION__, __FILE__);

            // First frame
            if (mImagePrevious.empty())
            {
                // Create mPersonEntries
                personEntriesFromOP(mPersonEntries, poseKeypoints, poseIds, mConfidenceThreshold);
                // Capture current frame as floating point
                cvMatInput.convertTo(mImagePrevious, CV_8UC3);
                // Rescale
                if(mRescale)
                {
                    cv::Size rescaleSize(mRescale,mImagePrevious.size().height/(mImagePrevious.size().width/mRescale));
                    cv::resize(mImagePrevious, mImagePrevious, rescaleSize, 0, 0, cv::INTER_CUBIC);
                }
                // Save Last Ids
                mLastPoseIds = poseIds.clone();
            }
            // Any other frame
            else
            {
                // Update LK
                const bool newOPData = !poseKeypoints.empty() && !poseIds.empty();
                if ((newOPData && mergeResults) || (!newOPData))
                {
                    cv::Mat imageCurrent;
                    std::vector<cv::Mat> pyramidImagesCurrent;
                    cvMatInput.convertTo(imageCurrent, CV_8UC3);
                    float xScale = 1., yScale = 1.;
                    if(mRescale)
                    {
                        cv::Size rescaleSize(mRescale,imageCurrent.size().height/(imageCurrent.size().width/mRescale));
                        xScale = (float)imageCurrent.size().width / (float)rescaleSize.width;
                        yScale = (float)imageCurrent.size().height / (float)rescaleSize.height;
                        cv::resize(imageCurrent, imageCurrent, rescaleSize, 0, 0, cv::INTER_CUBIC);
                    }
                    scaleKeypoints(mPersonEntries, 1./xScale, 1./yScale);
                    updateLK(mPersonEntries, mPyramidImagesPrevious, pyramidImagesCurrent, mImagePrevious,
                             imageCurrent, mLevels, mPatchSize, mTrackVelocity, mScaleVarying);
                    scaleKeypoints(mPersonEntries, xScale, yScale);
                    mImagePrevious = imageCurrent;
                    mPyramidImagesPrevious = pyramidImagesCurrent;
                }

                // There is new OP Data
                if (newOPData)
                {
                    mLastPoseIds = poseIds.clone();
                    syncPersonEntriesWithOP(mPersonEntries, poseKeypoints, mLastPoseIds, mConfidenceThreshold,
                                            mergeResults);
                    opFromPersonEntries(poseKeypoints, mPersonEntries, mLastPoseIds);
                }
                // There is no new OP Data
                else
                {
                    opFromPersonEntries(poseKeypoints, mPersonEntries, mLastPoseIds);
                    poseIds = mLastPoseIds.clone();
                }
            }

            // cv::Mat debugImage = cvMatInput.clone();
            // vizPersonEntries(debugImage, mPersonEntries, mTrackVelocity);
            // cv::imshow("win", debugImage);
            // cv::waitKey(15);
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

    bool PersonTracker::getMergeResults() const
    {
        try
        {
            return mMergeResults;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }
}
