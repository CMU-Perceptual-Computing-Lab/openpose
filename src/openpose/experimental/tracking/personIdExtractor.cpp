#include <iostream>
#include <openpose/experimental/tracking/pyramidalLK.hpp>
#include <openpose/experimental/tracking/personIdExtractor.hpp>

// #define LK_CUDA

namespace op
{
    float getEuclideanDistance(const cv::Point2f& a, const cv::Point2f& b)
    {
        try
        {
            const auto difference = a - b;
            return std::sqrt(difference.x * difference.x + difference.y * difference.y);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    void captureKeypoints(std::vector<PersonEntry>& personEntries, const Array<float>& poseKeypoints,
                          const float confidenceThreshold)
    {
        try
        {
            personEntries.clear();

            for (auto p = 0; p < poseKeypoints.getSize(0); p++)
            {
                // Create person entry in the tracking map
                PersonEntry personEntry;
                auto& keypoints = personEntry.keypoints;
                auto& status = personEntry.status;
                personEntry.counterLastDetection = 0;

                for (auto kp = 0; kp < poseKeypoints.getSize(1); kp++)
                {
                    cv::Point2f cp;
                    cp.x = poseKeypoints[{p,kp,0}];
                    cp.y = poseKeypoints[{p,kp,1}];
                    keypoints.emplace_back(cp);

                    if (poseKeypoints[{p,kp,2}] < confidenceThreshold)
                        status.emplace_back(1);
                    else
                        status.emplace_back(0);
                }

                // Add person entry in the tracking map
                personEntries.emplace_back(personEntry);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void updateLK(std::unordered_map<int,PersonEntry>& mPointsLK, std::vector<cv::Mat>& pyramidImagesPrevious,
                  std::vector<cv::Mat>& pyramidImagesCurrent, const cv::Mat& imagePrevious,
                  const cv::Mat& imageCurrent, const int numberFramesToDeletePerson)
    {
        try
        {
            for (auto& entry: mPointsLK)
            {
                int idx = entry.first;

                if (mPointsLK[idx].counterLastDetection++ > numberFramesToDeletePerson)
                {
                    mPointsLK.erase(idx);
                    continue;
                }

                // Update all keypoints for that entry
                PersonEntry personEntry;
                #ifdef LK_CUDA
                    UNUSED(pyramidImagesPrevious);
                    UNUSED(pyramidImagesCurrent);
                    pyramidalLKGpu(mPointsLK[idx].keypoints, personEntry.keypoints, mPointsLK[idx].status,
                                   imagePrevious, imageCurrent, 3, 21);
                #else
                    pyramidalLKCpu(mPointsLK[idx].keypoints, personEntry.keypoints, pyramidImagesPrevious,
                                   pyramidImagesCurrent, mPointsLK[idx].status, imagePrevious, imageCurrent, 3, 21);
                #endif
                personEntry.status = mPointsLK[idx].status;
                mPointsLK[idx] = personEntry;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void initializeLK(std::unordered_map<int,PersonEntry>& mPointsLK,
                     long long& mNextPersonId,
                     const Array<float>& poseKeypoints,
                     const float confidenceThreshold)
    {
        try
        {
            for (auto p = 0; p < poseKeypoints.getSize(0); p++)
            {
                const int currentPerson = mNextPersonId++;

                // Create person entry in the tracking map
                auto& personEntry = mPointsLK[currentPerson];
                auto& keypoints = personEntry.keypoints;
                auto& status = personEntry.status;
                personEntry.counterLastDetection = 0;

                for (auto kp = 0; kp < poseKeypoints.getSize(1); kp++)
                {
                    const cv::Point2f cp{poseKeypoints[{p,kp,0}], poseKeypoints[{p,kp,1}]};
                    keypoints.emplace_back(cp);

                    if (poseKeypoints[{p,kp,2}] < confidenceThreshold)
                        status.emplace_back(1);
                    else
                        status.emplace_back(0);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Array<long long> matchLKAndOP(std::unordered_map<int,PersonEntry>& mPointsLK,
                                  long long& mNextPersonId,
                                  const std::vector<PersonEntry>& openposePersonEntries,
                                  const float inlierRatioThreshold,
                                  const float distanceThreshold)
    {
        try
        {
            Array<long long> poseIds{(int)openposePersonEntries.size(), -1};
            std::unordered_map<int,PersonEntry> pendingQueue;

            if (!openposePersonEntries.empty())
            {
                const auto numberKeypoints = openposePersonEntries[0].keypoints.size();
                for (auto i = 0u; i < openposePersonEntries.size(); i++)
                {
                    auto bestMatch = -1ll;
                    auto bestScore = 0.f;

                    // Find best correspondance in the LK set
                    for (auto& entry_lk : mPointsLK)
                    {
                        auto inliers = 0;
                        auto active = 0;
                        int idx = entry_lk.first;

                        // Iterate through all keypoints
                        for (auto kp = 0u; kp < numberKeypoints; kp++)
                        {
                            // Not enough threshold
                            if (mPointsLK.at(idx).status.at(kp) || openposePersonEntries.at(i).status.at(kp))
                                continue;

                            active++;
                            const auto dist = getEuclideanDistance(mPointsLK.at(idx).keypoints.at(kp),
                                                                   openposePersonEntries.at(i).keypoints.at(kp));
                            // std::cout<<dist<<std::endl;
                            if (dist < distanceThreshold)
                                inliers ++;
                        }

                        float score = 0.f;

                        if (active)
                            score = inliers / (float)active;

                        //std::cout<<inliers<<std::endl;

                        if (score >= inlierRatioThreshold && score > bestScore)
                        {
                            bestScore = score;
                            bestMatch = entry_lk.first;
                            // std::cout<<"BEST MATCH ENCOUNTERED"<<std::endl;
                        }
                    }
                    // Found a best match, update LK table and poseIds
                    if (bestMatch != -1)
                        poseIds.at(i) = bestMatch;
                    else
                        poseIds.at(i) = mNextPersonId++;

                    pendingQueue[poseIds.at(i)] = openposePersonEntries.at(i);
                }
            }

            // Update LK table with pending queue
            for (auto& pendingQueueEntry: pendingQueue)
                mPointsLK[pendingQueueEntry.first] = pendingQueueEntry.second;

            return poseIds;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<long long>{};
        }
    }

    PersonIdExtractor::PersonIdExtractor(const float confidenceThreshold, const float inlierRatioThreshold,
                                         const float distanceThreshold, const int numberFramesToDeletePerson) :
        mConfidenceThreshold{confidenceThreshold},
        mInlierRatioThreshold{inlierRatioThreshold},
        mDistanceThreshold{distanceThreshold},
        mNumberFramesToDeletePerson{numberFramesToDeletePerson},
        mNextPersonId{0ll}
    {
    }

    PersonIdExtractor::~PersonIdExtractor()
    {
    }

    Array<long long> PersonIdExtractor::extractIds(const Array<float>& poseKeypoints, const cv::Mat& cvMatInput)
    {
        try
        {
            Array<long long> poseIds;
            std::vector<PersonEntry> openposePersonEntries;
            captureKeypoints(openposePersonEntries, poseKeypoints, mConfidenceThreshold);

            // First frame
            if (mImagePrevious.empty())
            {
                // Add first persons to the LK set
                initializeLK(mPointsLK, mNextPersonId, poseKeypoints, mConfidenceThreshold);

                // Capture current frame as floating point
                cvMatInput.convertTo(mImagePrevious, CV_32F);
            }
            // Rest
            else
            {
                cv::Mat imageCurrent;
                std::vector<cv::Mat> pyramidImagesCurrent;
                cvMatInput.convertTo(imageCurrent, CV_32F);
                updateLK(mPointsLK, mPyramidImagesPrevious, pyramidImagesCurrent, mImagePrevious, imageCurrent,
                         mNumberFramesToDeletePerson);
                mImagePrevious = imageCurrent;
                mPyramidImagesPrevious = pyramidImagesCurrent;
            }

            // Get poseIds and update LKset according to OpenPose set
            poseIds = matchLKAndOP(mPointsLK, mNextPersonId, openposePersonEntries, mInlierRatioThreshold,
                                   mDistanceThreshold);

            return poseIds;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<long long>{};
        }
    }
}
