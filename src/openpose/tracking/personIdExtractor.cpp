#include <openpose/tracking/pyramidalLK.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/tracking/personIdExtractor.hpp>

// #define LK_CUDA

namespace op
{
    const std::string errorMessage = "ID extractor function (`--identification` flag) not implemented"
                                     " for multiple-view processing.";

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

    std::vector<PersonEntry> captureKeypoints(const Array<float>& poseKeypoints, const float confidenceThreshold)
    {
        try
        {
            // Define result
            std::vector<PersonEntry> personEntries(poseKeypoints.getSize(0));
            // Fill personEntries
            for (auto p = 0; p < (int)personEntries.size(); p++)
            {
                // Create person entry in the tracking map
                auto& personEntry = personEntries[p];
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
            }
            // Return result
            return personEntries;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    void updateLK(std::unordered_map<int,PersonEntry>& personEntries, std::vector<cv::Mat>& pyramidImagesPrevious,
                  std::vector<cv::Mat>& pyramidImagesCurrent, const cv::Mat& imagePrevious,
                  const cv::Mat& imageCurrent, const int numberFramesToDeletePerson)
    {
        try
        {
            // Get all key values
            // Otherwise, `erase` provokes core dumped when removing elements
            std::vector<int> keyValues;
            keyValues.reserve(personEntries.size());
            for (const auto& entry : personEntries)
                keyValues.emplace_back(entry.first);
            // Update or remove elements
            for (auto& key : keyValues)
            {
                auto& element = personEntries[key];

                // Remove keypoint
                if (element.counterLastDetection++ > numberFramesToDeletePerson)
                    personEntries.erase(key);
                // Update all keypoints for that entry
                else
                {
                    PersonEntry personEntry;
                    personEntry.counterLastDetection = element.counterLastDetection;
                    #ifdef LK_CUDA
                        UNUSED(pyramidImagesPrevious);
                        UNUSED(pyramidImagesCurrent);
                        pyramidalLKGpu(element.keypoints, personEntry.keypoints, element.status,
                                       imagePrevious, imageCurrent, 3, 21);
                    #else
                        pyramidalLKCpu(element.keypoints, personEntry.keypoints, pyramidImagesPrevious,
                                       pyramidImagesCurrent, element.status, imagePrevious, imageCurrent, 3, 21);
                    #endif
                    personEntry.status = element.status;
                    element = personEntry;
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void initializeLK(std::unordered_map<int,PersonEntry>& personEntries,
                      long long& mNextPersonId,
                      const Array<float>& poseKeypoints,
                      const float confidenceThreshold)
    {
        try
        {
            for (auto p = 0; p < poseKeypoints.getSize(0); p++)
            {
                const auto currentPerson = int(mNextPersonId++);

                // Create person entry in the tracking map
                auto& personEntry = personEntries[currentPerson];
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

    bool compareCandidates(std::tuple<float, int, int> a, std::tuple<float, int, int> b)
    {
        return std::get<0>(a) > std::get<0>(b);
    }

    Array<long long> matchLKAndOPGreedy(std::unordered_map<int,PersonEntry>& personEntries,
                                        long long& nextPersonId,
                                        const std::vector<PersonEntry>& openposePersonEntries,
                                        const cv::Mat& imagePrevious,
                                        const float inlierRatioThreshold,
                                        const float distanceThreshold)
    {
        try
        {
            Array<long long> poseIds{(int)openposePersonEntries.size(), -1};
            std::unordered_map<int, PersonEntry> pendingQueue;
            std::vector<bool> processed((int)openposePersonEntries.size(), false);
            std::unordered_set<int> used;
            bool converged = false;

            while (!openposePersonEntries.empty() && !converged)
            {
                const auto numberKeypoints = openposePersonEntries[0].keypoints.size();
                std::vector<std::tuple<float, int, int>> candidates;
                float bestScore = 0.0f;
                converged = true;

                for (auto i = 0u; i < openposePersonEntries.size(); i++)
                {
                    if (poseIds.at(i) != -1)
                        continue;

                    const auto& openposePersonEntry = openposePersonEntries.at(i);
                    const auto personDistanceThreshold = fastMax(10.f,
                        distanceThreshold*float(std::sqrt(imagePrevious.cols*imagePrevious.rows)) / 960.f);

                    // Find best correspondance in the LK set
                    for (const auto& personEntry : personEntries)
                    {
                        if (used.find(personEntry.first) != used.end())
                            continue;

                        const auto& element = personEntry.second;
                        auto inliers = 0;
                        auto active = 0;
                        auto distance = 0.f;
                        auto total_distance = 0.0f;

                        // Sanity checks
                        if (element.status.size() != numberKeypoints)
                            error("element.status.size() != numberKeypoints ||", __LINE__, __FUNCTION__, __FILE__);
                        if (openposePersonEntry.status.size() != numberKeypoints)
                            error("openposePersonEntry.status.size() != numberKeypoints",
                                  __LINE__, __FUNCTION__, __FILE__);
                        if (element.keypoints.size() != numberKeypoints)
                            error("element.keypoints.size() != numberKeypoints ||", __LINE__, __FUNCTION__, __FILE__);
                        if (openposePersonEntry.keypoints.size() != numberKeypoints)
                            error("openposePersonEntry.keypoints.size() != numberKeypoints",
                                  __LINE__, __FUNCTION__, __FILE__);

                        // Iterate through all keypoints
                        for (auto kp = 0u; kp < numberKeypoints; kp++)
                        {
                            // If enough threshold
                            if (!element.status[kp] && !openposePersonEntry.status[kp])
                            {
                                active++;
                                distance = getEuclideanDistance(element.keypoints[kp],
                                                                openposePersonEntry.keypoints[kp]);
                                total_distance += distance;

                                if (distance < personDistanceThreshold)
                                    inliers++;
                            }
                        }

                        if (active > 0)
                        {
                            const auto score = inliers / (float)active;

                            if (score == bestScore && score >= inlierRatioThreshold)
                            {
                                candidates.push_back(std::make_tuple(total_distance,i, personEntry.first));
                                bestScore = score;
                            }
                            else if (score > bestScore && score >= inlierRatioThreshold)
                            {
                                bestScore = score;
                                candidates.clear();
                                candidates.push_back(std::make_tuple(total_distance, i, personEntry.first));
                            }
                        }
                    }
                }
                std::sort(candidates.begin(), candidates.end(), compareCandidates);

                while (candidates.size())
                {
                    auto top_candidate = candidates.back();
                    candidates.pop_back();
                    auto idx_lk = std::get<2>(top_candidate);
                    auto idx_op = std::get<1>(top_candidate);

                    if (used.find(idx_lk) != used.end())
                        continue;

                    poseIds[idx_op] = idx_lk;
                    used.insert(idx_lk);
                    converged = false;

                }
            }
            for (auto i = 0u; i < openposePersonEntries.size(); i++)
            {
                if (poseIds[i] == -1)
                    poseIds[i] = nextPersonId++;
                const auto& openposePersonEntry = openposePersonEntries.at(i);
                personEntries[(int)poseIds[i]] = openposePersonEntry;
            }

            return poseIds;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<long long>{};
        }
    }


    // Array<long long> matchLKAndOP(std::unordered_map<int,PersonEntry>& personEntries,
    //                               long long& nextPersonId,
    //                               const std::vector<PersonEntry>& openposePersonEntries,
    //                               const cv::Mat& imagePrevious,
    //                               const float inlierRatioThreshold,
    //                               const float distanceThreshold)
    // {
    //     try
    //     {
    //         Array<long long> poseIds{(int)openposePersonEntries.size(), -1};
    //         std::unordered_map<int, PersonEntry> pendingQueue;

    //         if (!openposePersonEntries.empty())
    //         {
    //             const auto numberKeypoints = openposePersonEntries[0].keypoints.size();
    //             for (auto i = 0u; i < openposePersonEntries.size(); i++)
    //             {
    //                 auto& poseId = poseIds.at(i);

    //                 const auto& openposePersonEntry = openposePersonEntries.at(i);
    //                 const auto personDistanceThreshold = fastMax(10.f,
    //                     distanceThreshold*float(std::sqrt(imagePrevious.cols*imagePrevious.rows)) / 960.f);

    //                 // Find best correspondance in the LK set
    //                 auto bestMatch = -1ll;
    //                 auto bestScore = 0.f;
    //                 for (const auto& personEntry : personEntries)
    //                 {
    //                     const auto& element = personEntry.second;
    //                     auto inliers = 0;
    //                     auto active = 0;

    //                     // Sanity checks
    //                     if (element.status.size() != numberKeypoints)
    //                         error("element.status.size() != numberKeypoints ||", __LINE__, __FUNCTION__, __FILE__);
    //                     if (openposePersonEntry.status.size() != numberKeypoints)
    //                         error("openposePersonEntry.status.size() != numberKeypoints",
    //                               __LINE__, __FUNCTION__, __FILE__);
    //                     if (element.keypoints.size() != numberKeypoints)
    //                         error("element.keypoints.size() != numberKeypoints ||", __LINE__, __FUNCTION__, __FILE__);
    //                     if (openposePersonEntry.keypoints.size() != numberKeypoints)
    //                         error("openposePersonEntry.keypoints.size() != numberKeypoints",
    //                               __LINE__, __FUNCTION__, __FILE__);
    //                     // Iterate through all keypoints
    //                     for (auto kp = 0u; kp < numberKeypoints; kp++)
    //                     {
    //                         // If enough threshold
    //                         if (!element.status[kp] && !openposePersonEntry.status[kp])
    //                         {
    //                             active++;
    //                             const auto distance = getEuclideanDistance(element.keypoints[kp],
    //                                                                        openposePersonEntry.keypoints[kp]);
    //                             if (distance < personDistanceThreshold)
    //                                 inliers++;
    //                         }
    //                     }

    //                     if (active > 0)
    //                     {
    //                         const auto score = inliers / (float)active;
    //                         if (score > bestScore && score >= inlierRatioThreshold)
    //                         {
    //                             bestScore = score;
    //                             bestMatch = personEntry.first;
    //                         }
    //                     }
    //                 }
    //                 // Found a best match, update LK table and poseIds
    //                 if (bestMatch != -1)
    //                     poseId = bestMatch;
    //                 else
    //                     poseId = nextPersonId++;

    //                 pendingQueue[poseId] = openposePersonEntry;
    //             }
    //         }

    //         // Update LK table with pending queue
    //         for (auto& pendingQueueEntry: pendingQueue)
    //             personEntries[pendingQueueEntry.first] = pendingQueueEntry.second;

    //         return poseIds;
    //     }
    //     catch (const std::exception& e)
    //     {
    //         error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    //         return Array<long long>{};
    //     }
    // }

    PersonIdExtractor::PersonIdExtractor(const float confidenceThreshold, const float inlierRatioThreshold,
                                         const float distanceThreshold, const int numberFramesToDeletePerson) :
        mConfidenceThreshold{confidenceThreshold},
        mInlierRatioThreshold{inlierRatioThreshold},
        mDistanceThreshold{distanceThreshold},
        mNumberFramesToDeletePerson{numberFramesToDeletePerson},
        mNextPersonId{0ll},
        mLastFrameId{-1ll}
    {
        try
        {
            error("PersonIdExtractor (`identification` flag) buggy and not working yet, but we are working on it!"
                  " Coming soon!", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    PersonIdExtractor::~PersonIdExtractor()
    {
    }

    Array<long long> PersonIdExtractor::extractIds(const Array<float>& poseKeypoints, const cv::Mat& cvMatInput,
                                                   const unsigned long long imageViewIndex)
    {
        try
        {
            // Sanity check
            if (imageViewIndex > 0)
                error(errorMessage, __LINE__, __FUNCTION__, __FILE__);

            // Result initialization
            Array<long long> poseIds;
            const auto openposePersonEntries = captureKeypoints(poseKeypoints, mConfidenceThreshold);
// log(mPersonEntries.size());

            // First frame
            if (mImagePrevious.empty())
            {
                // Add first persons to the LK set
                initializeLK(mPersonEntries, mNextPersonId, poseKeypoints, mConfidenceThreshold);
                // Capture current frame as floating point
                cvMatInput.convertTo(mImagePrevious, CV_32F);
            }
            // Rest
            else
            {
                cv::Mat imageCurrent;
                std::vector<cv::Mat> pyramidImagesCurrent;
                cvMatInput.convertTo(imageCurrent, CV_32F);
                updateLK(mPersonEntries, mPyramidImagesPrevious, pyramidImagesCurrent, mImagePrevious, imageCurrent,
                         mNumberFramesToDeletePerson);
                mImagePrevious = imageCurrent;
                mPyramidImagesPrevious = pyramidImagesCurrent;
            }

            // Get poseIds and update LKset according to OpenPose set
            // poseIds = matchLKAndOP(
            poseIds = matchLKAndOPGreedy(
                mPersonEntries, mNextPersonId, openposePersonEntries, mImagePrevious, mInlierRatioThreshold,
                mDistanceThreshold);

            return poseIds;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<long long>{};
        }
    }

    Array<long long> PersonIdExtractor::extractIdsLockThread(const Array<float>& poseKeypoints,
                                                             const cv::Mat& cvMatInput,
                                                             const unsigned long long imageViewIndex,
                                                             const long long frameId)
    {
        try
        {
            // Sanity check
            if (imageViewIndex > 0)
                error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
            // Wait for desired order
            while (mLastFrameId < frameId - 1)
                std::this_thread::sleep_for(std::chrono::microseconds{100});
            // Extract IDs
            const auto ids = extractIds(poseKeypoints, cvMatInput, imageViewIndex);
            // Update last frame id
            mLastFrameId = frameId;
            // Return person ids
            return ids;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<long long>{};
        }
    }
}
