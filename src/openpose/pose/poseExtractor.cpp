#include <openpose/pose/poseExtractor.hpp>

namespace op
{
    const std::string errorMessage = "Either person identification (`--identification`) must be enabled or"
                                     " `--number_people_max 1` in order to run the person tracker (`--tracking`).";

    PoseExtractor::PoseExtractor(const std::shared_ptr<PoseExtractorNet>& poseExtractorNet,
                                 const std::shared_ptr<KeepTopNPeople>& keepTopNPeople,
                                 const std::shared_ptr<PersonIdExtractor>& personIdExtractor,
                                 const std::shared_ptr<std::vector<std::shared_ptr<PersonTracker>>>& personTrackers,
                                 const int numberPeopleMax, const int tracking) :
        mNumberPeopleMax{numberPeopleMax},
        mTracking{tracking},
        spPoseExtractorNet{poseExtractorNet},
        spKeepTopNPeople{keepTopNPeople},
        spPersonIdExtractor{personIdExtractor},
        spPersonTrackers{personTrackers}
    {
    }

    PoseExtractor::~PoseExtractor()
    {
    }

    void PoseExtractor::initializationOnThread()
    {
        try
        {
            spPoseExtractorNet->initializationOnThread();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PoseExtractor::forwardPass(const std::vector<Array<float>>& inputNetData,
                                    const Point<int>& inputDataSize,
                                    const std::vector<double>& scaleInputToNetInputs,
                                    const Array<float>& poseNetOutput,
                                    const long long frameId)
    {
        try
        {
            if (mTracking < 1 || frameId % (mTracking+1) == 0)
                spPoseExtractorNet->forwardPass(inputNetData, inputDataSize, scaleInputToNetInputs, poseNetOutput);
            else
                spPoseExtractorNet->clear();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Array<float> PoseExtractor::getHeatMapsCopy() const
    {
        try
        {
            return spPoseExtractorNet->getHeatMapsCopy();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }

    std::vector<std::vector<std::array<float, 3>>> PoseExtractor::getCandidatesCopy() const
    {
        try
        {
            return spPoseExtractorNet->getCandidatesCopy();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::vector<std::vector<std::array<float,3>>>{};
        }
    }

    Array<float> PoseExtractor::getPoseKeypoints() const
    {
        try
        {
            return spPoseExtractorNet->getPoseKeypoints();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }

    Array<float> PoseExtractor::getPoseScores() const
    {
        try
        {
            return spPoseExtractorNet->getPoseScores();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }

    float PoseExtractor::getScaleNetToOutput() const
    {
        try
        {
            return spPoseExtractorNet->getScaleNetToOutput();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.;
        }
    }

    void PoseExtractor::keepTopPeople(Array<float>& poseKeypoints, const Array<float>& poseScores) const
    {
        try
        {
            // Keep only top N people
            if (spKeepTopNPeople)
                poseKeypoints = spKeepTopNPeople->keepTopPeople(poseKeypoints, poseScores);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Array<long long> PoseExtractor::extractIds(const Array<float>& poseKeypoints, const Matrix& cvMatInput,
                                               const unsigned long long imageViewIndex)
    {
        try
        {
            // Run person ID extractor
            return (spPersonIdExtractor
                ? spPersonIdExtractor->extractIds(poseKeypoints, cvMatInput, imageViewIndex)
                : Array<long long>{poseKeypoints.getSize(0), -1});
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<long long>{};
        }
    }

    Array<long long> PoseExtractor::extractIdsLockThread(const Array<float>& poseKeypoints,
                                                         const Matrix& cvMatInput,
                                                         const unsigned long long imageViewIndex,
                                                         const long long frameId)
    {
        try
        {
            // Run person ID extractor
            return (spPersonIdExtractor
                ? spPersonIdExtractor->extractIdsLockThread(poseKeypoints, cvMatInput, imageViewIndex, frameId)
                : Array<long long>{poseKeypoints.getSize(0), -1});
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<long long>{};
        }
    }

    void PoseExtractor::track(Array<float>& poseKeypoints, Array<long long>& poseIds,
                              const Matrix& cvMatInput,
                              const unsigned long long imageViewIndex)
    {
        try
        {
            if (!spPersonTrackers->empty())
            {
                // Resize if required
                while (spPersonTrackers->size() <= imageViewIndex)
                    spPersonTrackers->emplace_back(std::make_shared<PersonTracker>(
                        (*spPersonTrackers)[0]->getMergeResults()));
                // Sanity check
                if (!poseKeypoints.empty() && poseIds.empty() && mNumberPeopleMax != 1)
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
                // Reset poseIds if keypoints is empty
                if (poseKeypoints.empty())
                    poseIds.reset();
                // Run person tracker
                if (spPersonTrackers->at(imageViewIndex))
                    (*spPersonTrackers)[imageViewIndex]->track(poseKeypoints, poseIds, cvMatInput);
                // Run person tracker
                (*spPersonTrackers)[imageViewIndex]->track(poseKeypoints, poseIds, cvMatInput);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PoseExtractor::trackLockThread(Array<float>& poseKeypoints, Array<long long>& poseIds,
                                        const Matrix& cvMatInput,
                                        const unsigned long long imageViewIndex, const long long frameId)
    {
        try
        {
            if (!spPersonTrackers->empty())
            {
                // Resize if required
                while (spPersonTrackers->size() <= imageViewIndex)
                    spPersonTrackers->emplace_back(std::make_shared<PersonTracker>(
                        (*spPersonTrackers)[0]->getMergeResults()));
                // Sanity check
                if (!poseKeypoints.empty() && poseIds.empty() && mNumberPeopleMax != 1)
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
                // Reset poseIds if keypoints is empty
                if (poseKeypoints.empty())
                    poseIds.reset();
                // Run person tracker
                if (spPersonTrackers->at(imageViewIndex))
                    (*spPersonTrackers)[imageViewIndex]->trackLockThread(
                        poseKeypoints, poseIds, cvMatInput, frameId);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
