#include <openpose/experimental/tracking/personIdExtractor.hpp>

namespace op
{
    PersonIdExtractor::PersonIdExtractor() :
        mNextPersonId{0ll}
    {
        try
        {
            error("PersonIdExtractor (`identification` flag) not available yet, but we are working on it! Coming"
                  " soon!", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    PersonIdExtractor::~PersonIdExtractor()
    {
    }

    Array<long long> PersonIdExtractor::extractIds(const Array<float>& poseKeypoints)
    {
        try
        {
            // Dummy: giving a new id to each element
            Array<long long> poseIds{poseKeypoints.getSize(0), -1};
            for (auto i = 0u ; i < poseIds.getVolume() ; i++)
                poseIds[i] = mNextPersonId++;
            return poseIds;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<long long>{};
        }
    }
}
