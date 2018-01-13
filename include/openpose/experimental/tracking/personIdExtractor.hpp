#ifndef OPENPOSE_TRACKING_PERSON_ID_EXTRACTOR_HPP
#define OPENPOSE_TRACKING_PERSON_ID_EXTRACTOR_HPP

#include <openpose/core/common.hpp>

namespace op
{
    class OP_API PersonIdExtractor
    {
    public:
        PersonIdExtractor();

        virtual ~PersonIdExtractor();

        Array<long long> extractIds(const Array<float>& poseKeypoints);

    private:
    	long long mNextPersonId;

        DELETE_COPY(PersonIdExtractor);
    };
}

#endif // OPENPOSE_TRACKING_PERSON_ID_EXTRACTOR_HPP
