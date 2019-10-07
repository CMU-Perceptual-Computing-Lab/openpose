#ifndef OPENPOSE_TRACKING_PERSON_ID_EXTRACTOR_HPP
#define OPENPOSE_TRACKING_PERSON_ID_EXTRACTOR_HPP

#include <openpose/core/common.hpp>

namespace op
{
    class OP_API PersonIdExtractor
    {
    public:
        PersonIdExtractor(const float confidenceThreshold = 0.1f, const float inlierRatioThreshold = 0.5f,
                          const float distanceThreshold = 30.f, const int numberFramesToDeletePerson = 10);

        virtual ~PersonIdExtractor();

        Array<long long> extractIds(const Array<float>& poseKeypoints, const Matrix& cvMatInput,
                                    const unsigned long long imageViewIndex = 0ull);

        Array<long long> extractIdsLockThread(const Array<float>& poseKeypoints, const Matrix& cvMatInput,
                                              const unsigned long long imageViewIndex,
                                              const long long frameId);

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplPersonIdExtractor;
        std::shared_ptr<ImplPersonIdExtractor> spImpl;

        DELETE_COPY(PersonIdExtractor);
    };
}

#endif // OPENPOSE_TRACKING_PERSON_ID_EXTRACTOR_HPP
