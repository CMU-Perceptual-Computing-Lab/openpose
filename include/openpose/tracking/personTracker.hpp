#ifndef OPENPOSE_OPENPOSE_PRIVATE_TRACKING_PERSON_TRACKER_HPP
#define OPENPOSE_OPENPOSE_PRIVATE_TRACKING_PERSON_TRACKER_HPP

#include <openpose/core/common.hpp>

namespace op
{
    class OP_API PersonTracker
    {
    public:
        PersonTracker(const bool mergeResults, const int levels = 3, const int patchSize = 31,
                      const float confidenceThreshold = 0.05f, const bool trackVelocity = false,
                      const bool scaleVarying = false, const float rescale = 640);

        virtual ~PersonTracker();

        void track(Array<float>& poseKeypoints, Array<long long>& poseIds, const Matrix& cvMatInput);

        void trackLockThread(Array<float>& poseKeypoints, Array<long long>& poseIds, const Matrix& cvMatInput,
                             const long long frameId);

        bool getMergeResults() const;

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplPersonTracker;
        std::shared_ptr<ImplPersonTracker> spImpl;

        DELETE_COPY(PersonTracker);
    };
}

#endif // OPENPOSE_OPENPOSE_PRIVATE_TRACKING_PERSON_TRACKER_HPP
