#ifndef OPENPOSE_HAND_HAND_DETECTOR_HPP
#define OPENPOSE_HAND_HAND_DETECTOR_HPP

#include <mutex>
#include <openpose/core/common.hpp>
#include <openpose/pose/enumClasses.hpp>

namespace op
{
    // Note: This class is thread-safe, so several GPUs can be running hands and using `updateTracker`, and updateTracker will keep the latest known
    // tracking
    class OP_API HandDetector
    {
    public:
        explicit HandDetector(const PoseModel poseModel);

        virtual ~HandDetector();

        std::vector<std::array<Rectangle<float>, 2>> detectHands(const Array<float>& poseKeypoints) const;

        std::vector<std::array<Rectangle<float>, 2>> trackHands(const Array<float>& poseKeypoints);

        void updateTracker(const std::array<Array<float>, 2>& handKeypoints, const unsigned long long id);

    private:
        enum class PosePart : unsigned int
        {
            LWrist = 0,
            LElbow,
            LShoulder,
            RWrist,
            RElbow,
            RShoulder,
            Size,
        };

        const std::array<unsigned int, (int)PosePart::Size> mPoseIndexes;
        std::vector<std::array<Point<float>, (int)PosePart::Size>> mPoseTrack;
        std::vector<Rectangle<float>> mHandLeftPrevious;
        std::vector<Rectangle<float>> mHandRightPrevious;
        unsigned long long mCurrentId;
        std::mutex mMutex;

        std::array<unsigned int, (int)PosePart::Size> getPoseKeypoints(const PoseModel poseModel,
                                                                       const std::array<std::string, (int)PosePart::Size>& poseStrings) const;

        DELETE_COPY(HandDetector);
    };
}

#endif // OPENPOSE_HAND_HAND_DETECTOR_HPP
