#ifndef OPENPOSE_HAND_HAND_DETECTOR_HPP
#define OPENPOSE_HAND_HAND_DETECTOR_HPP

#include <array>
#include <mutex>
#include <vector>
#include <openpose/core/array.hpp>
#include <openpose/core/point.hpp>
#include <openpose/core/rectangle.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/utilities/macros.hpp>
#include "enumClasses.hpp"

namespace op
{
    // Note: This class is thread-safe, so several GPUs can be running hands and using `updateTracker`, and updateTracker will keep the latest known
    // tracking
    class HandDetector
    {
    public:
        explicit HandDetector(const PoseModel poseModel);

        std::vector<std::array<Rectangle<float>, 2>> detectHands(const Array<float>& poseKeypoints, const float scaleInputToOutput) const;

        std::vector<std::array<Rectangle<float>, 2>> trackHands(const Array<float>& poseKeypoints, const float scaleInputToOutput);

        void updateTracker(const Array<float>& poseKeypoints, const std::array<Array<float>, 2>& handKeypoints);

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
