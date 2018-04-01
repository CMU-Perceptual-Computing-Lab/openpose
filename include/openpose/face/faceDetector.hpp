#ifndef OPENPOSE_FACE_FACE_DETECTOR_HPP
#define OPENPOSE_FACE_FACE_DETECTOR_HPP

#include <openpose/core/common.hpp>
#include <openpose/pose/enumClasses.hpp>

namespace op
{
    class OP_API FaceDetector
    {
    public:
        explicit FaceDetector(const PoseModel poseModel);

        std::vector<Rectangle<float>> detectFaces(const Array<float>& poseKeypoints) const;

    private:
        const unsigned int mNeck;
        const unsigned int mNose;
        const unsigned int mLEar;
        const unsigned int mREar;
        const unsigned int mLEye;
        const unsigned int mREye;

        DELETE_COPY(FaceDetector);
    };
}

#endif // OPENPOSE_FACE_FACE_DETECTOR_HPP
