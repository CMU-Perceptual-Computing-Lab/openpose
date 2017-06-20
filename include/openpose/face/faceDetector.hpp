#ifndef OPENPOSE_FACE_FACE_DETECTOR_HPP
#define OPENPOSE_FACE_FACE_DETECTOR_HPP

#include <vector>
#include <openpose/core/array.hpp>
#include <openpose/core/rectangle.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/utilities/macros.hpp>
#include "enumClasses.hpp"

namespace op
{
    class FaceDetector
    {
    public:
        explicit FaceDetector(const PoseModel poseModel);

        std::vector<Rectangle<float>> detectFaces(const Array<float>& poseKeypoints, const float scaleInputToOutput) const;

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
