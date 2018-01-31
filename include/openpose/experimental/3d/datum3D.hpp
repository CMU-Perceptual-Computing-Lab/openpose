#ifndef OPENPOSE_EXPERIMENTAL_3D_DATUM_3D_HPP
#define OPENPOSE_EXPERIMENTAL_3D_DATUM_3D_HPP

#include <openpose/core/array.hpp>
#include <openpose/core/datum.hpp>

namespace op
{
    // Following OpenPose `tutorial_wrapper/` examples, we create our own class inherited from Datum
    // See `examples/tutorial_wrapper/` for more details
    struct OP_API Datum3D : public Datum
    {
        Array<float> poseKeypoints3D;
        Array<float> faceKeypoints3D;
        Array<float> leftHandKeypoints3D;
        Array<float> rightHandKeypoints3D;
    };
}

#endif // OPENPOSE_EXPERIMENTAL_3D_DATUM_3D_HPP
