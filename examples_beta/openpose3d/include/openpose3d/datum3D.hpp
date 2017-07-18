#ifndef OPENPOSE3D_DATUM_3D_HPP
#define OPENPOSE3D_DATUM_3D_HPP

#include <openpose/core/array.hpp>
#include <openpose/core/datum.hpp>

// Following OpenPose `tutorial_wrapper/` examples, we create our own class inherited from Datum
// See `examples/tutorial_wrapper/` for more details
struct Datum3D : public op::Datum
{
    op::Array<float> poseKeypoints3D;
    op::Array<float> faceKeypoints3D;
    op::Array<float> leftHandKeypoints3D;
    op::Array<float> rightHandKeypoints3D;
};

#endif // OPENPOSE3D_DATUM_3D_HPP
