#ifndef OPENPOSE_FACE_RENDER_FACE_HPP
#define OPENPOSE_FACE_RENDER_FACE_HPP

#include <openpose/core/common.hpp>
#include <openpose/face/faceParameters.hpp>

namespace op
{
    OP_API void renderFaceKeypointsCpu(Array<float>& frameArray, const Array<float>& faceKeypoints, const float renderThreshold);

    OP_API void renderFaceKeypointsGpu(float* framePtr, const Point<int>& frameSize, const float* const facePtr, const int numberPeople,
                                       const float renderThreshold, const float alphaColorToAdd = FACE_DEFAULT_ALPHA_KEYPOINT);
}

#endif // OPENPOSE_FACE_RENDER_FACE_HPP
