#ifndef OPENPOSE_FACE_RENDER_FACE_HPP
#define OPENPOSE_FACE_RENDER_FACE_HPP

#include <openpose/core/array.hpp>
#include <openpose/core/point.hpp>
#include "faceParameters.hpp"

namespace op
{
    void renderFaceKeypointsCpu(Array<float>& frameArray, const Array<float>& faceKeypoints);

    void renderFaceKeypointsGpu(float* framePtr, const Point<int>& frameSize, const float* const facePtr, const int numberFace,
                                const float alphaColorToAdd = FACE_DEFAULT_ALPHA_KEYPOINT);
}

#endif // OPENPOSE_FACE_RENDER_FACE_HPP
