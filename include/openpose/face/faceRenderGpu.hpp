#ifndef OPENPOSE_FACE_GPU_FACE_RENDER_HPP
#define OPENPOSE_FACE_GPU_FACE_RENDER_HPP

#include <openpose/core/point.hpp>
#include "faceParameters.hpp"

namespace op
{
	void renderFaceGpu(float* framePtr, const Point<int>& frameSize, const float* const facePtr, const int numberFace, const float alphaColorToAdd = FACE_DEFAULT_ALPHA_KEYPOINT);
}

#endif // OPENPOSE_FACE_GPU_FACE_RENDER_HPP
