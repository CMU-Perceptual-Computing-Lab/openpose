#ifndef OPENPOSE_CORE_SCALE_KEYPOINTS_HPP
#define OPENPOSE_CORE_SCALE_KEYPOINTS_HPP

#include "array.hpp"

namespace op
{
    void scaleKeypoints(Array<float>& keypoints, const float scale);

    void scaleKeypoints(Array<float>& keypoints, const float scaleX, const float scaleY);

    void scaleKeypoints(Array<float>& keypoints, const float scaleX, const float scaleY, const float offsetX, const float offsetY);
}

#endif // OPENPOSE_CORE_SCALE_KEYPOINTS_HPP
