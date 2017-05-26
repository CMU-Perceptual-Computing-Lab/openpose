#ifndef OPENPOSE__CORE__SCALE_KEY_POINTS_HPP
#define OPENPOSE__CORE__SCALE_KEY_POINTS_HPP

#include "array.hpp"

namespace op
{
    void scaleKeyPoints(Array<float>& keyPoints, const float scale);

    void scaleKeyPoints(Array<float>& keyPoints, const float scaleX, const float scaleY);

    void scaleKeyPoints(Array<float>& keyPoints, const float scaleX, const float scaleY, const float offsetX, const float offsetY);
}

#endif // OPENPOSE__CORE__SCALE_KEY_POINTS_HPP
