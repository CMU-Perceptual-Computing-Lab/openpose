#ifndef OPENPOSE__CORE__SCALE_KEY_POINTS_HPP
#define OPENPOSE__CORE__SCALE_KEY_POINTS_HPP

#include "array.hpp"

namespace op
{
    void scaleKeyPoints(Array<float>& keyPoints, const double scale);

    void scaleKeyPoints(Array<float>& keyPoints, const double scaleX, const double scaleY);

    void scaleKeyPoints(Array<float>& keyPoints, const double scaleX, const double scaleY, const double offsetX, const double offsetY);
}

#endif // OPENPOSE__CORE__SCALE_KEY_POINTS_HPP
