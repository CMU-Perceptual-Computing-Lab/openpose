#ifndef OPENPOSE__CORE__SCALE_POSE_HPP
#define OPENPOSE__CORE__SCALE_POSE_HPP

#include "array.hpp"

namespace op
{
    void scalePose(Array<float>& pose, const double scale);

    void scalePose(Array<float>& pose, const double scaleX, const double scaleY);

    void scalePose(Array<float>& pose, const double scaleX, const double scaleY, const double offsetX, const double offsetY);
}

#endif // OPENPOSE__CORE__SCALE_POSE_HPP
