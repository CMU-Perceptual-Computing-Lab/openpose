#ifndef OPENPOSE_CORE_KEYPOINT_SCALER_HPP
#define OPENPOSE_CORE_KEYPOINT_SCALER_HPP

#include <vector>
#include "array.hpp"
#include "point.hpp"
#include "enumClasses.hpp"

namespace op
{
    class KeypointScaler
    {
    public:
        explicit KeypointScaler(const ScaleMode scaleMode);

        void scale(Array<float>& arrayToScale, const float scaleInputToOutput, const float scaleNetToOutput, const Point<int>& producerSize) const;

        void scale(std::vector<Array<float>>& arraysToScale, const float scaleInputToOutput, const float scaleNetToOutput, const Point<int>& producerSize) const;

    private:
        const ScaleMode mScaleMode;
    };
}

#endif // OPENPOSE_CORE_KEYPOINT_SCALER_HPP
