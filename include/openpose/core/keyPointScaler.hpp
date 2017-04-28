#ifndef OPENPOSE__CORE__KEY_POINT_SCALER_HPP
#define OPENPOSE__CORE__KEY_POINT_SCALER_HPP

#include <vector>
#include "array.hpp"
#include "enumClasses.hpp"

namespace op
{
    class KeyPointScaler
    {
    public:
        explicit KeyPointScaler(const ScaleMode scaleMode);

        void scale(Array<float>& arrayToScale, const double scaleInputToOutput, const double scaleNetToOutput, const cv::Size& producerSize) const;

        void scale(std::vector<Array<float>>& arraysToScale, const double scaleInputToOutput, const double scaleNetToOutput, const cv::Size& producerSize) const;

    private:
        const ScaleMode mScaleMode;
    };
}

#endif // OPENPOSE__CORE__KEY_POINT_SCALER_HPP
