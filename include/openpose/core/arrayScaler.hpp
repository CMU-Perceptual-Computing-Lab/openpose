#ifndef OPENPOSE__CORE__ARRAYS_SCALER_HPP
#define OPENPOSE__CORE__ARRAYS_SCALER_HPP

#include <vector>
#include "array.hpp"
#include "enumClasses.hpp"

namespace op
{
    class ArrayScaler
    {
    public:
        explicit ArrayScaler(const ScaleMode scalePose);

        void scale(Array<float>& array, const double scaleInputToOutput, const double scaleNetToOutput, const cv::Size& producerSize) const;

        void scale(std::vector<Array<float>>& arrays, const double scaleInputToOutput, const double scaleNetToOutput, const cv::Size& producerSize) const;

    private:
        const ScaleMode mScaleMode;
    };
}

#endif // OPENPOSE__CORE__ARRAYS_SCALER_HPP
