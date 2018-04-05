#ifndef OPENPOSE_CORE_KEYPOINT_SCALER_HPP
#define OPENPOSE_CORE_KEYPOINT_SCALER_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/enumClasses.hpp>

namespace op
{
    class OP_API KeypointScaler
    {
    public:
        explicit KeypointScaler(const ScaleMode scaleMode);

        void scale(Array<float>& arrayToScale, const double scaleInputToOutput, const double scaleNetToOutput,
                   const Point<int>& producerSize) const;

        void scale(std::vector<Array<float>>& arraysToScale, const double scaleInputToOutput,
                   const double scaleNetToOutput, const Point<int>& producerSize) const;

        void scale(std::vector<std::vector<std::array<float,3>>>& poseCandidates, const double scaleInputToOutput,
                   const double scaleNetToOutput, const Point<int>& producerSize) const;

    private:
        const ScaleMode mScaleMode;
    };
}

#endif // OPENPOSE_CORE_KEYPOINT_SCALER_HPP
