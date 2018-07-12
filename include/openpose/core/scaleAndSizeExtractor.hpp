#ifndef OPENPOSE_CORE_SCALE_AND_SIZE_EXTRACTOR_HPP
#define OPENPOSE_CORE_SCALE_AND_SIZE_EXTRACTOR_HPP

#include <tuple>
#include <openpose/core/common.hpp>

namespace op
{
    class OP_API ScaleAndSizeExtractor
    {
    public:
        ScaleAndSizeExtractor(const Point<int>& netInputResolution, const Point<int>& outputResolution,
                              const int scaleNumber = 1, const double scaleGap = 0.25);

        std::tuple<std::vector<double>, std::vector<Point<int>>, double, Point<int>> extract(
            const Point<int>& inputResolution) const;

    private:
        const Point<int> mNetInputResolution;
        const Point<int> mOutputSize;
        const int mScaleNumber;
        const double mScaleGap;
    };
}

#endif // OPENPOSE_CORE_SCALE_AND_SIZE_EXTRACTOR_HPP
