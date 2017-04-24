#ifndef OPENPOSE__CORE__RESIZE_AND_MERGE_BASE_HPP
#define OPENPOSE__CORE__RESIZE_AND_MERGE_BASE_HPP

#include <array>

namespace op
{
    template <typename T>
    void resizeAndMergeCpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize, const T scaleGap = 0.f);

    template <typename T>
    void resizeAndMergeGpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize, const T scaleGap = 0.f);
}

#endif // OPENPOSE__CORE__RESIZE_AND_MERGE_BASE_HPP
