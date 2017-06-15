#ifndef OPENPOSE_CORE_RESIZE_AND_MERGE_BASE_HPP
#define OPENPOSE_CORE_RESIZE_AND_MERGE_BASE_HPP

#include <array>
#include <vector>

namespace op
{
    template <typename T>
    void resizeAndMergeCpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize,
                           const std::vector<T>& scaleRatios = {1});

    template <typename T>
    void resizeAndMergeGpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize,
                           const std::vector<T>& scaleRatios = {1});
}

#endif // OPENPOSE_CORE_RESIZE_AND_MERGE_BASE_HPP
