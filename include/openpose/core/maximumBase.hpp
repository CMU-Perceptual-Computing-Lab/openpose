#ifndef OPENPOSE_CORE_MAXIMUM_BASE_HPP
#define OPENPOSE_CORE_MAXIMUM_BASE_HPP

#include <array>

namespace op
{
    template <typename T>
    void maximumCpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);

    template <typename T>
    void maximumGpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);
}

#endif // OPENPOSE_CORE_MAXIMUM_BASE_HPP
