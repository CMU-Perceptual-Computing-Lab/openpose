#ifndef OPENPOSE_NET_MAXIMUM_BASE_HPP
#define OPENPOSE_NET_MAXIMUM_BASE_HPP

#include <openpose/core/common.hpp>

namespace op
{
    template <typename T>
    OP_API void maximumCpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize,
                           const std::array<int, 4>& sourceSize);

    template <typename T>
    OP_API void maximumGpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize,
                           const std::array<int, 4>& sourceSize);
}

#endif // OPENPOSE_NET_MAXIMUM_BASE_HPP
