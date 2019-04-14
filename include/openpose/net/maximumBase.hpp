#ifndef OPENPOSE_NET_MAXIMUM_BASE_HPP
#define OPENPOSE_NET_MAXIMUM_BASE_HPP

#include <openpose/core/common.hpp>

namespace op
{
    template <typename T>
    void maximumCpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize,
                    const std::array<int, 4>& sourceSize);

    // Windows: Cuda functions do not include OP_API
    template <typename T>
    void maximumGpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize,
                    const std::array<int, 4>& sourceSize);
}

#endif // OPENPOSE_NET_MAXIMUM_BASE_HPP
