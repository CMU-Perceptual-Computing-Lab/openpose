#ifndef OPENPOSE_CORE_NMS_BASE_HPP
#define OPENPOSE_CORE_NMS_BASE_HPP

#include <array>

namespace op
{
    template <typename T>
    void nmsCpu(T* targetPtr, int* kernelPtr, const T* const sourcePtr, const T threshold, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);

    template <typename T>
    void nmsGpu(T* targetPtr, int* kernelPtr, const T* const sourcePtr, const T threshold, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);
}

#endif // OPENPOSE_CORE_NMS_BASE_HPP
