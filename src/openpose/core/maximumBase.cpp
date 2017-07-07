#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/macros.hpp>
#include <openpose/core/maximumBase.hpp>

namespace op
{
    template <typename T>
    void maximumCpu(T* targetPtr, int* kernelPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize)
    {
        try
        {
            UNUSED(targetPtr);
            UNUSED(kernelPtr);
            UNUSED(sourcePtr);
            UNUSED(targetSize);
            UNUSED(sourceSize);
            error("CPU version not completely implemented.", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void maximumCpu(float* targetPtr, int* kernelPtr, const float* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);
    template void maximumCpu(double* targetPtr, int* kernelPtr, const double* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);
}
