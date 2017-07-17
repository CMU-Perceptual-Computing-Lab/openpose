#ifndef OPENPOSE_CORE_NET_HPP
#define OPENPOSE_CORE_NET_HPP

#include <openpose/core/common.hpp>

namespace op
{
    class OP_API Net
    {
    public:
        virtual void initializationOnThread() = 0;

        // Alternative a) getInputDataCpuPtr or getInputDataGpuPtr + forwardPass()
        virtual float* getInputDataCpuPtr() const = 0;

        virtual float* getInputDataGpuPtr() const = 0;

        // Alternative b)
        virtual void forwardPass(const float* const inputData = nullptr) const = 0;
    };
}

#endif // OPENPOSE_CORE_NET_HPP
