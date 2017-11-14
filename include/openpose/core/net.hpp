#ifndef OPENPOSE_CORE_NET_HPP
#define OPENPOSE_CORE_NET_HPP

#include <openpose/core/common.hpp>

namespace op
{
    class OP_API Net
    {
    public:
        virtual void initializationOnThread() = 0;

        virtual void forwardPass(const Array<float>& inputData) const = 0;
    };
}

#endif // OPENPOSE_CORE_NET_HPP
