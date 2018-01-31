#ifndef OPENPOSE_GPU_GPU_HPP
#define OPENPOSE_GPU_GPU_HPP

#include <openpose/core/common.hpp>
#include <openpose/gpu/enumClasses.hpp>

namespace op
{
    OP_API int getGpuNumber();

    OP_API GpuMode getGpuMode();
}

#endif // OPENPOSE_GPU_GPU_HPP
