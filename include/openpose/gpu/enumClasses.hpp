#ifndef OPENPOSE_GPU_ENUM_CLASSES_HPP
#define OPENPOSE_GPU_ENUM_CLASSES_HPP

namespace op
{
    enum class GpuMode : unsigned char
    {
        CUDA = 0,
        OPEN_CL = 1,
        CPU_ONLY = 2,
        Size,
    };
}

#endif // OPENPOSE_GPU_ENUM_CLASSES_HPP
