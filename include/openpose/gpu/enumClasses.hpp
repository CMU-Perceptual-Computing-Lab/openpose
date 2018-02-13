#ifndef OPENPOSE_GPU_ENUM_CLASSES_HPP
#define OPENPOSE_GPU_ENUM_CLASSES_HPP

namespace op
{
    enum class GpuMode : unsigned char
    {
        Cuda = 0,
        OpenCL = 1,
        NoGpu = 2,
        Size,
    };
}

#endif // OPENPOSE_GPU_ENUM_CLASSES_HPP
