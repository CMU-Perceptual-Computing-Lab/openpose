#ifndef OPENPOSE_CORE_ENUM_CLASSES_HPP
#define OPENPOSE_CORE_ENUM_CLASSES_HPP

namespace op
{
    enum class ScaleMode : unsigned char
    {
        InputResolution,
        NetOutputResolution,
        OutputResolution,
        ZeroToOne, // [0, 1]
        PlusMinusOne, // [-1, 1]
        UnsignedChar, // [0, 255]
        NoScale,
    };

    enum class HeatMapType : unsigned char
    {
        Parts,
        Background,
        PAFs,
    };

    enum class RenderMode : unsigned char
    {
        None,
        Auto, // It will select Gpu if CUDA verison, or Cpu otherwise
        Cpu,
        Gpu,
    };

    enum class ElementToRender : unsigned char
    {
        Skeleton,
        Background,
        AddKeypoints,
        AddPAFs,
    };
}

#endif // OPENPOSE_CORE_ENUM_CLASSES_HPP
