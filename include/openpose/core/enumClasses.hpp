#ifndef OPENPOSE__CORE__ENUM_CLASSES_HPP
#define OPENPOSE__CORE__ENUM_CLASSES_HPP

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
    };

    enum class HeatMapType : unsigned char
    {
        Parts,
        Background,
        PAFs,
    };
}

#endif // OPENPOSE__CORE__ENUM_CLASSES_HPP
