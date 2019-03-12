#ifndef OPENPOSE_FILESTREAM_ENUM_CLASSES_HPP
#define OPENPOSE_FILESTREAM_ENUM_CLASSES_HPP

namespace op
{
    enum class DataFormat : unsigned char
    {
        Json,
        Xml,
        Yaml,
        Yml,
    };

    enum class CocoJsonFormat : unsigned char
    {
        Body,
        Hand21,
        Hand42,
        Face,
        Foot,
        Car,
        Size,
    };
}

#endif // OPENPOSE_FILESTREAM_ENUM_CLASSES_HPP
