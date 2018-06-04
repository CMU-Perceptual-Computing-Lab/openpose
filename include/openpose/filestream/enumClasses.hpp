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
    enum class CocoJsonFormat : bool
    {
        Body,
        Foot,
    };
}

#endif // OPENPOSE_FILESTREAM_ENUM_CLASSES_HPP
