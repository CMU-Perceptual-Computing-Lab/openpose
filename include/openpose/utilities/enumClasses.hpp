#ifndef OPENPOSE_UTILITIES_ENUM_CLASSES_HPP
#define OPENPOSE_UTILITIES_ENUM_CLASSES_HPP

namespace op
{
    enum class ErrorMode : unsigned char
    {
        StdRuntimeError,
        FileLogging,
        StdCerr,
        All,
    };

    enum class LogMode : unsigned char
    {
        FileLogging,
        StdCout,
        All,
    };

    enum class Priority : unsigned char
    {
        None = 0,
        Low = 1,
        Normal = 2,
        High = 3,
        Max = 4,
        NoOutput = 255,
    };

    enum class Extensions : unsigned char
    {
        Images, // jpg, png, ...
        Size
    };
}

#endif // OPENPOSE_UTILITIES_ENUM_CLASSES_HPP
