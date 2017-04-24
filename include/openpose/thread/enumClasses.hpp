#ifndef OPENPOSE__THREAD__ENUM_CLASSES_HPP
#define OPENPOSE__THREAD__ENUM_CLASSES_HPP

namespace op
{
    enum class ThreadMode : unsigned char
    {
        Asynchronous,
        AsynchronousIn,
        AsynchronousOut,
        Synchronous,
    };
}

#endif // OPENPOSE__THREAD__ENUM_CLASSES_HPP
