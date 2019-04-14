#ifndef OPENPOSE_WRAPPER_ENUM_CLASSES_HPP
#define OPENPOSE_WRAPPER_ENUM_CLASSES_HPP

namespace op
{
    enum class PoseMode : unsigned char
    {
        Disabled = 0,
        Enabled,
        NoNetwork,
        Size,
    };

    enum class Detector : unsigned char
    {
        Body = 0,
        OpenCV,
        Provided,
        BodyWithTracking,
        Size,
    };

    enum class WorkerType : unsigned char
    {
        Input = 0,
        PreProcessing,
        PostProcessing,
        Output,
        Size,
    };
}

#endif // OPENPOSE_WRAPPER_ENUM_CLASSES_HPP
