#include <openpose/wrapper/wrapperStructInput.hpp>

namespace op
{
    WrapperStructInput::WrapperStructInput(
        const ProducerType producerType_, const String& producerString_, const unsigned long long frameFirst_,
        const unsigned long long frameStep_, const unsigned long long frameLast_, const bool realTimeProcessing_,
        const bool frameFlip_, const int frameRotate_, const bool framesRepeat_, const Point<int>& cameraResolution_,
        const String& cameraParameterPath_, const bool undistortImage_, const int numberViews_) :
        producerType{producerType_},
        producerString{producerString_},
        frameFirst{frameFirst_},
        frameStep{frameStep_},
        frameLast{frameLast_},
        realTimeProcessing{realTimeProcessing_},
        frameFlip{frameFlip_},
        frameRotate{frameRotate_},
        framesRepeat{framesRepeat_},
        cameraResolution{cameraResolution_},
        cameraParameterPath{cameraParameterPath_},
        undistortImage{undistortImage_},
        numberViews{numberViews_}
    {
    }
}
