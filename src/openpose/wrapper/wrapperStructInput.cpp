#include <openpose/wrapper/wrapperStructInput.hpp>

namespace op
{
    WrapperStructInput::WrapperStructInput(const std::shared_ptr<Producer> producerSharedPtr_,
                                           const unsigned long long frameFirst_, const unsigned long long frameLast_,
                                           const bool realTimeProcessing_, const bool frameFlip_,
                                           const int frameRotate_, const bool framesRepeat_) :
        producerSharedPtr{producerSharedPtr_},
        frameFirst{frameFirst_},
        frameLast{frameLast_},
        realTimeProcessing{realTimeProcessing_},
        frameFlip{frameFlip_},
        frameRotate{frameRotate_},
        framesRepeat{framesRepeat_}
    {
    }
}
