#include <openpose/wrapper/wrapperStructExtra.hpp>

namespace op
{
    WrapperStructExtra::WrapperStructExtra(
        const bool reconstruct3d_, const int minViews3d_, const bool identification_, const int tracking_,
        const int ikThreads_) :
        reconstruct3d{reconstruct3d_},
        minViews3d{minViews3d_},
        identification{identification_},
        tracking{tracking_},
        ikThreads{ikThreads_}
    {
    }
}
