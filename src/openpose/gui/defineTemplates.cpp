#include <openpose/gui/headers.hpp>

namespace op
{
    DEFINE_TEMPLATE_DATUM(WGui);
#ifdef USE_3D_ADAM_MODEL
    DEFINE_TEMPLATE_DATUM(WGuiAdam);
#endif
    DEFINE_TEMPLATE_DATUM(WGui3D);
    DEFINE_TEMPLATE_DATUM(WGuiInfoAdder);
}
