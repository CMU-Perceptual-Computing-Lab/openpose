#include <openpose/filestream/headers.hpp>

namespace op
{
#ifdef USE_3D_ADAM_MODEL
    DEFINE_TEMPLATE_DATUM(WBvhSaver);
#endif
    DEFINE_TEMPLATE_DATUM(WCocoJsonSaver);
    DEFINE_TEMPLATE_DATUM(WFaceSaver);
    DEFINE_TEMPLATE_DATUM(WHandSaver);
    DEFINE_TEMPLATE_DATUM(WHeatMapSaver);
    DEFINE_TEMPLATE_DATUM(WImageSaver);
    DEFINE_TEMPLATE_DATUM(WPeopleJsonSaver);
    DEFINE_TEMPLATE_DATUM(WPoseSaver);
    DEFINE_TEMPLATE_DATUM(WUdpSender);
    DEFINE_TEMPLATE_DATUM(WVideoSaver);
    DEFINE_TEMPLATE_DATUM(WVideoSaver3D);
}
