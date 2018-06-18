#include <openpose/3d/headers.hpp>

namespace op
{
#ifdef USE_3D_ADAM_MODEL
    DEFINE_TEMPLATE_DATUM(WJointAngleEstimation);
#endif
    DEFINE_TEMPLATE_DATUM(WPoseTriangulation);
}
