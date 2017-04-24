#ifndef OPENPOSE__HANDS__HANDS_PARAMETERS_HPP
#define OPENPOSE__HANDS__HANDS_PARAMETERS_HPP

#include "enumClasses.hpp"
#include "../../pose/poseParameters.hpp"

namespace op
{
    const unsigned char HANDS_MAX_NUMBER_HANDS = 2;

    const unsigned char HANDS_NUMBER_PARTS = 21;
    #define HANDS_PAIRS_TO_RENDER {0,1,  1,2,  2,3,  3,4,  0,5,  5,6,  6,7,  7,8,  0,9,  9,10,  10,11,  11,12,  0,13,  13,14,  14,15,  15,16,  0,17,  17,18,  18,19,  19,20}

    // Constant Global Parameters
    // const unsigned char HANDS_MAX_PEOPLE = 1;

    // Constant parameters
    const auto HANDS_CCN_DECREASE_FACTOR = 8.f;
    const unsigned int HANDS_MAX_PEAKS = 64u;
    const std::string HANDS_PROTOTXT{"hand/pose_deploy.prototxt"};
    const std::string HANDS_TRAINED_MODEL{"hand/pose_iter_120000.caffemodel"};

    // Default Model Parameters
    // They might be modified on running time
    const auto HANDS_DEFAULT_NMS_THRESHOLD = 0.1f;

    // Rendering default parameters
    const auto HANDS_DEFAULT_ALPHA_HANDS = POSE_DEFAULT_ALPHA_POSE;
    // const auto HANDS_DEFAULT_ALPHA_HEATMAP = POSE_DEFAULT_ALPHA_HEATMAP;
}

#endif // OPENPOSE__HANDS__HANDS_PARAMETERS_HPP
