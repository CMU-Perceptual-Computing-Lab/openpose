#ifndef OPENPOSE_HAND_HAND_PARAMETERS_HPP
#define OPENPOSE_HAND_HAND_PARAMETERS_HPP

#include "enumClasses.hpp"
#include <openpose/pose/poseParameters.hpp>

namespace op
{
    const auto HAND_MAX_HANDS = 2*POSE_MAX_PEOPLE;

    const auto HAND_NUMBER_PARTS = 21u;
    #define HAND_PAIRS_TO_RENDER {0,1,  1,2,  2,3,  3,4,  0,5,  5,6,  6,7,  7,8,  0,9,  9,10,  10,11,  11,12,  0,13,  13,14,  14,15,  15,16,  0,17,  17,18,  18,19,  19,20}

    // Constant parameters
    const auto HAND_CCN_DECREASE_FACTOR = 8.f;
    const auto HAND_MAX_PEAKS = 64u;
    const std::string HAND_PROTOTXT{"hand/pose_deploy.prototxt"};
    const std::string HAND_TRAINED_MODEL{"hand/pose_iter_120000.caffemodel"};

    // Default Model Parameters
    // They might be modified on running time
    const auto HAND_DEFAULT_NMS_THRESHOLD = 0.1f;

    // Rendering default parameters
    const auto HAND_DEFAULT_ALPHA_KEYPOINT = POSE_DEFAULT_ALPHA_KEYPOINT;
    const auto HAND_DEFAULT_ALPHA_HEAT_MAP = POSE_DEFAULT_ALPHA_HEAT_MAP;
}

#endif // OPENPOSE_HAND_HAND_PARAMETERS_HPP
