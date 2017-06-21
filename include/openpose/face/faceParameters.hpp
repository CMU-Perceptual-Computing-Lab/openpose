#ifndef OPENPOSE_FACE_FACE_PARAMETERS_HPP
#define OPENPOSE_FACE_FACE_PARAMETERS_HPP

#include "enumClasses.hpp"
#include <openpose/pose/poseParameters.hpp>

namespace op
{
    const auto FACE_MAX_FACES = POSE_MAX_PEOPLE;

    const auto FACE_NUMBER_PARTS = 70u;
    #define FACE_PAIRS_RENDER_GPU {0,1,  1,2,  2,3,  3,4,  4,5,  5,6,  6,7,  7,8,  8,9,  9,10,  10,11,  11,12,  12,13,  13,14,  14,15,  15,16,  17,18,  18,19,  19,20, \
                                  20,21,  22,23,  23,24,  24,25,  25,26,  27,28,  28,29,  29,30,  31,32,  32,33,  33,34,  34,35,  36,37,  37,38,  38,39,  39,40,  40,41, \
                                  41,36,  42,43,  43,44,  44,45,  45,46,  46,47,  47,42,  48,49,  49,50,  50,51,  51,52,  52,53,  53,54,  54,55,  55,56,  56,57,  57,58, \
                                  58,59,  59,48,  60,61,  61,62,  62,63,  63,64,  64,65,  65,66,  66,67,  67,60}
    const std::vector<unsigned int> FACE_PAIRS_RENDER {FACE_PAIRS_RENDER_GPU};
    #define FACE_COLORS_RENDER      255.f,    255.f,    255.f

    // Constant parameters
    const auto FACE_CCN_DECREASE_FACTOR = 8.f;
    const auto FACE_MAX_PEAKS = 64u;
    const std::string FACE_PROTOTXT{"face/pose_deploy.prototxt"};
    const std::string FACE_TRAINED_MODEL{"face/pose_iter_116000.caffemodel"};

    // Default Model Parameters
    // They might be modified on running time
    const auto FACE_DEFAULT_NMS_THRESHOLD = 0.1f;

    // Rendering parameters
    const auto FACE_DEFAULT_ALPHA_KEYPOINT = POSE_DEFAULT_ALPHA_KEYPOINT;
    const auto FACE_DEFAULT_ALPHA_HEAT_MAP = POSE_DEFAULT_ALPHA_HEAT_MAP;
    const auto FACE_RENDER_THRESHOLD = 0.4f;
}

#endif // OPENPOSE_FACE_FACE_PARAMETERS_HPP
