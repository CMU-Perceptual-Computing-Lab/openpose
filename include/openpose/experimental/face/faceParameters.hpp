#ifndef OPENPOSE__FACE__FACE_PARAMETERS_HPP
#define OPENPOSE__FACE__FACE_PARAMETERS_HPP

#include "enumClasses.hpp"
#include "../../pose/poseParameters.hpp"

namespace op
{
    const unsigned char FACE_MAX_NUMBER_FACE = 1;

    const unsigned char FACE_NUMBER_PARTS = 71;
    #define FACE_PAIRS_TO_RENDER {0,1,  1,2,  2,3,  3,4,  4,5,  5,6,  6,7,  7,8,  8,9,  9,10,  10,11,  11,12,  12,13,  13,14,  14,15,  15,16,  17,18,  18,19,  19,20, \
                                  20,21,  22,23,  23,24,  24,25,  25,26,  27,28,  28,29,  29,30,  31,32,  32,33,  33,34,  34,35,  36,37,  37,38,  38,39,  39,40,  40,41, \
                                  41,36,  42,43,  43,44,  44,45,  45,46,  46,47,  47,42,  48,49,  49,50,  50,51,  51,52,  52,53,  53,54,  54,55,  55,56,  56,57,  57,58, \
                                  58,59,  59,48,  60,61,  61,62,  62,63,  63,64,  64,65,  65,66,  66,67,  67,60}

    // Constant Global Parameters
    // const unsigned char FACE_MAX_PEOPLE = 1;

    // Constant parameters
    const auto FACE_CCN_DECREASE_FACTOR = 8.f;
    const unsigned int FACE_MAX_PEAKS = 64u;
    const std::string FACE_PROTOTXT{"face/pose_deploy.prototxt"};
    const std::string FACE_TRAINED_MODEL{"face/pose_iter_116000.caffemodel"};

    // Default Model Parameters
    // They might be modified on running time
    const auto FACE_DEFAULT_NMS_THRESHOLD = 0.1f;

    // Rendering default parameters
    const auto FACE_DEFAULT_ALPHA_FACE = POSE_DEFAULT_ALPHA_POSE;
    // const auto FACE_DEFAULT_ALPHA_HEATMAP = POSE_DEFAULT_ALPHA_HEATMAP;
}

#endif // OPENPOSE__FACE__FACE_PARAMETERS_HPP
