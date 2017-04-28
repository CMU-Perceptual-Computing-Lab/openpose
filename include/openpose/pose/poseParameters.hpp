#ifndef OPENPOSE__POSE__POSE_PARAMETERS_HPP
#define OPENPOSE__POSE__POSE_PARAMETERS_HPP

#include <array>
#include <map>
#include <vector>
#include "enumClasses.hpp"

namespace op
{
    // Model-Dependent Parameters
    // #define when needed in CUDA code
    const std::map<unsigned char, std::string> POSE_COCO_BODY_PARTS {
        {0,  "Nose"},
        {1,  "Neck"},
        {2,  "RShoulder"},
        {3,  "RElbow"},
        {4,  "RWrist"},
        {5,  "LShoulder"},
        {6,  "LElbow"},
        {7,  "LWrist"},
        {8,  "RHip"},
        {9,  "RKnee"},
        {10, "RAnkle"},
        {11, "LHip"},
        {12, "LKnee"},
        {13, "LAnkle"},
        {14, "REye"},
        {15, "LEye"},
        {16, "REar"},
        {17, "LEar"},
        {18, "Background"}
    };
    const unsigned char POSE_COCO_NUMBER_PARTS          = 18; // Equivalent to size of std::map POSE_COCO_BODY_PARTS - 1 (removing background)
    const std::vector<unsigned char> POSE_COCO_MAP_IDX  {31,32, 39,40, 33,34, 35,36, 41,42, 43,44, 19,20, 21,22, 23,24, 25,26, 27,28, 29,30, 47,48, 49,50, 53,54, 51,52, 55,56, 37,38, 45,46};
    const std::vector<unsigned char> POSE_COCO_PAIRS    {1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10, 1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17,   2,16,  5,17};
    #define POSE_COCO_PAIRS_TO_RENDER                   {1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10, 1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17}

    const std::map<unsigned char, std::string> POSE_MPI_BODY_PARTS{
        {0,  "Head"},
        {1,  "Neck"},
        {2,  "RShoulder"},
        {3,  "RElbow"},
        {4,  "RWrist"},
        {5,  "LShoulder"},
        {6,  "LElbow"},
        {7,  "LWrist"},
        {8,  "RHip"},
        {9,  "RKnee"},
        {10, "RAnkle"},
        {11, "LHip"},
        {12, "LKnee"},
        {13, "LAnkle"},
        {14, "Chest"},
        {15, "Background"}
    };
    const unsigned char POSE_MPI_NUMBER_PARTS           = 15; // Equivalent to size of std::map POSE_MPI_NUMBER_PARTS - 1 (removing background)
    const std::vector<unsigned char> POSE_MPI_MAP_IDX   {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 38, 39, 40, 41, 42, 43, 32, 33, 34, 35, 36, 37};
    const std::vector<unsigned char> POSE_MPI_PAIRS     {0,1, 1,2, 2,3, 3,4, 1,5, 5,6, 6,7, 1,14, 14,11, 11,12, 12,13, 14,8, 8,9, 9,10};
    #define POSE_MPI_PAIRS_TO_RENDER                    {0,1,      2,3, 3,4,      5,6, 6,7,              11,12, 12,13,       8,9, 9,10}

    // Constant Global Parameters
    const unsigned char POSE_MAX_PEOPLE = 96;

    // Constant Array Parameters
    const std::array<float, (int)PoseModel::Size>               POSE_CCN_DECREASE_FACTOR{   8.f,                    8.f,                    8.f};
    const std::array<unsigned int, (int)PoseModel::Size>        POSE_MAX_PEAKS{             POSE_MAX_PEOPLE,        POSE_MAX_PEOPLE,        POSE_MAX_PEOPLE};
    const std::array<unsigned char, (int)PoseModel::Size>       POSE_NUMBER_BODY_PARTS{     POSE_COCO_NUMBER_PARTS, POSE_MPI_NUMBER_PARTS,  POSE_MPI_NUMBER_PARTS};
    const std::array<std::map<unsigned char, std::string>, 3>   POSE_BODY_PART_MAPPING{     POSE_COCO_BODY_PARTS,   POSE_MPI_BODY_PARTS,    POSE_MPI_BODY_PARTS};
    const std::array<std::vector<unsigned char>, 3>             POSE_BODY_PART_PAIRS{       POSE_COCO_PAIRS,        POSE_MPI_PAIRS,         POSE_MPI_PAIRS};
    const std::array<std::vector<unsigned char>, 3>             POSE_MAP_IDX{               POSE_COCO_MAP_IDX,      POSE_MPI_MAP_IDX,       POSE_MPI_MAP_IDX};
    const std::array<std::string, (int)PoseModel::Size> POSE_PROTOTXT{  "pose/coco/pose_deploy_linevec.prototxt",
                                                                        "pose/mpi/pose_deploy_linevec.prototxt",
                                                                        "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"};
    const std::array<std::string, (int)PoseModel::Size> POSE_TRAINED_MODEL{ "pose/coco/pose_iter_440000.caffemodel",
                                                                            "pose/mpi/pose_iter_160000.caffemodel",
                                                                            "pose/mpi/pose_iter_160000.caffemodel"};

    // Default Model Parameters
    // They might be modified on running time
    const std::array<float, (int)PoseModel::Size>           POSE_DEFAULT_NMS_THRESHOLD{                     0.05f,      0.6f,       0.3f};
    const std::array<unsigned char, (int)PoseModel::Size>   POSE_DEFAULT_CONNECT_INTER_MIN_ABOVE_THRESHOLD{ 9,          8,          8};
    const std::array<float, (int)PoseModel::Size>           POSE_DEFAULT_CONNECT_INTER_THRESHOLD{           0.05f,      0.01f,      0.01f};
    const std::array<unsigned char, (int)PoseModel::Size>   POSE_DEFAULT_CONNECT_MIN_SUBSET_CNT{            3,          3,          3};
    const std::array<float, (int)PoseModel::Size>           POSE_DEFAULT_CONNECT_MIN_SUBSET_SCORE{          0.4f,       0.4f,       0.4f};

    // Rendering default parameters
    const auto POSE_DEFAULT_ALPHA_POSE = 0.6f;
    const auto POSE_DEFAULT_ALPHA_HEATMAP = 0.7f;

    // Auxiliary functions
    unsigned char poseBodyPartMapStringToKey(const PoseModel poseModel, const std::string& string);
    unsigned char poseBodyPartMapStringToKey(const PoseModel poseModel, const std::vector<std::string>& strings);
}

#endif // OPENPOSE__POSE__POSE_PARAMETERS_HPP
