#ifndef OPENPOSE_POSE_POSE_PARAMETERS_HPP
#define OPENPOSE_POSE_POSE_PARAMETERS_HPP

#include <map>
#include <openpose/core/common.hpp>
#include <openpose/pose/enumClasses.hpp>

namespace op
{
    // All #define are for CUDA code

    // Constant Global Parameters
    const unsigned int POSE_MAX_PEOPLE = 96u;

    // Model-Dependent Parameters
    // COCO
    const std::map<unsigned int, std::string> POSE_COCO_BODY_PARTS {
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
    const unsigned int POSE_COCO_NUMBER_PARTS               = 18u; // Equivalent to size of std::map POSE_COCO_BODY_PARTS - 1 (removing background)
    const std::vector<unsigned int> POSE_COCO_MAP_IDX       {31,32, 39,40, 33,34, 35,36, 41,42, 43,44, 19,20, 21,22, 23,24, 25,26, 27,28, 29,30, 47,48, 49,50, 53,54, 51,52, 55,56, 37,38, 45,46};
    #define POSE_COCO_PAIRS_RENDER_GPU                      {1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10,  1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17}
    const std::vector<unsigned int> POSE_COCO_PAIRS_RENDER  {POSE_COCO_PAIRS_RENDER_GPU};
    const std::vector<unsigned int> POSE_COCO_PAIRS         {1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10,  1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17,  2,16,  5,17};
    #define POSE_COCO_COLORS_RENDER_GPU \
        255.f,     0.f,    85.f, \
        255.f,     0.f,     0.f, \
        255.f,    85.f,     0.f, \
        255.f,   170.f,     0.f, \
        255.f,   255.f,     0.f, \
        170.f,   255.f,     0.f, \
         85.f,   255.f,     0.f, \
          0.f,   255.f,     0.f, \
          0.f,   255.f,    85.f, \
          0.f,   255.f,   170.f, \
          0.f,   255.f,   255.f, \
          0.f,   170.f,   255.f, \
          0.f,    85.f,   255.f, \
          0.f,     0.f,   255.f, \
        255.f,     0.f,   170.f, \
        170.f,     0.f,   255.f, \
        255.f,     0.f,   255.f, \
         85.f,     0.f,   255.f
    const std::vector<float> POSE_COCO_COLORS_RENDER{POSE_COCO_COLORS_RENDER_GPU};
    // MPI
    const std::map<unsigned int, std::string> POSE_MPI_BODY_PARTS {
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
    const unsigned int POSE_MPI_NUMBER_PARTS            = 15; // Equivalent to size of std::map POSE_MPI_NUMBER_PARTS - 1 (removing background)
    const std::vector<unsigned int> POSE_MPI_MAP_IDX    {16,17, 18,19, 20,21, 22,23, 24,25, 26,27, 28,29, 30,31, 32,33, 34,35, 36,37, 38,39, 40,41, 42,43};
    #define POSE_MPI_PAIRS_RENDER_GPU                   { 0,1,   1,2,   2,3,   3,4,   1,5,   5,6,   6,7,   1,14,  14,8,  8,9,  9,10,  14,11, 11,12, 12,13}
    const std::vector<unsigned int> POSE_MPI_PAIRS      POSE_MPI_PAIRS_RENDER_GPU;
    // MPI colors chosen such that they are closed to COCO colors
    #define POSE_MPI_COLORS_RENDER_GPU \
        255.f,     0.f,    85.f, \
        255.f,     0.f,     0.f, \
        255.f,    85.f,     0.f, \
        255.f,   170.f,     0.f, \
        255.f,   255.f,     0.f, \
        170.f,   255.f,     0.f, \
         85.f,   255.f,     0.f, \
         43.f,   255.f,     0.f, \
          0.f,   255.f,     0.f, \
          0.f,   255.f,    85.f, \
          0.f,   255.f,   170.f, \
          0.f,   255.f,   255.f, \
          0.f,   170.f,   255.f, \
          0.f,    85.f,   255.f, \
          0.f,     0.f,   255.f
    const std::vector<float> POSE_MPI_COLORS_RENDER{POSE_MPI_COLORS_RENDER_GPU};
    // BODY_18
    const std::map<unsigned int, std::string> POSE_BODY_18_BODY_PARTS { // Windows map copy error if `= POSE_COCO_BODY_PARTS`
        { 0,  "Nose" },
        { 1,  "Neck" },
        { 2,  "RShoulder" },
        { 3,  "RElbow" },
        { 4,  "RWrist" },
        { 5,  "LShoulder" },
        { 6,  "LElbow" },
        { 7,  "LWrist" },
        { 8,  "RHip" },
        { 9,  "RKnee" },
        { 10, "RAnkle" },
        { 11, "LHip" },
        { 12, "LKnee" },
        { 13, "LAnkle" },
        { 14, "REye" },
        { 15, "LEye" },
        { 16, "REar" },
        { 17, "LEar" },
        { 18, "Background" }
    };
    const unsigned int POSE_BODY_18_NUMBER_PARTS                        {POSE_COCO_NUMBER_PARTS};
    const std::vector<unsigned int> POSE_BODY_18_MAP_IDX                {POSE_COCO_MAP_IDX};
    #define POSE_BODY_18_PAIRS_RENDER_GPU                               POSE_COCO_PAIRS_RENDER_GPU
    const std::vector<unsigned int> POSE_BODY_18_PAIRS_RENDER           {POSE_BODY_18_PAIRS_RENDER_GPU};
    const std::vector<unsigned int> POSE_BODY_18_PAIRS                  {POSE_COCO_PAIRS};
    #define POSE_BODY_18_COLORS_RENDER_GPU                              POSE_COCO_COLORS_RENDER_GPU
    const std::vector<float> POSE_BODY_18_COLORS_RENDER                 {POSE_BODY_18_COLORS_RENDER_GPU};
    // BODY_19
    const std::map<unsigned int, std::string> POSE_BODY_19_BODY_PARTS {
        { 0,  "Nose" },
        { 1,  "Neck" },
        { 2,  "RShoulder" },
        { 3,  "RElbow" },
        { 4,  "RWrist" },
        { 5,  "LShoulder" },
        { 6,  "LElbow" },
        { 7,  "LWrist" },
        { 8,  "LowerAbs" },
        { 9,  "RHip" },
        { 10, "RKnee" },
        { 11, "RAnkle" },
        { 12, "LHip" },
        { 13, "LKnee" },
        { 14, "LAnkle" },
        { 15, "REye" },
        { 16, "LEye" },
        { 17, "REar" },
        { 18, "LEar" },
        { 19, "Background" }
    };
    const unsigned int POSE_BODY_19_NUMBER_PARTS               = 19u; // Equivalent to size of std::map POSE_BODY_19_BODY_PARTS - 1 (removing background)
    const std::vector<unsigned int> POSE_BODY_19_MAP_IDX       {20,21, 34,35, 42,43, 36,37, 38,39, 44,45, 46,47, 26,27, 22,23, 24,25, 28,29, 30,31, 32,33, 50,51, 52,53, 56,57, 54,55, 58,59, 40,41, 48,49};
    #define POSE_BODY_19_PAIRS_RENDER_GPU                      {1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18}
    const std::vector<unsigned int> POSE_BODY_19_PAIRS_RENDER  {POSE_COCO_PAIRS_RENDER_GPU};
    const std::vector<unsigned int> POSE_BODY_19_PAIRS         {1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18};
    #define POSE_BODY_19_COLORS_RENDER_GPU \
        255.f,     0.f,    85.f, \
        255.f,     0.f,     0.f, \
        255.f,    85.f,     0.f, \
        255.f,   170.f,     0.f, \
        255.f,   255.f,     0.f, \
        170.f,   255.f,     0.f, \
         85.f,   255.f,     0.f, \
          0.f,   255.f,     0.f, \
        255.f,     0.f,     0.f, \
          0.f,   255.f,    85.f, \
          0.f,   255.f,   170.f, \
          0.f,   255.f,   255.f, \
          0.f,   170.f,   255.f, \
          0.f,    85.f,   255.f, \
          0.f,     0.f,   255.f, \
        255.f,     0.f,   170.f, \
        170.f,     0.f,   255.f, \
        255.f,     0.f,   255.f, \
         85.f,     0.f,   255.f
    const std::vector<float> POSE_BODY_19_COLORS_RENDER{POSE_BODY_19_COLORS_RENDER_GPU};
    // BODY_23
    const std::map<unsigned int, std::string> POSE_BODY_23_BODY_PARTS {
        {0,  "Neck"},
        {1,  "RShoulder"},
        {2,  "RElbow"},
        {3,  "RWrist"},
        {4,  "LShoulder"},
        {5,  "LElbow"},
        {6,  "LWrist"},
        {7,  "LowerAbs"},
        {8,  "RHip"},
        {9,  "RKnee"},
        {10, "RAnkle"},
        {11, "RBigToe"},
        {12, "RSmallToe"},
        {13, "LHip"},
        {14, "LKnee"},
        {15, "LAnkle"},
        {16, "LBigToe"},
        {17, "LSmallToe"},
        {18, "Nose"},
        {19, "REye"},
        {20, "REar"},
        {21, "LEye"},
        {22, "LEar"},
        {23, "Background"}
    };
    const unsigned int POSE_BODY_23_NUMBER_PARTS               = 23u; // Equivalent to size of std::map POSE_BODY_23_BODY_PARTS - 1 (removing background)
    const std::vector<unsigned int> POSE_BODY_23_MAP_IDX       {24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71};
    #define POSE_BODY_23_PAIRS_RENDER_GPU                      { 0,1,  0,4,  1,2,  2,3,  4,5,  5,6,  0,7,  7,8,  7,13, 8,9,  9,10,10,11,11,12,13,14,14,15,15,16,16,17, 0,18,18,19,18,21,19,20,21,22}
    const std::vector<unsigned int> POSE_BODY_23_PAIRS_RENDER  {POSE_BODY_23_PAIRS_RENDER_GPU};
    const std::vector<unsigned int> POSE_BODY_23_PAIRS         { 0,1,  0,4,  1,2,  2,3,  4,5,  5,6,  0,7,  7,8,  7,13, 8,9,  9,10,10,11,10,12,13,14,14,15,15,16,15,17, 0,18,18,19,18,21,19,20,21,22, 1,20, 4,22};
    #define POSE_BODY_23_COLORS_RENDER_GPU \
        255.f,     0.f,     0.f, \
        255.f,    55.f,     0.f, \
        255.f,   110.f,     0.f, \
        255.f,   165.f,     0.f, \
        255.f,   215.f,     0.f, \
        255.f,   235.f,     0.f, \
        255.f,   255.f,     0.f, \
        255.f,     0.f,     0.f, \
        175.f,   255.f,     0.f, \
         85.f,   255.f,     0.f, \
          0.f,   255.f,     0.f, \
          0.f,   255.f,    85.f, \
          0.f,   255.f,   170.f, \
         25.f,    25.f,   128.f, \
          0.f,    85.f,   255.f, \
          0.f,   170.f,   255.f, \
          0.f,  212.5f,   255.f, \
          0.f,   255.f,   255.f, \
        255.f,     0.f,     0.f, \
        255.f,     0.f,   255.f, \
        238.f,   130.f,   238.f, \
        138.f,    43.f,   226.f, \
         75.f,     0.f,   130.f
    const std::vector<float> POSE_BODY_23_COLORS_RENDER{POSE_BODY_23_COLORS_RENDER_GPU};

    // Constant Array Parameters
    const std::array<float, (int)PoseModel::Size> POSE_CCN_DECREASE_FACTOR{
        8.f,        8.f,        8.f,        8.f,        8.f,        8.f
    };
    const std::array<unsigned int, (int)PoseModel::Size> POSE_MAX_PEAKS{
        POSE_MAX_PEOPLE,        POSE_MAX_PEOPLE,        POSE_MAX_PEOPLE,        POSE_MAX_PEOPLE,        POSE_MAX_PEOPLE,        POSE_MAX_PEOPLE
    };
    const std::array<unsigned int, (int)PoseModel::Size> POSE_NUMBER_BODY_PARTS{
        POSE_COCO_NUMBER_PARTS, POSE_MPI_NUMBER_PARTS,  POSE_MPI_NUMBER_PARTS,  POSE_BODY_18_NUMBER_PARTS,  POSE_BODY_19_NUMBER_PARTS,  POSE_BODY_23_NUMBER_PARTS
    };
    const std::array<std::vector<unsigned int>, (int)PoseModel::Size> POSE_BODY_PART_PAIRS{
        POSE_COCO_PAIRS,        POSE_MPI_PAIRS,         POSE_MPI_PAIRS,         POSE_BODY_18_PAIRS,         POSE_BODY_19_PAIRS,         POSE_BODY_23_PAIRS
    };
    const std::array<std::vector<unsigned int>, (int)PoseModel::Size> POSE_BODY_PART_PAIRS_RENDER{
        POSE_COCO_PAIRS_RENDER, POSE_MPI_PAIRS,         POSE_MPI_PAIRS,         POSE_BODY_18_PAIRS_RENDER,  POSE_BODY_19_PAIRS_RENDER,  POSE_BODY_23_PAIRS_RENDER
    };
    const std::array<std::vector<unsigned int>, (int)PoseModel::Size> POSE_MAP_IDX{
        POSE_COCO_MAP_IDX,      POSE_MPI_MAP_IDX,       POSE_MPI_MAP_IDX,       POSE_BODY_18_MAP_IDX,       POSE_BODY_19_MAP_IDX,       POSE_BODY_23_MAP_IDX
    };
    const std::array<std::vector<float>, (int)PoseModel::Size> POSE_COLORS{
        POSE_COCO_COLORS_RENDER,POSE_MPI_COLORS_RENDER, POSE_MPI_COLORS_RENDER, POSE_BODY_18_COLORS_RENDER, POSE_BODY_19_COLORS_RENDER, POSE_BODY_23_COLORS_RENDER
    };
    const std::array<std::string, (int)PoseModel::Size> POSE_PROTOTXT{
        "pose/coco/pose_deploy_linevec.prototxt",
        "pose/mpi/pose_deploy_linevec.prototxt",
        "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt",
        "pose/body_18/pose_deploy.prototxt",
        "pose/body_19/pose_deploy.prototxt",
        "pose/body_23/pose_deploy.prototxt"
    };
    const std::array<std::string, (int)PoseModel::Size> POSE_TRAINED_MODEL{
        "pose/coco/pose_iter_440000.caffemodel",
        "pose/mpi/pose_iter_160000.caffemodel",
        "pose/mpi/pose_iter_160000.caffemodel",
        "pose/body_18/pose_iter_XXXXXX.caffemodel",
        "pose/body_19/pose_iter_XXXXXX.caffemodel",
        "pose/body_23/pose_iter_XXXXXX.caffemodel"
    };
    // POSE_BODY_PART_MAPPING crashes on Windows at dynamic initialization, to avoid this crash:
    // POSE_BODY_PART_MAPPING has been moved to poseParameters.cpp and getPoseBodyPartMapping() wraps it
    // const std::array<std::map<unsigned int, std::string>, (int)PoseModel::Size>   POSE_BODY_PART_MAPPING{
        // POSE_COCO_BODY_PARTS,   POSE_MPI_BODY_PARTS,    POSE_MPI_BODY_PARTS
    // };
    const std::map<unsigned int, std::string>& getPoseBodyPartMapping(const PoseModel poseModel);

    // Default Model Parameters
    // They might be modified on running time
    const std::array<float, (int)PoseModel::Size>           POSE_DEFAULT_NMS_THRESHOLD{
        0.05f,      0.6f,       0.3f,       0.05f,      0.05f,      0.05f
    };
    const std::array<float, (int)PoseModel::Size>    POSE_DEFAULT_CONNECT_INTER_MIN_ABOVE_THRESHOLD{
        0.95f,      0.95f,      0.95f,      0.95f,      0.95f,      0.95f
        // 0.85f,      0.85f,      0.85f,      0.85f,      0.85f,      0.85f // Matlab version
    };
    const std::array<float, (int)PoseModel::Size>           POSE_DEFAULT_CONNECT_INTER_THRESHOLD{
        0.05f,      0.01f,      0.01f,      0.05f,      0.05f,      0.05f
    };
    const std::array<unsigned int, (int)PoseModel::Size>    POSE_DEFAULT_CONNECT_MIN_SUBSET_CNT{
        3,          3,          3,          3,          3,          3
    };
    const std::array<float, (int)PoseModel::Size>           POSE_DEFAULT_CONNECT_MIN_SUBSET_SCORE{
        0.4f,       0.4f,       0.4f,       0.4f,       0.4f,       0.4f
        // 0.2f,       0.4f,       0.4f,       0.4f,       0.4f,       0.4f // Matlab version
    };

    // Rendering parameters
    const auto POSE_DEFAULT_ALPHA_KEYPOINT = 0.6f;
    const auto POSE_DEFAULT_ALPHA_HEAT_MAP = 0.7f;

    // Auxiliary functions
    OP_API unsigned int poseBodyPartMapStringToKey(const PoseModel poseModel, const std::string& string);
    OP_API unsigned int poseBodyPartMapStringToKey(const PoseModel poseModel, const std::vector<std::string>& strings);
}

#endif // OPENPOSE_POSE_POSE_PARAMETERS_HPP
