#include <openpose/pose/poseParametersRender.hpp>
#include <openpose/pose/poseParameters.hpp>

namespace op
{
    // Body parts mapping
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
    const std::map<unsigned int, std::string> POSE_BODY_19_BODY_PARTS {
        {0,  "Nose"},
        {1,  "Neck"},
        {2,  "RShoulder"},
        {3,  "RElbow"},
        {4,  "RWrist"},
        {5,  "LShoulder"},
        {6,  "LElbow"},
        {7,  "LWrist"},
        {8,  "LowerAbs"},
        {9,  "RHip"},
        {10, "RKnee"},
        {11, "RAnkle"},
        {12, "LHip"},
        {13, "LKnee"},
        {14, "LAnkle"},
        {15, "REye"},
        {16, "LEye"},
        {17, "REar"},
        {18, "LEar"},
        {19, "Background"}
    };
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
    const std::map<unsigned int, std::string> POSE_BODY_59_BODY_PARTS {
        // Body
        {0,  "Nose"},
        {1,  "Neck"},
        {2,  "RShoulder"},
        {3,  "RElbow"},
        {4,  "RWrist"},
        {5,  "LShoulder"},
        {6,  "LElbow"},
        {7,  "LWrist"},
        {8,  "LowerAbs"},
        {9,  "RHip"},
        {10, "RKnee"},
        {11, "RAnkle"},
        {12, "LHip"},
        {13, "LKnee"},
        {14, "LAnkle"},
        {15, "REye"},
        {16, "LEye"},
        {17, "REar"},
        {18, "LEar"},
        // Left hand
        {19, "LThumb1CMC"},         {20, "LThumb2Knuckles"},{21, "LThumb3IP"},  {22, "LThumb4FingerTip"},
        {23, "LIndex1Knuckles"},    {24, "LIndex2PIP"},     {25, "LIndex3DIP"}, {26, "LIndex4FingerTip"},
        {27, "LMiddle1Knuckles"},   {28, "LMiddle2PIP"},    {29, "LMiddle3DIP"},{30, "LMiddle4FingerTip"},
        {31, "LRing1Knuckles"},     {32, "LRing2PIP"},      {33, "LRing3DIP"},  {34, "LRing4FingerTip"},
        {35, "LPinky1Knuckles"},    {36, "LPinky2PIP"},     {37, "LPinky3DIP"}, {38, "LPinky4FingerTip"},
        // Right hand
        {39, "RThumb1CMC"},         {40, "RThumb2Knuckles"},{41, "RThumb3IP"},  {42, "RThumb4FingerTip"},
        {43, "RIndex1Knuckles"},    {44, "RIndex2PIP"},     {45, "RIndex3DIP"}, {46, "RIndex4FingerTip"},
        {47, "RMiddle1Knuckles"},   {48, "RMiddle2PIP"},    {49, "RMiddle3DIP"},{50, "RMiddle4FingerTip"},
        {51, "RRing1Knuckles"},     {52, "RRing2PIP"},      {53, "RRing3DIP"},  {54, "RRing4FingerTip"},
        {55, "RPinky1Knuckles"},    {56, "RPinky2PIP"},     {57, "RPinky3DIP"}, {58, "RPinky4FingerTip"},
        // Background
        {59, "Background"},
    };
    // Hand legend:
    //     - Thumb:
    //         - Carpometacarpal Joints (CMC)
    //         - Interphalangeal Joints (IP)
    //     - Other fingers:
    //         - Knuckles or Metacarpophalangeal Joints (MCP)
    //         - PIP (Proximal Interphalangeal Joints)
    //         - DIP (Distal Interphalangeal Joints)
    //     - All fingers:
    //         - Fingertips
    // More information: Page 6 of http://www.mccc.edu/~behrensb/documents/TheHandbig.pdf
    const std::array<std::vector<unsigned int>, (int)PoseModel::Size> POSE_MAP_INDEX{
        // COCO
        std::vector<unsigned int>{
            31,32, 39,40, 33,34, 35,36, 41,42, 43,44, 19,20, 21,22, 23,24, 25,26, 27,28, 29,30, 47,48, 49,50, 53,54, 51,52, 55,56, 37,38, 45,46
        },
        // MPI_15
        std::vector<unsigned int>{
            16,17, 18,19, 20,21, 22,23, 24,25, 26,27, 28,29, 30,31, 32,33, 34,35, 36,37, 38,39, 40,41, 42,43
        },
        // MPI_15_4
        std::vector<unsigned int>{
            16,17, 18,19, 20,21, 22,23, 24,25, 26,27, 28,29, 30,31, 32,33, 34,35, 36,37, 38,39, 40,41, 42,43
        },
        // BODY_18
        std::vector<unsigned int>{
            31,32, 39,40, 33,34, 35,36, 41,42, 43,44, 19,20, 21,22, 23,24, 25,26, 27,28, 29,30, 47,48, 49,50, 53,54, 51,52, 55,56, 37,38, 45,46
        },
        // BODY_19
        std::vector<unsigned int>{
            20,21, 34,35, 42,43, 36,37, 38,39, 44,45, 46,47, 26,27, 22,23, 24,25, 28,29, 30,31, 32,33, 50,51, 52,53, 56,57, 54,55, 58,59, 40,41, 48,49
        },
        // BODY_19_X2
        std::vector<unsigned int>{
            20,21, 34,35, 42,43, 36,37, 38,39, 44,45, 46,47, 26,27, 22,23, 24,25, 28,29, 30,31, 32,33, 50,51, 52,53, 56,57, 54,55, 58,59, 40,41, 48,49
        },
        // BODY_23
        std::vector<unsigned int>{
            24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71
        },
        // BODY_59
        std::vector<unsigned int>{
            60,61, 74,75, 82,83, 76,77, 78,79, 84,85, 86,87, 66,67, 62,63, 64,65, 68,69, 70,71, 72,73, 90,91, 92,93, 96,97, 94,95, 98,99, 80,81, 88,89, // Body
            100,101, 102,103, 104,105, 106,107, 108,109, 110,111, 112,113, 114,115, 116,117, 118,119,
            120,121, 122,123, 124,125, 126,127, 128,129, 130,131, 132,133, 134,135, 136,137, 138,139,// Left hand
            140,141, 142,143, 144,145, 146,147, 148,149, 150,151, 152,153, 154,155, 156,157, 158,159,
            160,161, 162,163, 164,165, 166,167, 168,169, 170,171, 172,173, 174,175, 176,177, 178,179 // Right hand
        },
        // BODY_19N
        std::vector<unsigned int>{
            20,21, 34,35, 42,43, 36,37, 38,39, 44,45, 46,47, 26,27, 22,23, 24,25, 28,29, 30,31, 32,33, 50,51, 52,53, 56,57, 54,55, 58,59, 40,41, 48,49
        },
        // BODY_19b
        std::vector<unsigned int>{
            20,21, 34,35, 42,43, 36,37, 38,39, 44,45, 46,47, 26,27, 22,23, 24,25, 28,29, 30,31, 32,33, 50,51, 52,53, 56,57, 54,55, 58,59, 40,41, 48,49, 60,61, 62,63
        },
    };
    // POSE_BODY_PART_MAPPING on HPP crashes on Windows at dynamic initialization if it's on hpp
    const std::array<std::map<unsigned int, std::string>, (int)PoseModel::Size>   POSE_BODY_PART_MAPPING{
        POSE_COCO_BODY_PARTS,   POSE_MPI_BODY_PARTS,    POSE_MPI_BODY_PARTS,    POSE_COCO_BODY_PARTS,
        POSE_BODY_19_BODY_PARTS,POSE_BODY_19_BODY_PARTS,POSE_BODY_23_BODY_PARTS,POSE_BODY_59_BODY_PARTS,
        POSE_BODY_19_BODY_PARTS,POSE_BODY_19_BODY_PARTS
    };

    const std::array<std::string, (int)PoseModel::Size> POSE_PROTOTXT{
        "pose/coco/pose_deploy_linevec.prototxt",
        "pose/mpi/pose_deploy_linevec.prototxt",
        "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt",
        "pose/body_18/pose_deploy.prototxt",
        "pose/body_19/pose_deploy.prototxt",
        "pose/body_19_x2/pose_deploy.prototxt",
        "pose/body_23/pose_deploy.prototxt",
        "pose/body_59/pose_deploy.prototxt",
        "pose/body_19n/pose_deploy.prototxt",
        "pose/body_19b/pose_deploy.prototxt",
    };
    const std::array<std::string, (int)PoseModel::Size> POSE_TRAINED_MODEL{
        "pose/coco/pose_iter_440000.caffemodel",
        "pose/mpi/pose_iter_160000.caffemodel",
        "pose/mpi/pose_iter_160000.caffemodel",
        "pose/body_18/pose_iter_XXXXXX.caffemodel",
        "pose/body_19/pose_iter_XXXXXX.caffemodel",
        "pose/body_19_x2/pose_iter_XXXXXX.caffemodel",
        "pose/body_23/pose_iter_XXXXXX.caffemodel",
        "pose/body_59/pose_iter_XXXXXX.caffemodel",
        "pose/body_19n/pose_iter_XXXXXX.caffemodel",
        "pose/body_19b/pose_iter_XXXXXX.caffemodel",
    };

    // Constant Array Parameters
    // POSE_NUMBER_BODY_PARTS equivalent to size of std::map POSE_BODY_XX_BODY_PARTS - 1 (removing background)
    const std::array<unsigned int, (int)PoseModel::Size> POSE_NUMBER_BODY_PARTS{
        18, 15, 15, 18, 19, 19, 23, 59, 19, 19
    };
    const std::array<std::vector<unsigned int>, (int)PoseModel::Size> POSE_BODY_PART_PAIRS{
        // COCO
        std::vector<unsigned int>{
            1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10,  1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17,  2,16,  5,17
        },
        // MPI_15
        std::vector<unsigned int>{POSE_MPI_PAIRS_RENDER_GPU},
        // MPI_15_4
        std::vector<unsigned int>{POSE_MPI_PAIRS_RENDER_GPU},
        // BODY_18
        std::vector<unsigned int>{
            1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10,  1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17,  2,16,  5,17
        },
        // BODY_19
        std::vector<unsigned int>{
            1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18
        },
        // BODY_19_X2
        std::vector<unsigned int>{
            1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18
        },
        // BODY_23
        std::vector<unsigned int>{
            0,1,  0,4,  1,2,  2,3,  4,5,  5,6,  0,7,  7,8,  7,13, 8,9,  9,10,10,11,10,12,13,14,14,15,15,16,15,17, 0,18,18,19,18,21,19,20,21,22, 1,20, 4,22
        },
        // BODY_59
        std::vector<unsigned int>{
            1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18,// Body
            7,19, 19,20, 20,21, 21,22, 7,23, 23,24, 24,25, 25,26, 7,27, 27,28, 28,29, 29,30, 7,31, 31,32, 32,33, 33,34, 7,35, 35,36, 36,37, 37,38,      // Left hand
            4,39, 39,40, 40,41, 41,42, 4,43, 43,44, 44,45, 45,46, 4,47, 47,48, 48,49, 49,50, 4,51, 51,52, 52,53, 53,54, 4,55, 55,56, 56,57, 57,58       // Right hand
        },
        // BODY_19N
        std::vector<unsigned int>{
            1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18
        },
        // BODY_19b
        std::vector<unsigned int>{
            1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18, 2,9, 5,12
        },
    };
    const std::array<unsigned int, (int)PoseModel::Size> POSE_MAX_PEAKS{
        POSE_MAX_PEOPLE,    // COCO
        POSE_MAX_PEOPLE,    // MPI_15
        POSE_MAX_PEOPLE,    // MPI_15_4
        POSE_MAX_PEOPLE,    // BODY_18
        POSE_MAX_PEOPLE,    // BODY_19
        POSE_MAX_PEOPLE,    // BODY_19_X2
        POSE_MAX_PEOPLE,    // BODY_23
        POSE_MAX_PEOPLE,    // BODY_59
        POSE_MAX_PEOPLE,    // BODY_19N
        POSE_MAX_PEOPLE,    // BODY_19b
    };
    const std::array<float, (int)PoseModel::Size> POSE_CCN_DECREASE_FACTOR{
        8.f,    // COCO
        8.f,    // MPI_15
        8.f,    // MPI_15_4
        8.f,    // BODY_18
        8.f,    // BODY_19
        4.f,    // BODY_19_X2
        8.f,    // BODY_23
        8.f,    // BODY_59
        8.f,    // BODY_19N
        8.f,    // BODY_19b
    };

    // Default Model Parameters
    // They might be modified on running time
    const std::array<float, (int)PoseModel::Size>           POSE_DEFAULT_NMS_THRESHOLD{
        0.05f,      0.6f,       0.3f,       0.05f,      0.05f,      0.05f,      0.05f,      0.05f,      0.05f,      0.05f
    };
    const std::array<float, (int)PoseModel::Size>    POSE_DEFAULT_CONNECT_INTER_MIN_ABOVE_THRESHOLD{
        0.95f,      0.95f,      0.95f,      0.95f,      0.95f,      0.95f,      0.95f,      0.95f,      0.95f,      0.95f
        // 0.85f,      0.85f,      0.85f,      0.85f,      0.85f,      0.85f // Matlab version
    };
    const std::array<float, (int)PoseModel::Size>           POSE_DEFAULT_CONNECT_INTER_THRESHOLD{
        0.05f,      0.01f,      0.01f,      0.05f,      0.05f,      0.05f,      0.05f,      0.05f,      0.05f,      0.05f
    };
    const std::array<unsigned int, (int)PoseModel::Size>    POSE_DEFAULT_CONNECT_MIN_SUBSET_CNT{
        3,          3,          3,          3,          3,          3,          3,          3,          3,          3
    };
    const std::array<float, (int)PoseModel::Size>           POSE_DEFAULT_CONNECT_MIN_SUBSET_SCORE{
        0.4f,       0.4f,       0.4f,       0.4f,       0.4f,       0.4f,       0.4f,       0.4f,       0.4f,       0.4f
        // 0.2f,       0.4f,       0.4f,       0.4f,       0.4f,       0.4f // Matlab version
    };

    const std::map<unsigned int, std::string>& getPoseBodyPartMapping(const PoseModel poseModel)
    {
        try
        {
            return POSE_BODY_PART_MAPPING.at((int)poseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return POSE_BODY_PART_MAPPING[(int)poseModel];
        }
    }

    const std::string& getPoseProtoTxt(const PoseModel poseModel)
    {
        try
        {
            return POSE_PROTOTXT.at((int)poseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return POSE_PROTOTXT[(int)poseModel];
        }
    }

    const std::string& getPoseTrainedModel(const PoseModel poseModel)
    {
        try
        {
            return POSE_TRAINED_MODEL.at((int)poseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return POSE_TRAINED_MODEL[(int)poseModel];
        }
    }

    unsigned int getPoseNumberBodyParts(const PoseModel poseModel)
    {
        try
        {
            return POSE_NUMBER_BODY_PARTS.at((int)poseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0u;
        }
    }

    const std::vector<unsigned int>& getPosePartPairs(const PoseModel poseModel)
    {
        try
        {
            return POSE_BODY_PART_PAIRS.at((int)poseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return POSE_BODY_PART_PAIRS[(int)poseModel];
        }
    }

    const std::vector<unsigned int>& getPoseMapIndex(const PoseModel poseModel)
    {
        try
        {
            return POSE_MAP_INDEX.at((int)poseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return POSE_MAP_INDEX[(int)poseModel];
        }
    }

    unsigned int getPoseMaxPeaks(const PoseModel poseModel)
    {
        try
        {
            return POSE_MAX_PEAKS.at((int)poseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0u;
        }
    }

    float getPoseNetDecreaseFactor(const PoseModel poseModel)
    {
        try
        {
            return POSE_CCN_DECREASE_FACTOR.at((int)poseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    unsigned int poseBodyPartMapStringToKey(const PoseModel poseModel, const std::vector<std::string>& strings)
    {
        try
        {
            const auto& poseBodyPartMapping = POSE_BODY_PART_MAPPING[(int)poseModel];
            for (auto& string : strings)
                for (auto& pair : poseBodyPartMapping)
                    if (pair.second == string)
                        return pair.first;
            error("String(s) could not be found.", __LINE__, __FUNCTION__, __FILE__);
            return 0;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0;
        }
    }

    unsigned int poseBodyPartMapStringToKey(const PoseModel poseModel, const std::string& string)
    {
        try
        {
            return poseBodyPartMapStringToKey(poseModel, std::vector<std::string>{string});
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0;
        }
    }

    // Default Model Parameters
    float getPoseDefaultNmsThreshold(const PoseModel poseModel)
    {
        try
        {
            return POSE_DEFAULT_NMS_THRESHOLD.at((int)poseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    float getPoseDefaultConnectInterMinAboveThreshold(const PoseModel poseModel)
    {
        try
        {
            return POSE_DEFAULT_CONNECT_INTER_MIN_ABOVE_THRESHOLD.at((int)poseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    float getPoseDefaultConnectInterThreshold(const PoseModel poseModel)
    {
        try
        {
            return POSE_DEFAULT_CONNECT_INTER_THRESHOLD.at((int)poseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    unsigned int getPoseDefaultMinSubsetCnt(const PoseModel poseModel)
    {
        try
        {
            return POSE_DEFAULT_CONNECT_MIN_SUBSET_CNT.at((int)poseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0u;
        }
    }

    float getPoseDefaultConnectMinSubsetScore(const PoseModel poseModel)
    {
        try
        {
            return POSE_DEFAULT_CONNECT_MIN_SUBSET_SCORE.at((int)poseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }
}
