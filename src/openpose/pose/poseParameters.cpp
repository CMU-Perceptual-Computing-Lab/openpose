#include <openpose/pose/poseParameters.hpp>
#include <openpose/pose/poseParametersRender.hpp>

namespace op
{
    // Body parts mapping
    const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS {
        {0,  "Nose"},
        {1,  "Neck"},
        {2,  "RShoulder"},
        {3,  "RElbow"},
        {4,  "RWrist"},
        {5,  "LShoulder"},
        {6,  "LElbow"},
        {7,  "LWrist"},
        {8,  "MidHip"},
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
        {19, "LBigToe"},
        {20, "LSmallToe"},
        {21, "LHeel"},
        {22, "RBigToe"},
        {23, "RSmallToe"},
        {24, "RHeel"},
        {25, "Background"}
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
    const std::map<unsigned int, std::string> POSE_BODY_19_BODY_PARTS {
        {0,  "Nose"},
        {1,  "Neck"},
        {2,  "RShoulder"},
        {3,  "RElbow"},
        {4,  "RWrist"},
        {5,  "LShoulder"},
        {6,  "LElbow"},
        {7,  "LWrist"},
        {8,  "MidHip"},
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
        {0,  "Nose"},
        {1,  "RShoulder"},
        {2,  "RElbow"},
        {3,  "RWrist"},
        {4,  "LShoulder"},
        {5,  "LElbow"},
        {6,  "LWrist"},
        {7,  "RHip"},
        {8,  "RKnee"},
        {9,  "RAnkle"},
        {10, "LHip"},
        {11, "LKnee"},
        {12, "LAnkle"},
        {13, "REye"},
        {14, "LEye"},
        {15, "REar"},
        {16, "LEar"},
        {17, "LBigToe"},
        {18, "LSmallToe"},
        {19, "LHeel"},
        {20, "RBigToe"},
        {21, "RSmallToe"},
        {22, "RHeel"},
        {23, "Background"}
    };
    const std::map<unsigned int, std::string> POSE_BODY_25B_BODY_PARTS {
        {0,  "Nose"},
        {1,  "LEye"},
        {2,  "REye"},
        {3,  "LEar"},
        {4,  "REar"},
        {5,  "LShoulder"},
        {6,  "RShoulder"},
        {7,  "LElbow"},
        {8,  "RElbow"},
        {9,  "LWrist"},
        {10, "RWrist"},
        {11, "LHip"},
        {12, "RHip"},
        {13, "LKnee"},
        {14, "RKnee"},
        {15, "LAnkle"},
        {16, "RAnkle"},
        {17, "UpperNeck"},
        {18, "HeadTop"},
        {19, "LBigToe"},
        {20, "LSmallToe"},
        {21, "LHeel"},
        {22, "RBigToe"},
        {23, "RSmallToe"},
        {24, "RHeel"},
    };
    const std::map<unsigned int, std::string> POSE_BODY_135_BODY_PARTS {
        {0,  "Nose"},
        {1,  "LEye"},
        {2,  "REye"},
        {3,  "LEar"},
        {4,  "REar"},
        {5,  "LShoulder"},
        {6,  "RShoulder"},
        {7,  "LElbow"},
        {8,  "RElbow"},
        {9,  "LWrist"},
        {10, "RWrist"},
        {11, "LHip"},
        {12, "RHip"},
        {13, "LKnee"},
        {14, "RKnee"},
        {15, "LAnkle"},
        {16, "RAnkle"},
        {17, "UpperNeck"},
        {18, "HeadTop"},
        {19, "LBigToe"},
        {20, "LSmallToe"},
        {21, "LHeel"},
        {22, "RBigToe"},
        {23, "RSmallToe"},
        {24, "RHeel"},
        // Left hand
        {H135+0, "LThumb1CMC"},       {H135+1, "LThumb2Knuckles"}, {H135+2, "LThumb3IP"},   {H135+3, "LThumb4FingerTip"},
        {H135+4, "LIndex1Knuckles"},  {H135+5, "LIndex2PIP"},      {H135+6, "LIndex3DIP"},  {H135+7, "LIndex4FingerTip"},
        {H135+8, "LMiddle1Knuckles"}, {H135+9, "LMiddle2PIP"},     {H135+10, "LMiddle3DIP"},{H135+11, "LMiddle4FingerTip"},
        {H135+12, "LRing1Knuckles"},  {H135+13, "LRing2PIP"},      {H135+14, "LRing3DIP"},  {H135+15, "LRing4FingerTip"},
        {H135+16, "LPinky1Knuckles"}, {H135+17, "LPinky2PIP"},     {H135+18, "LPinky3DIP"}, {H135+19, "LPinky4FingerTip"},
        // Right hand
        {H135+20, "RThumb1CMC"},      {H135+21, "RThumb2Knuckles"},{H135+22, "RThumb3IP"},  {H135+23, "RThumb4FingerTip"},
        {H135+24, "RIndex1Knuckles"}, {H135+25, "RIndex2PIP"},     {H135+26, "RIndex3DIP"}, {H135+27, "RIndex4FingerTip"},
        {H135+28, "RMiddle1Knuckles"},{H135+29, "RMiddle2PIP"},    {H135+30, "RMiddle3DIP"},{H135+31, "RMiddle4FingerTip"},
        {H135+32, "RRing1Knuckles"},  {H135+33, "RRing2PIP"},      {H135+34, "RRing3DIP"},  {H135+35, "RRing4FingerTip"},
        {H135+36, "RPinky1Knuckles"}, {H135+37, "RPinky2PIP"},     {H135+38, "RPinky3DIP"}, {H135+39, "RPinky4FingerTip"},
        // Face
        {F135+0, "FaceContour0"},   {F135+1, "FaceContour1"},   {F135+2, "FaceContour2"},   {F135+3, "FaceContour3"},   {F135+4, "FaceContour4"},   {F135+5, "FaceContour5"},   // Contour 1/3
        {F135+6, "FaceContour6"},   {F135+7, "FaceContour7"},   {F135+8, "FaceContour8"},   {F135+9, "FaceContour9"},   {F135+10, "FaceContour10"}, {F135+11, "FaceContour11"}, // Contour 2/3
        {F135+12, "FaceContour12"}, {F135+13, "FaceContour13"}, {F135+14, "FaceContour14"}, {F135+15, "FaceContour15"}, {F135+16, "FaceContour16"},                             // Contour 3/3
        {F135+17, "REyeBrow0"},  {F135+18, "REyeBrow1"},  {F135+19, "REyeBrow2"},  {F135+20, "REyeBrow3"},  {F135+21, "REyeBrow4"}, // Right eyebrow
        {F135+22, "LEyeBrow4"},  {F135+23, "LEyeBrow3"},  {F135+24, "LEyeBrow2"},  {F135+25, "LEyeBrow1"},  {F135+26, "LEyeBrow0"}, // Left eyebrow
        {F135+27, "NoseUpper0"}, {F135+28, "NoseUpper1"}, {F135+29, "NoseUpper2"}, {F135+30, "NoseUpper3"}, // Upper nose
        {F135+31, "NoseLower0"}, {F135+32, "NoseLower1"}, {F135+33, "NoseLower2"}, {F135+34, "NoseLower3"}, {F135+35, "NoseLower4"}, // Lower nose
        {F135+36, "REye0"}, {F135+37, "REye1"}, {F135+38, "REye2"}, {F135+39, "REye3"}, {F135+40, "REye4"}, {F135+41, "REye5"}, // Right eye
        {F135+42, "LEye0"}, {F135+43, "LEye1"}, {F135+44, "LEye2"}, {F135+45, "LEye3"}, {F135+46, "LEye4"}, {F135+47, "LEye5"}, // Left eye
        {F135+48, "OMouth0"}, {F135+49, "OMouth1"}, {F135+50, "OMouth2"}, {F135+51, "OMouth3"}, {F135+52, "OMouth4"}, {F135+53, "OMouth5"}, // Outer mouth 1/2
        {F135+54, "OMouth6"}, {F135+55, "OMouth7"}, {F135+56, "OMouth8"}, {F135+57, "OMouth9"}, {F135+58, "OMouth10"}, {F135+59, "OMouth11"}, // Outer mouth 2/2
        {F135+60, "IMouth0"}, {F135+61, "IMouth1"}, {F135+62, "IMouth2"}, {F135+63, "IMouth3"}, {F135+64, "IMouth4"}, {F135+65, "IMouth5"}, {F135+66, "IMouth6"}, {F135+67, "IMouth7"}, // Inner mouth
        {F135+68, "RPupil"}, {F135+69, "LPupil"}, // Pupils
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
    const std::map<unsigned int, std::string> POSE_CAR_12_PARTS {
        {0,  "FRWheel"},
        {1,  "FLWheel"},
        {2,  "BRWheel"},
        {3,  "BLWheel"},
        {4,  "FRLight"},
        {5,  "FLLight"},
        {6,  "BRLight"},
        {7,  "BLLight"},
        {8,  "FRTop"},
        {9,  "FLTop"},
        {10, "BRTop"},
        {11, "BLTop"},
        {12, "Background"},
    };
    const std::map<unsigned int, std::string> POSE_CAR_22_PARTS {
        {0,  "FLWheel"},
        {1,  "BLWheel"},
        {2,  "FRWheel"},
        {3,  "BRWheel"},
        {4,  "FRFogLight"},
        {5,  "FLFogLight"},
        {6,  "FRLight"},
        {7,  "FLLight"},
        {8,  "Grilles"},
        {9,  "FBumper"},
        {10, "LMirror"},
        {11, "RMirror"},
        {12, "FRTop"},
        {13, "FLTop"},
        {14, "BLTop"},
        {15, "BRTop"},
        {16, "BLLight"},
        {17, "BRLight"},
        {18, "Trunk"},
        {19, "BBumper"},
        {20, "BLCorner"},
        {21, "BRCorner"},
        {22, "Background"},
    };
    const std::array<std::vector<unsigned int>, (int)PoseModel::Size> POSE_MAP_INDEX{
        // BODY_25
        std::vector<unsigned int>{
            0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3, 4,5, 8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 20,21, 28,29, 40,41,42,43,44,45, 46,47,48,49,50,51
        },
        // COCO
        std::vector<unsigned int>{
            12,13, 20,21, 14,15, 16,17, 22,23, 24,25, 0,1, 2,3, 4,5, 6,7, 8,9, 10,11, 28,29, 30,31, 34,35, 32,33, 36,37, 18,19, 26,27
        },
        // MPI_15
        std::vector<unsigned int>{
            0,1, 2,3, 4,5, 6,7, 8,9, 10,11, 12,13, 14,15, 16,17, 18,19, 20,21, 22,23, 24,25, 26,27
        },
        // MPI_15_4
        std::vector<unsigned int>{
            0,1, 2,3, 4,5, 6,7, 8,9, 10,11, 12,13, 14,15, 16,17, 18,19, 20,21, 22,23, 24,25, 26,27
        },
        // BODY_19
        std::vector<unsigned int>{
            0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3, 4,5, 8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 20,21, 28,29
        },
        // BODY_19_X2
        std::vector<unsigned int>{
            0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3, 4,5, 8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 20,21, 28,29
        },
        // BODY_19N
        std::vector<unsigned int>{
            0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3, 4,5, 8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 20,21, 28,29
        },
        // BODY_25E
        std::vector<unsigned int>{
            // Minimum spanning tree
            0,1,   2,3,   4,5,   6,7,   8,9,  10,11,  12,13, 14,15, 16,17, 18,19, 20,21, 22,23, 24,25, 26,27, 28,29, 30,31, 32,33, 34,35, 36,37, 38,39, 40,41, 42,43, 44,45, 46,47,
            // Redundant ones
            48,49, 50,51, 52,53, 54,55, 56,57, 58,59, 60,61, 62,63, 64,65, 66,67, 68,69, 70,71, 72,73, 74,75
        },
        // CAR_12
        std::vector<unsigned int>{
            0,1,   2,3,   4,5,   6,7,   8,9,  10,11,  12,13, 14,15, 16,17, 18,19, 20,21
        },
        // BODY_25D
        std::vector<unsigned int>{
            0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3, 4,5, 8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 20,21, 28,29, 40,41,42,43,44,45, 46,47,48,49,50,51
        },
        // BODY_23
        std::vector<unsigned int>{
            // Minimum spanning tree
            0,1, 2,3, 4,5, 6,7, 8,9, 10,11, 12,13, 14,15, 16,17, 18,19, 20,21, 22,23, 24,25, 26,27, 28,29, 30,31, 32,33, 34,35, 36,37, 38,39, 40,41, 42,43,
            // Redundant ones
               44,45,46,47,    48,49,50,51,  52,53,54,55,    56,57,58,59,  60,61, 62,63,  64,65,66,67,    68,69, 70,71
        },
        // CAR_22
        std::vector<unsigned int>{
            0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,
            38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63
        },
        // BODY_19E
        std::vector<unsigned int>{
            // Minimum spanning tree
            0,1,   2,3,   4,5,   6,7,   8,9,  10,11,  12,13, 14,15, 16,17, 18,19, 20,21, 22,23, 24,25, 26,27, 28,29, 30,31, 32,33, 34,35, 36,37,
            // Redundant ones
            38,39, 40,41, 42,43, 44,45, 46,47, 48,49, 50,51, 52,53, 54,55, 56,57, 58,59, 60,61
        },
        // BODY_25B
        std::vector<unsigned int>{
            // Minimum spanning tree
            // |------------------------------------------- COCO Body -------------------------------------------|
            0,1, 2,3, 4,5, 6,7,   8,9,10,11, 12,13,14,15,16,17,18,19, 20,21, 22,23,  24,25, 26,27,   28,29, 30,31,
            // Redundant ones
            // |------------------ Foot ------------------| |-- MPII --|
               32,33, 34,35, 36,37,   38,39, 40,41, 42,43,  44,45, 46,47,
            // Redundant ones
            // MPII redundant, ears, ears-shoulders, shoulders-wrists, wrists, wrists-hips, hips, ankles)
                48,49, 50,51, 52,53,  54,55, 56,57,    58,59, 60,61,   62,63, 64,65, 66,67, 68,69, 70,71
        },
        // BODY_135
        std::vector<unsigned int>{
            // Minimum spanning tree
            // |------------------------------------------- COCO Body -------------------------------------------|
            0,1, 2,3, 4,5, 6,7,   8,9,10,11, 12,13,14,15,16,17,18,19, 20,21, 22,23,  24,25, 26,27,   28,29, 30,31,
            // Redundant ones
            // |------------------ Foot ------------------| |-- MPII --|
               32,33, 34,35, 36,37,   38,39, 40,41, 42,43,  44,45, 46,47,
            // Redundant ones
            // MPII redundant, ears, ears-shoulders, shoulders-wrists, wrists, wrists-hips, hips, ankles)
                   48,49,     50,51,  52,53, 54,55,    56,57, 58,59,   60,61, 62,63, 64,65, 66,67, 68,69,
            // Hand
            // Left Hand
                70,71, 72,73, 74,75, 76,77, 78,79,   80,81, 82,83, 84,85, 86,87, 88,89,
                90,91, 92,93, 94,95, 96,97, 98,99,   100,101, 102,103, 104,105, 106,107, 108,109,
            // Right Hand
                110,111, 112,113, 114,115, 116,117, 118,119,   120,121, 122,123, 124,125, 126,127, 128,129,
                130,131, 132,133, 134,135, 136,137, 138,139,   140,141, 142,143, 144,145, 146,147, 148,149,
            // Face
            // COCO-Face
               150,151, 152,153, 154,155,
            // Contour
               156,157, 158,159, 160,161, 162,163, 164,165, 166,167, 168,169, 170,171, 172,173, 174,175, 176,177, 178,179, 180,181,
               182,183, 184,185, 186,187,
            // Countour-Eyebrow + Eyebrows
               188,189, 190,191, 192,193, 194,195, 196,197, 198,199, 200,201, 202,203, 204,205, 206,207, 208,209,
            // Eyebrow-Nose + Nose
               210,211, 212,213, 214,215, 216,217, 218,219, 220,221, 222,223, 224,225, 226,227, 228,229,
            // Nose-Eyes + Eyes
               230,231, 232,233, 234,235, 236,237, 238,239, 240,241, 242,243, 244,245, 246,247, 248,249, 250,251,
               252,253,
            // Nose-Mouth + Outer Mouth
               254,255, 256,257, 258,259, 260,261, 262,263, 264,265, 266,267, 268,269, 270,271, 272,273, 274,275,
               276,277,
            // Outer-Inner + Inner Mouth
               278,279, 280,281, 282,283, 284,285, 286,287, 288,289, 290,291, 292,293, 294,295,
            // Eyes-Pupils
               296,297, 298,299, 300,301, 302,303
        },
    };
    // POSE_BODY_PART_MAPPING on HPP crashes on Windows at dynamic initialization if it's on hpp
    const std::array<std::map<unsigned int, std::string>, (int)PoseModel::Size> POSE_BODY_PART_MAPPING{
        POSE_BODY_25_BODY_PARTS,POSE_COCO_BODY_PARTS,   POSE_MPI_BODY_PARTS,    POSE_MPI_BODY_PARTS,
        POSE_BODY_19_BODY_PARTS,POSE_BODY_19_BODY_PARTS,POSE_BODY_19_BODY_PARTS,POSE_BODY_25_BODY_PARTS,
        POSE_CAR_12_PARTS,      POSE_BODY_25_BODY_PARTS,POSE_BODY_23_BODY_PARTS,POSE_CAR_22_PARTS,
        POSE_BODY_19_BODY_PARTS,POSE_BODY_25B_BODY_PARTS,POSE_BODY_135_BODY_PARTS
    };

    const std::array<std::string, (int)PoseModel::Size> POSE_PROTOTXT{
        "pose/body_25/pose_deploy.prototxt",
        "pose/coco/pose_deploy_linevec.prototxt",
        "pose/mpi/pose_deploy_linevec.prototxt",
        "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt",
        "pose/body_19/pose_deploy.prototxt",
        "pose/body_19_x2/pose_deploy.prototxt",
        "pose/body_19n/pose_deploy.prototxt",
        "pose/body_25e/pose_deploy.prototxt",
        "car/car_12/pose_deploy.prototxt",
        "pose/body_25d/pose_deploy.prototxt",
        "pose/body_23/pose_deploy.prototxt",
        "car/car_22/pose_deploy.prototxt",
        "pose/body_19e/pose_deploy.prototxt",
        "pose/body_25b/pose_deploy.prototxt",
        "pose/body_135/pose_deploy.prototxt",
    };
    const std::array<std::string, (int)PoseModel::Size> POSE_TRAINED_MODEL{
        "pose/body_25/pose_iter_584000.caffemodel",
        "pose/coco/pose_iter_440000.caffemodel",
        "pose/mpi/pose_iter_160000.caffemodel",
        "pose/mpi/pose_iter_160000.caffemodel",
        "pose/body_19/pose_iter_XXXXXX.caffemodel",
        "pose/body_19_x2/pose_iter_XXXXXX.caffemodel",
        "pose/body_19n/pose_iter_XXXXXX.caffemodel",
        "pose/body_25e/pose_iter_XXXXXX.caffemodel",
        "car/car_12/pose_iter_XXXXXX.caffemodel",
        "pose/body_25d/pose_iter_XXXXXX.caffemodel",
        "pose/body_23/pose_iter_XXXXXX.caffemodel",
        "car/car_22/pose_iter_XXXXXX.caffemodel",
        "pose/body_19e/pose_iter_XXXXXX.caffemodel",
        "pose/body_25b/pose_iter_XXXXXX.caffemodel",
        "pose/body_135/pose_iter_XXXXXX.caffemodel",
    };

    // Constant Array Parameters
    // POSE_NUMBER_BODY_PARTS equivalent to size of std::map POSE_BODY_XX_BODY_PARTS - 1 (removing background)
    const std::array<unsigned int, (int)PoseModel::Size> POSE_NUMBER_BODY_PARTS{
        25, 18, 15, 15, 19, 19, 19, 25, 12, 25, 23, 22, 19, 25, 135
    };
    const std::array<std::vector<unsigned int>, (int)PoseModel::Size> POSE_BODY_PART_PAIRS{
        // BODY_25
        std::vector<unsigned int>{
            1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18,   14,19,19,20,14,21, 11,22,22,23,11,24
        },
        // COCO
        std::vector<unsigned int>{
            1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10,  1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17,  2,16,  5,17
        },
        // MPI_15
        std::vector<unsigned int>{POSE_MPI_PAIRS_RENDER_GPU},
        // MPI_15_4
        std::vector<unsigned int>{POSE_MPI_PAIRS_RENDER_GPU},
        // BODY_19
        std::vector<unsigned int>{
            1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18
        },
        // BODY_19_X2
        std::vector<unsigned int>{
            1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18
        },
        // BODY_19N
        std::vector<unsigned int>{
            1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18
        },
        // BODY_25E
        std::vector<unsigned int>{
            // Minimum spanning tree
            1,8,   1,2, 2,3, 3,4,   1,5, 5,6, 6,7,   8,9, 9,10, 10,11,   8,12, 12,13, 13,14,   1,0, 0,15, 15,17, 0,16, 16,18,
            // Foot (minimum spanning tree)
            14,19,19,20,14,21, 11,22,22,23,11,24,
            // Redundant ones
            // Ears-shoulders, shoulders-hips, shoulders-wrists, hips-ankles, wrists,  ankles, wrists-hips, small toes-ankles)
                 2,17, 5,18,        2,9, 5,12,      2,4, 5,7,    9,11, 12,14,   4,7,   11,14,   4,9, 7,12,   11,23, 14,20
        },
        // CAR_12
        std::vector<unsigned int>{
            4,5,   4,6,   4,0,   0,2,   4,8,   8,10,   5,7,   5,1,   1,3,   5,9,   9,11
        },
        // BODY_25D
        std::vector<unsigned int>{
            1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18,   14,19,19,20,14,21, 11,22,22,23,11,24
        },
        // BODY_23
        std::vector<unsigned int>{
            // Minimum spanning tree
            0,1, 1,2, 2,3, 0,4, 4,5,  5,6,   7,8,   8,9,  10,11, 11,12,  0,13, 13,15,  0,14, 14,16, 12,17, 17,18, 12,19,  9,20, 20,21,  9,22,  1,7,   4,10,
            // Redundant ones
            // Ears-shoulders, ears,  hips, shoulders-wrists, hips-ankles, wrists, ankles, wrists-hips, small toes-ankles)
            //     +2.4%       +0.2%  +0.3%      +0.1%       -0.1% (bad!)  +0.2%   +0.2%      +0.1%       -0.1% (bad!)
                1,15, 4,16,    15,16, 7,10,    1,3, 4,6,      7,9, 10,12,   3,6,   9,12,    3,7, 6,10,     9,21, 12,18
        },
        // CAR_22
        std::vector<unsigned int>{
        //       Wheels                Lights                   Top                       Front                Mirrors
            0,1,1,3,3,2,2,0,   6,7,7,16,16,17,17,6,   12,13,13,14,14,15,15,12,   6,8,7,8,6,9,7,9,6,4,7,5,   12,11,13,10,
        //            Back                  Vertical         Back-light replacement
            16,18,17,18,16,19,17,19,   0,7,3,17,6,12,16,14,   6,21,7,20,3,21,20,14
        },
        // BODY_19E
        std::vector<unsigned int>{
            // Minimum spanning tree
            1,8,   1,2, 2,3, 3,4,   1,5, 5,6, 6,7,   8,9, 9,10, 10,11,   8,12, 12,13, 13,14,   1,0, 0,15, 15,17, 0,16, 16,18,
            // Redundant ones
            // Ears-shoulders, shoulders-hips, shoulders-wrists, hips-ankles, wrists,  ankles, wrists-hips, small toes-ankles)
                 2,17, 5,18,        2,9, 5,12,      2,4, 5,7,    9,11, 12,14,   4,7,   11,14,   4,9, 7,12//,   11,23, 14,20
        },
        // BODY_25B
        std::vector<unsigned int>{
            // Minimum spanning tree
            // |------------------------------------------- COCO Body -------------------------------------------|
            0,1, 0,2, 1,3, 2,4,   0,5, 0,6,   5,7, 6,8,   7,9, 8,10,   5,11, 6,12,   11,13, 12,14,   13,15, 14,16,
            // |------------------ Foot ------------------| |-- MPII --|
               15,19, 19,20, 15,21,   16,22, 22,23, 16,24,   5,17, 5,18,
            // Redundant ones
            // MPII redundant, ears, ears-shoulders, shoulders-wrists, wrists, wrists-hips, hips, ankles)
                 6,17, 6,18,   3,4,     3,5, 4,6,        5,9, 6,10,     9,10,  9,11, 10,12, 11,12, 15,16
        },
        // BODY_135
        std::vector<unsigned int>{
            // Minimum spanning tree
            // |------------------------------------------- COCO Body -------------------------------------------|
            0,1, 0,2, 1,3, 2,4,   0,5, 0,6,   5,7, 6,8,   7,9, 8,10,   5,11, 6,12,   11,13, 12,14,   13,15, 14,16,
            // |------------------ Foot ------------------| |-- MPII --|
               15,19, 19,20, 15,21,   16,22, 22,23, 16,24,   5,17,17,18,
            // Redundant ones
            // MPII redundant, ears, ears-shoulders, shoulders-wrists, wrists, wrists-hips, hips, ankles)
                    6,17,      3,4,     3,5, 4,6,        5,9, 6,10,     9,10,  9,11, 10,12, 11,12, 15,16,
            // Left Hand
               9,H135+0, H135+0,H135+1, H135+1,H135+2, H135+2,H135+3,           9,H135+4, H135+4,H135+5, H135+5,H135+6, H135+6,H135+7,
               9,H135+8, H135+8,H135+9, H135+9,H135+10, H135+10,H135+11,        9,H135+12, H135+12,H135+13, H135+13,H135+14, H135+14,H135+15,
               9,H135+16, H135+16,H135+17, H135+17,H135+18, H135+18,H135+19,
            // Right Hand
               10,H135+20, H135+20,H135+21, H135+21,H135+22, H135+22,H135+23,   10,H135+24, H135+24,H135+25, H135+25,H135+26, H135+26,H135+27,
               10,H135+28, H135+28,H135+29, H135+29,H135+30, H135+30,H135+31,   10,H135+32, H135+32,H135+33, H135+33,H135+34, H135+34,H135+35,
               10,H135+36, H135+36,H135+37, H135+37,H135+38, H135+38,H135+39,
            // Face
            // COCO-Face
            0,F135+30, 2,F135+39,1,F135+42,
            // Contour
               F135+0,F135+1, F135+1,F135+2, F135+2,F135+3, F135+3,F135+4, F135+4,F135+5, F135+5,F135+6, F135+6,F135+7, F135+7,F135+8,
               F135+8,F135+9, F135+9,F135+10, F135+10,F135+11, F135+11,F135+12, F135+12,F135+13, F135+13,F135+14, F135+14,F135+15,
               F135+15,F135+16,
            // Countour-Eyebrow + Eyebrows
               F135+0,F135+17, F135+16,F135+26, F135+17,F135+18, F135+18,F135+19, F135+19,F135+20, F135+20,F135+21, F135+21,F135+22,
               F135+22,F135+23, F135+23,F135+24, F135+24,F135+25, F135+25,F135+26,
            // Eyebrow-Nose + Nose
               F135+21,F135+27, F135+22,F135+27, F135+27,F135+28, F135+28,F135+29, F135+29,F135+30, F135+30,F135+33, F135+33,F135+32,
               F135+32,F135+31, F135+33,F135+34, F135+34,F135+35,
            // Nose-Eyes + Eyes
               F135+27,F135+39, F135+27,F135+42, F135+36,F135+37, F135+37,F135+38, F135+38,F135+39, F135+39,F135+40, F135+40,F135+41,
               F135+42,F135+43, F135+43,F135+44, F135+44,F135+45, F135+45,F135+46, F135+46,F135+47,
            // Nose-Mouth + Outer Mouth
               F135+33,F135+51, F135+48,F135+49, F135+49,F135+50, F135+50,F135+51, F135+51,F135+52, F135+52,F135+53, F135+53,F135+54,
               F135+54,F135+55, F135+55,F135+56, F135+56,F135+57, F135+57,F135+58, F135+58,F135+59,
            // Outer-Inner + Inner Mouth
               F135+48,F135+60, F135+54,F135+64, F135+60,F135+61, F135+61,F135+62, F135+62,F135+63, F135+63,F135+64, F135+64,F135+65,
               F135+65,F135+66, F135+66,F135+67,
            // Eyes-Pupils
               F135+36,F135+68, F135+39,F135+68, F135+42,F135+69, F135+45,F135+69
        }
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

    unsigned int getPoseMaxPeaks()
    {
        try
        {
            return POSE_MAX_PEOPLE;
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
            return (poseModel != PoseModel::BODY_19_X2 ? 8.f : 4.f);
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
            for (const auto& string : strings)
                for (const auto& pair : poseBodyPartMapping)
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
    // They might be modified on running time
    float getPoseDefaultNmsThreshold(const PoseModel poseModel, const bool maximizePositives)
    {
        try
        {
            // MPI models
            if (poseModel == PoseModel::MPI_15)
                return 0.6f;
            else if (poseModel == PoseModel::MPI_15_4)
                return 0.3f;
            // Non-MPI models
            else
                return (maximizePositives ? 0.02f : 0.05f);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    float getPoseDefaultConnectInterMinAboveThreshold(const bool maximizePositives)
    {
        try
        {
            return (maximizePositives ? 0.75f : 0.95f);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    float getPoseDefaultConnectInterThreshold(const PoseModel poseModel, const bool maximizePositives)
    {
        try
        {
            // MPI models
            if (poseModel == PoseModel::MPI_15 || poseModel == PoseModel::MPI_15_4)
                return 0.01f;
            // Non-MPI models
            else
                // return (maximizePositives ? 0.01f : 0.5f); // 0.485 but much less false positive connections
                // return (maximizePositives ? 0.01f : 0.1f); // 0.518
                // return (maximizePositives ? 0.01f : 0.075f); // 0.521
                return (maximizePositives ? 0.01f : 0.05f); // 0.523
                // return (maximizePositives ? 0.01f : 0.01f); // 0.527 but huge amount of false positives joints
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    unsigned int getPoseDefaultMinSubsetCnt(const bool maximizePositives)
    {
        try
        {
            return (maximizePositives ? 2u : 3u);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0u;
        }
    }

    float getPoseDefaultConnectMinSubsetScore(const bool maximizePositives)
    {
        try
        {
            return (maximizePositives ? 0.05f : 0.4f);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    bool addBkgChannel(const PoseModel poseModel)
    {
        try
        {
            return (POSE_BODY_PART_MAPPING[(int)poseModel].size() != POSE_NUMBER_BODY_PARTS[(int)poseModel]);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }
}
