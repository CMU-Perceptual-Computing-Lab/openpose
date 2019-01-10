#include <openpose/pose/poseParameters.hpp>
#include <openpose/pose/poseParametersRender.hpp>

namespace op
{
    const std::array<std::vector<float>, (int)PoseModel::Size> POSE_SCALES{
        std::vector<float>{POSE_BODY_25_SCALES_RENDER_GPU},       // BODY_25
        std::vector<float>{POSE_COCO_SCALES_RENDER_GPU},          // COCO
        std::vector<float>{POSE_MPI_SCALES_RENDER_GPU},           // MPI_15
        std::vector<float>{POSE_MPI_SCALES_RENDER_GPU},           // MPI_15_4
        std::vector<float>{POSE_BODY_19_SCALES_RENDER_GPU},       // BODY_19
        std::vector<float>{POSE_BODY_19_SCALES_RENDER_GPU},       // BODY_19_X2
        std::vector<float>{POSE_BODY_19_SCALES_RENDER_GPU},       // BODY_19N
        std::vector<float>{POSE_BODY_25_SCALES_RENDER_GPU},       // BODY_25E
        std::vector<float>{POSE_BODY_65_SCALES_RENDER_GPU},       // BODY_65
        std::vector<float>{POSE_CAR_12_SCALES_RENDER_GPU},        // CAR_12
        std::vector<float>{POSE_BODY_25_SCALES_RENDER_GPU},       // BODY_25D
        std::vector<float>{POSE_BODY_23_SCALES_RENDER_GPU},       // BODY_23
        std::vector<float>{POSE_CAR_22_SCALES_RENDER_GPU},        // CAR_22
        std::vector<float>{POSE_BODY_19_SCALES_RENDER_GPU},       // BODY_19E
        std::vector<float>{POSE_BODY_25B_SCALES_RENDER_GPU},      // BODY_25B
        std::vector<float>{POSE_BODY_95_SCALES_RENDER_GPU},       // BODY_95
    };
    const std::array<std::vector<float>, (int)PoseModel::Size> POSE_COLORS{
        std::vector<float>{POSE_BODY_25_COLORS_RENDER_GPU},       // BODY_25
        std::vector<float>{POSE_COCO_COLORS_RENDER_GPU},          // COCO
        std::vector<float>{POSE_MPI_COLORS_RENDER_GPU},           // MPI_15
        std::vector<float>{POSE_MPI_COLORS_RENDER_GPU},           // MPI_15_4
        std::vector<float>{POSE_BODY_19_COLORS_RENDER_GPU},       // BODY_19
        std::vector<float>{POSE_BODY_19_COLORS_RENDER_GPU},       // BODY_19_X2
        std::vector<float>{POSE_BODY_19_COLORS_RENDER_GPU},       // BODY_19N
        std::vector<float>{POSE_BODY_25_COLORS_RENDER_GPU},       // BODY_25E
        std::vector<float>{POSE_BODY_65_COLORS_RENDER_GPU},       // BODY_65
        std::vector<float>{POSE_CAR_12_COLORS_RENDER_GPU},        // CAR_12
        std::vector<float>{POSE_BODY_25_COLORS_RENDER_GPU},       // BODY_25D
        std::vector<float>{POSE_BODY_23_COLORS_RENDER_GPU},       // BODY_23
        std::vector<float>{POSE_CAR_22_COLORS_RENDER_GPU},        // CAR_22
        std::vector<float>{POSE_BODY_19_COLORS_RENDER_GPU},       // BODY_19E
        std::vector<float>{POSE_BODY_25B_COLORS_RENDER_GPU},      // BODY_25B
        std::vector<float>{POSE_BODY_95_COLORS_RENDER_GPU},       // BODY_95
    };
    const std::array<std::vector<unsigned int>, (int)PoseModel::Size> POSE_BODY_PART_PAIRS_RENDER{
        std::vector<unsigned int>{POSE_BODY_25_PAIRS_RENDER_GPU},       // BODY_25
        std::vector<unsigned int>{POSE_COCO_PAIRS_RENDER_GPU},          // COCO
        std::vector<unsigned int>{POSE_MPI_PAIRS_RENDER_GPU},           // MPI_15
        std::vector<unsigned int>{POSE_MPI_PAIRS_RENDER_GPU},           // MPI_15_4
        std::vector<unsigned int>{POSE_BODY_19_PAIRS_RENDER_GPU},       // BODY_19
        std::vector<unsigned int>{POSE_BODY_19_PAIRS_RENDER_GPU},       // BODY_19_X2
        std::vector<unsigned int>{POSE_BODY_19_PAIRS_RENDER_GPU},       // BODY_19N
        std::vector<unsigned int>{POSE_BODY_25_PAIRS_RENDER_GPU},       // BODY_25E
        std::vector<unsigned int>{POSE_BODY_65_PAIRS_RENDER_GPU},       // BODY_65
        std::vector<unsigned int>{POSE_CAR_12_PAIRS_RENDER_GPU},        // CAR_12
        std::vector<unsigned int>{POSE_BODY_25_PAIRS_RENDER_GPU},       // BODY_25D
        std::vector<unsigned int>{POSE_BODY_23_PAIRS_RENDER_GPU},       // BODY_23
        std::vector<unsigned int>{POSE_CAR_22_PAIRS_RENDER_GPU},        // CAR_22
        std::vector<unsigned int>{POSE_BODY_19_PAIRS_RENDER_GPU},       // BODY_19E
        std::vector<unsigned int>{POSE_BODY_25B_PAIRS_RENDER_GPU},      // BODY_25B
        std::vector<unsigned int>{POSE_BODY_95_PAIRS_RENDER_GPU},       // BODY_95
    };

    // Rendering functions
    const std::vector<float>& getPoseScales(const PoseModel poseModel)
    {
        try
        {
            return POSE_SCALES.at((int)poseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return POSE_SCALES[(int)poseModel];
        }
    }

    const std::vector<float>& getPoseColors(const PoseModel poseModel)
    {
        try
        {
            return POSE_COLORS.at((int)poseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return POSE_COLORS[(int)poseModel];
        }
    }

    const std::vector<unsigned int>& getPoseBodyPartPairsRender(const PoseModel poseModel)
    {
        try
        {
            return POSE_BODY_PART_PAIRS_RENDER.at((int)poseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return POSE_BODY_PART_PAIRS_RENDER[(int)poseModel];
        }
    }

    unsigned int getNumberElementsToRender(const PoseModel poseModel)
    {
        try
        {
            return (unsigned int)(getPoseBodyPartMapping(poseModel).size()
                                  + getPosePartPairs(poseModel).size()/2 + 3
                                  + (poseModel == PoseModel::BODY_25D
                                     ? 2*(getPoseNumberBodyParts(poseModel) - 1) : 0)
                                  );
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0u;
        }
    }
}
