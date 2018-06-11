#include <openpose/pose/poseParameters.hpp>
#include <openpose/pose/poseParametersRender.hpp>
#include <openpose/pose/renderPose.hpp>
#include <openpose/pose/poseRenderer.hpp>

namespace op
{
    std::map<unsigned int, std::string> createPartToName(const PoseModel poseModel)
    {
        try
        {
            auto partToName = getPoseBodyPartMapping(poseModel);
            const auto& bodyPartPairs = getPosePartPairs(poseModel);
            const auto& mapIdx = getPoseMapIndex(poseModel);
            const auto numberBodyPartsPlusBkg = getPoseNumberBodyParts(poseModel)+1;

            for (auto bodyPart = 0u; bodyPart < bodyPartPairs.size(); bodyPart+=2)
            {
                const auto bodyPartPairsA = bodyPartPairs.at(bodyPart);
                const auto bodyPartPairsB = bodyPartPairs.at(bodyPart+1);
                const auto mapIdxA = numberBodyPartsPlusBkg + mapIdx.at(bodyPart);
                const auto mapIdxB = numberBodyPartsPlusBkg + mapIdx.at(bodyPart+1);
                partToName[mapIdxA] = partToName.at(bodyPartPairsA) + "->" + partToName.at(bodyPartPairsB) + "(X)";
                partToName[mapIdxB] = partToName.at(bodyPartPairsA) + "->" + partToName.at(bodyPartPairsB) + "(Y)";
            }

            return partToName;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::map<unsigned int, std::string>();
        }
    }

    PoseRenderer::PoseRenderer(const PoseModel poseModel) :
        mPoseModel{poseModel},
        mPartIndexToName{createPartToName(poseModel)}
    {
    }
}
