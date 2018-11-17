#include <openpose/pose/poseParameters.hpp>
#include <openpose/pose/poseParametersRender.hpp>
#include <openpose/pose/renderPose.hpp>
#include <openpose/utilities/fastMath.hpp>
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
            const auto numberBodyParts = getPoseNumberBodyParts(poseModel);
            const auto numberBodyPartsPlusBkg = numberBodyParts+1;

            // PAFs
            for (auto bodyPart = 0u; bodyPart < bodyPartPairs.size(); bodyPart+=2)
            {
                const auto bodyPartPairsA = bodyPartPairs.at(bodyPart);
                const auto bodyPartPairsB = bodyPartPairs.at(bodyPart+1);
                const auto mapIdxA = numberBodyPartsPlusBkg + mapIdx.at(bodyPart);
                const auto mapIdxB = numberBodyPartsPlusBkg + mapIdx.at(bodyPart+1);
                const auto baseLine = partToName.at(bodyPartPairsA) + "->" + partToName.at(bodyPartPairsB);
                partToName[mapIdxA] = baseLine + "(X)";
                partToName[mapIdxB] = baseLine + "(Y)";
            }
            // Distance PAFs
            if (poseModel == PoseModel::BODY_25D)
            {
                for (auto bodyPart = 0u; bodyPart < numberBodyParts; bodyPart++)
                {
                    if (bodyPart != 1u)
                    {
                        const auto mapIdxD = (unsigned int)
                            (numberBodyPartsPlusBkg + bodyPartPairs.size() + 2*bodyPart - (bodyPart > 0 ? 2 : 0));
                        const auto baseLine = partToName.at(1) + "->" + partToName.at(bodyPart);
                        partToName[mapIdxD] = baseLine + "(X)";
                        partToName[mapIdxD+1] = baseLine + "(Y)";
                    }
                }
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

    PoseRenderer::~PoseRenderer()
    {
    }
}
