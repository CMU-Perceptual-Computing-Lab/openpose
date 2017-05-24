#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/pose/poseParameters.hpp"
 
namespace op
{
	const std::array<std::map<unsigned char, std::string>, 3>   POSE_BODY_PART_MAPPING{ POSE_COCO_BODY_PARTS,   POSE_MPI_BODY_PARTS,    POSE_MPI_BODY_PARTS };

    unsigned char poseBodyPartMapStringToKey(const PoseModel poseModel, const std::vector<std::string>& strings)
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

	unsigned char poseBodyPartMapStringToKey(const PoseModel poseModel, const std::string& string)
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

	const std::map<unsigned char, std::string>& getPoseBodyPartMapping(const PoseModel poseModel)
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
}
