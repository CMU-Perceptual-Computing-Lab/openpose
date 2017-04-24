#include <opencv2/core/core.hpp>
#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/filestream/fileStream.hpp"
#include "openpose/filestream/poseSaver.hpp"

namespace op
{
    PoseSaver::PoseSaver(const std::string& directoryPath, const DataFormat format) :
        FileSaver{directoryPath},
        mFormat{format}
    {
    }

    void PoseSaver::savePoseVector(const std::vector<Array<float>>& poseVector, const std::string& fileName) const
    {
        try
        {
            if (!poseVector.empty())
            {
                // File path (no extension)
                const auto fileNameNoExtension = getNextFileName(fileName) + "_pose";

                // Get vector of people poses
                std::vector<cv::Mat> cvMatPoses(poseVector.size());
                for (auto i = 0; i < poseVector.size(); i++)
                    cvMatPoses[i] = poseVector[i].getConstCvMat();

                // Get names inside file
                std::vector<std::string> poseVectorNames(cvMatPoses.size());
                for (auto i = 0; i < cvMatPoses.size(); i++)
                    poseVectorNames[i] = {"pose_" + std::to_string(i)};

                // Record people poses in desired format
                saveData(cvMatPoses, poseVectorNames, fileNameNoExtension, mFormat);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
