#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/filestream/fileStream.hpp"
#include "openpose/filestream/poseJsonSaver.hpp"

namespace op
{
    PoseJsonSaver::PoseJsonSaver(const std::string& directoryPath) :
        FileSaver{directoryPath}
    {
    }

    void PoseJsonSaver::savePoseKeyPoints(const std::vector<Array<float>>& poseKeyPointsVector, const std::string& fileName) const
    {
        try
        {
            // Record json
            if (!poseKeyPointsVector.empty())
            {
                // File path (no extension)
                const auto fileNameNoExtension = getNextFileName(fileName) + "_pose";

                const bool humanReadable = true;
                for (auto i = 0; i < poseKeyPointsVector.size(); i++)
                {
                    const auto fileName = fileNameNoExtension + (i != 0 ? "_" + std::to_string(i) : "") + ".json";
                    savePoseJson(poseKeyPointsVector[i], fileName, humanReadable);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
