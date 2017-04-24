#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/filestream/fileStream.hpp"
#include "openpose/filestream/poseJsonSaver.hpp"

namespace op
{
    PoseJsonSaver::PoseJsonSaver(const std::string& directoryPath) :
        FileSaver{directoryPath}
    {
    }

    void PoseJsonSaver::savePoseVector(const std::vector<Array<float>>& poseVector, const std::string& fileName) const
    {
        try
        {
            // Record json
            if (!poseVector.empty())
            {
                // File path (no extension)
                const auto fileNameNoExtension = getNextFileName(fileName) + "_pose";

                const bool humanReadable = true;
                for (auto i = 0; i < poseVector.size(); i++)
                {
                    const auto fileName = fileNameNoExtension + (i != 0 ? "_" + std::to_string(i) : "") + ".json";
                    savePoseJson(poseVector[i], fileName, humanReadable);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
