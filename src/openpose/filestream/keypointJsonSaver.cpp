#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/filestream/fileStream.hpp>
#include <openpose/filestream/keypointJsonSaver.hpp>

namespace op
{
    KeypointJsonSaver::KeypointJsonSaver(const std::string& directoryPath) :
        FileSaver{directoryPath}
    {
    }

    void KeypointJsonSaver::saveKeypoints(const std::vector<Array<float>>& keypointVector, const std::string& fileName, const std::string& keypointName) const
    {
        try
        {
            // Record json
            if (!keypointVector.empty())
            {
                // File path (no extension)
                const auto fileNameNoExtension = getNextFileName(fileName) + "_" + keypointName;

                const bool humanReadable = true;
                for (auto i = 0; i < keypointVector.size(); i++)
                {
                    const auto finalFileName = fileNameNoExtension + (i != 0 ? "_" + std::to_string(i) : "") + ".json";
                    savePoseJson(keypointVector[i], finalFileName, humanReadable);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
