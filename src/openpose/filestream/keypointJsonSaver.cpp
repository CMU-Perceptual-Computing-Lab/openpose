#include <openpose/filestream/fileStream.hpp>
#include <openpose/filestream/keypointJsonSaver.hpp>

namespace op
{
    KeypointJsonSaver::KeypointJsonSaver(const std::string& directoryPath) :
        FileSaver{directoryPath}
    {
    }

    void KeypointJsonSaver::save(const std::vector<std::pair<Array<float>, std::string>>& keypointVector,
                                 const std::string& fileName, const bool humanReadable) const
    {
        try
        {
            // Record json
            const auto finalFileName = getNextFileName(fileName) + ".json";
            saveKeypointsJson(keypointVector, finalFileName, humanReadable);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
