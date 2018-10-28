#include <openpose/filestream/fileStream.hpp>
#include <openpose/filestream/keypointSaver.hpp>

namespace op
{
    KeypointSaver::KeypointSaver(const std::string& directoryPath, const DataFormat format) :
        FileSaver{directoryPath},
        mFormat{format}
    {
    }

    KeypointSaver::~KeypointSaver()
    {
    }

    void KeypointSaver::saveKeypoints(const std::vector<Array<float>>& keypointVector, const std::string& fileName, const std::string& keypointName) const
    {
        try
        {
            if (!keypointVector.empty())
            {
                // File path (no extension)
                const auto fileNameNoExtension = getNextFileName(fileName) + "_" + keypointName;

                // Get vector of people poses
                std::vector<cv::Mat> cvMatPoses(keypointVector.size());
                for (auto i = 0u; i < keypointVector.size(); i++)
                    cvMatPoses[i] = keypointVector[i].getConstCvMat();

                // Get names inside file
                std::vector<std::string> keypointVectorNames(cvMatPoses.size());
                for (auto i = 0u; i < cvMatPoses.size(); i++)
                    keypointVectorNames[i] = {keypointName + "_" + std::to_string(i)};

                // Record people poses in desired format
                saveData(cvMatPoses, keypointVectorNames, fileNameNoExtension, mFormat);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
