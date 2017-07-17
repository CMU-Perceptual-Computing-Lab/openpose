#include <openpose/utilities/openCv.hpp>
#include <openpose/filestream/fileStream.hpp>
#include <openpose/filestream/heatMapSaver.hpp>

namespace op
{
    HeatMapSaver::HeatMapSaver(const std::string& directoryPath, const std::string& imageFormat) :
        FileSaver{directoryPath},
        mImageFormat{imageFormat}
    {
        try
        {
            if (mImageFormat.empty())
                error("The string imageFormat should not be empty.", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void HeatMapSaver::saveHeatMaps(const std::vector<Array<float>>& heatMaps, const std::string& fileName) const
    {
        try
        {
            // Record cv::mat
            if (!heatMaps.empty())
            {
                // File path (no extension)
                const auto fileNameNoExtension = getNextFileName(fileName) + "_heatmaps";

                // Get names for each heatMap
                std::vector<std::string> fileNames(heatMaps.size());
                for (auto i = 0; i < fileNames.size(); i++)
                    fileNames[i] = {fileNameNoExtension + (i != 0 ? "_" + std::to_string(i) : "") + "." + mImageFormat};

                // heatMaps -> cvOutputDatas
                std::vector<cv::Mat> cvOutputDatas(heatMaps.size());
                for (auto i = 0; i < cvOutputDatas.size(); i++)
                    unrollArrayToUCharCvMat(cvOutputDatas[i], heatMaps[i]);

                // Save each heatMap
                for (auto i = 0; i < cvOutputDatas.size(); i++)
                    saveImage(cvOutputDatas[i], fileNames[i]);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
