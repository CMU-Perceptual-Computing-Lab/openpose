#include <openpose/filestream/heatMapSaver.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/filestream/fileStream.hpp>

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

    HeatMapSaver::~HeatMapSaver()
    {
    }

    void HeatMapSaver::saveHeatMaps(const std::vector<Array<float>>& heatMaps, const std::string& fileName) const
    {
        try
        {
            // Record cv::mat
            if (!heatMaps.empty())
            {
                // File path (no extension)
                const auto fileNameNoExtension = getNextFileName(fileName);

                // Get names for each heatMap
                std::vector<std::string> fileNames(heatMaps.size());
                for (auto i = 0u; i < fileNames.size(); i++)
                    fileNames[i] = {fileNameNoExtension + (i != 0 ? "_" + std::to_string(i) : "") + "." + mImageFormat};

                // Saving on custom floating type "float". Format it:
                // First, the number of dimensions of the array.
                // Next elements: the size of each dimension.
                // Next: all the elements.
                if (mImageFormat == "float")
                {
                    if (heatMaps.size() > 1)
                        error("Float only implemented for heatMaps.size() == 1.", __LINE__, __FUNCTION__, __FILE__);
                    for (auto i = 0u; i < heatMaps.size(); i++)
                        saveFloatArray(heatMaps[i], fileNames[i]);
                }
                // Saving on integer type (jpg, png, etc.)
                else
                {
                    // heatMaps -> cvOutputDatas
                    std::vector<Matrix> cvOutputDatas(heatMaps.size());
                    for (auto i = 0u; i < cvOutputDatas.size(); i++)
                        unrollArrayToUCharCvMat(cvOutputDatas[i], heatMaps[i]);
                    // Save each heatMap
                    for (auto i = 0u; i < cvOutputDatas.size(); i++)
                        saveImage(cvOutputDatas[i], fileNames[i]);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
