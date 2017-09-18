#include <openpose/filestream/fileStream.hpp>
#include <openpose/filestream/imageSaver.hpp>

namespace op
{
    ImageSaver::ImageSaver(const std::string& directoryPath, const std::string& imageFormat) :
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

    void ImageSaver::saveImages(const std::vector<cv::Mat>& cvOutputDatas, const std::string& fileName) const
    {
        try
        {
            // Record cv::mat
            if (!cvOutputDatas.empty())
            {
                // File path (no extension)
                const auto fileNameNoExtension = getNextFileName(fileName) + "_rendered";

                // Get names for each image
                std::vector<std::string> fileNames(cvOutputDatas.size());
                for (auto i = 0u; i < fileNames.size(); i++)
                    fileNames[i] = {fileNameNoExtension + (i != 0 ? "_" + std::to_string(i) : "") + "." + mImageFormat};

                // Save each image
                for (auto i = 0u; i < cvOutputDatas.size(); i++)
                    saveImage(cvOutputDatas[i], fileNames[i]);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
