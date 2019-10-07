#include <openpose/filestream/imageSaver.hpp>
#include <openpose/filestream/fileStream.hpp>

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

    ImageSaver::~ImageSaver()
    {
    }

    void ImageSaver::saveImages(const Matrix& cvOutputData, const std::string& fileName) const
    {
        try
        {
            saveImages(std::vector<Matrix>{cvOutputData}, fileName);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void ImageSaver::saveImages(const std::vector<Matrix>& matOutputDatas, const std::string& fileName) const
    {
        try
        {
            // Record cv::mat
            if (!matOutputDatas.empty())
            {
                // File path (no extension)
                const auto fileNameNoExtension = getNextFileName(fileName) + "_rendered";

                // Get names for each image
                std::vector<std::string> fileNames(matOutputDatas.size());
                for (auto i = 0u; i < fileNames.size(); i++)
                    fileNames[i] = {fileNameNoExtension + (i != 0 ? "_" + std::to_string(i) : "") + "." + mImageFormat};

                // Save each image
                for (auto i = 0u; i < matOutputDatas.size(); i++)
                    saveImage(matOutputDatas[i], fileNames[i]);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
