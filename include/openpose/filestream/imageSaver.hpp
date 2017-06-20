#ifndef OPENPOSE_FILESTREAM_IMAGE_SAVER_HPP
#define OPENPOSE_FILESTREAM_IMAGE_SAVER_HPP

#include <string>
#include <vector>
#include <opencv2/core/core.hpp> // cv::Mat
#include "fileSaver.hpp"

namespace op
{
    class ImageSaver : public FileSaver
    {
    public:
        ImageSaver(const std::string& directoryPath, const std::string& imageFormat);

        void saveImages(const std::vector<cv::Mat>& cvOutputDatas, const std::string& fileName) const;

    private:
        const std::string mImageFormat;
    };
}

#endif // OPENPOSE_FILESTREAM_IMAGE_SAVER_HPP
