#ifndef OPENPOSE__FILESTREAM__FILE_STREAM_HPP
#define OPENPOSE__FILESTREAM__FILE_STREAM_HPP

#include <string>
#include <vector>
#include <opencv2/core/core.hpp> // cv::Mat, cv::Size
#include <opencv2/highgui/highgui.hpp> // CV_LOAD_IMAGE_ANYDEPTH, CV_IMWRITE_PNG_COMPRESSION
#include "../core/array.hpp"
#include "enumClasses.hpp"

namespace op
{
    DataFormat stringToDataFormat(const std::string& dataFormat);

    // Save/load json, xml, yaml, yml
    void saveData(const std::vector<cv::Mat>& cvMats, const std::vector<std::string>& cvMatNames, const std::string& fileNameNoExtension, const DataFormat format);

    void saveData(const cv::Mat& cvMat, const std::string cvMatName, const std::string& fileNameNoExtension, const DataFormat format);

    std::vector<cv::Mat> loadData(const std::vector<std::string>& cvMatNames, const std::string& fileNameNoExtension, const DataFormat format);

    cv::Mat loadData(const std::string& cvMatName, const std::string& fileNameNoExtension, const DataFormat format);

    // Json - Saving as *.json not available in OpenCV verions < 3.0, this function is a quick fix
    void savePoseJson(const Array<float>& pose, const std::string& fileName, const bool humanReadable);

    // Save/load image
    void saveImage(const cv::Mat& cvMat, const std::string& fullFilePath, const std::vector<int>& openCvCompressionParams = {CV_IMWRITE_JPEG_QUALITY, 100, CV_IMWRITE_PNG_COMPRESSION, 9});

    cv::Mat loadImage(const std::string& fullFilePath, const int openCvFlags = CV_LOAD_IMAGE_ANYDEPTH);
}

#endif // OPENPOSE__FILESTREAM__FILE_STREAM_HPP
