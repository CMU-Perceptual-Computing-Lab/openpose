#ifndef OPENPOSE_FILESTREAM_FILE_STREAM_HPP
#define OPENPOSE_FILESTREAM_FILE_STREAM_HPP

#include <opencv2/core/core.hpp> // cv::Mat
#include <opencv2/highgui/highgui.hpp> // CV_LOAD_IMAGE_ANYDEPTH, CV_IMWRITE_PNG_COMPRESSION
#include <openpose/core/common.hpp>
#include <openpose/filestream/enumClasses.hpp>

namespace op
{
    OP_API DataFormat stringToDataFormat(const std::string& dataFormat);

    // Save/load json, xml, yaml, yml
    OP_API void saveData(const std::vector<cv::Mat>& cvMats, const std::vector<std::string>& cvMatNames, const std::string& fileNameNoExtension, const DataFormat format);

    OP_API void saveData(const cv::Mat& cvMat, const std::string cvMatName, const std::string& fileNameNoExtension, const DataFormat format);

    OP_API std::vector<cv::Mat> loadData(const std::vector<std::string>& cvMatNames, const std::string& fileNameNoExtension, const DataFormat format);

    OP_API cv::Mat loadData(const std::string& cvMatName, const std::string& fileNameNoExtension, const DataFormat format);

    // Json - Saving as *.json not available in OpenCV verions < 3.0, this function is a quick fix
    OP_API void saveKeypointsJson(const Array<float>& keypoints, const std::string& keypointName, const std::string& fileName, const bool humanReadable);

    // It will save a bunch of Array<float> elements
    OP_API void saveKeypointsJson(const std::vector<std::pair<Array<float>, std::string>>& keypointVector, const std::string& fileName, const bool humanReadable);

    // Save/load image
    OP_API void saveImage(const cv::Mat& cvMat, const std::string& fullFilePath, const std::vector<int>& openCvCompressionParams = {CV_IMWRITE_JPEG_QUALITY, 100, CV_IMWRITE_PNG_COMPRESSION, 9});

    OP_API cv::Mat loadImage(const std::string& fullFilePath, const int openCvFlags = CV_LOAD_IMAGE_ANYDEPTH);

    OP_API std::vector<std::array<Rectangle<float>, 2>> loadHandDetectorTxt(const std::string& txtFilePath);
}

#endif // OPENPOSE_FILESTREAM_FILE_STREAM_HPP
