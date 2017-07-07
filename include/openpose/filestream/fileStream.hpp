#ifndef OPENPOSE_FILESTREAM_FILE_STREAM_HPP
#define OPENPOSE_FILESTREAM_FILE_STREAM_HPP

#include <array>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp> // cv::Mat
#include <opencv2/highgui/highgui.hpp> // CV_LOAD_IMAGE_ANYDEPTH, CV_IMWRITE_PNG_COMPRESSION
#include <openpose/core/array.hpp>
#include <openpose/core/rectangle.hpp>
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
    void saveKeypointsJson(const Array<float>& keypoints, const std::string& keypointName, const std::string& fileName, const bool humanReadable);

    // It will save a bunch of Array<float> elements
    void saveKeypointsJson(const std::vector<std::pair<Array<float>, std::string>>& keypointVector, const std::string& fileName, const bool humanReadable);

    // Save/load image
    void saveImage(const cv::Mat& cvMat, const std::string& fullFilePath, const std::vector<int>& openCvCompressionParams = {CV_IMWRITE_JPEG_QUALITY, 100, CV_IMWRITE_PNG_COMPRESSION, 9});

    cv::Mat loadImage(const std::string& fullFilePath, const int openCvFlags = CV_LOAD_IMAGE_ANYDEPTH);

    std::vector<std::array<Rectangle<float>, 2>> loadHandDetectorTxt(const std::string& txtFilePath);
}

#endif // OPENPOSE_FILESTREAM_FILE_STREAM_HPP
