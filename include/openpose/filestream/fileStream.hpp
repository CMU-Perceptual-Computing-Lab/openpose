#ifndef OPENPOSE_FILESTREAM_FILE_STREAM_HPP
#define OPENPOSE_FILESTREAM_FILE_STREAM_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/enumClasses.hpp>
#include <openpose/utilities/openCv.hpp>

namespace op
{
    OP_API std::string dataFormatToString(const DataFormat dataFormat);

    OP_API DataFormat stringToDataFormat(const std::string& dataFormat);

    // Save custom float format
    // Example to read it in Python, assuming a (18 x 300 x 500) size Array
    // x = np.fromfile(heatMapFullPath, dtype=np.float32)
    // assert x[0] == 3 # First parameter saves the number of dimensions (18x300x500 = 3 dimensions)
    // shape_x = x[1:1+int(x[0])]
    // assert len(shape_x[0]) == 3 # Number of dimensions
    // assert shape_x[0] == 18 # Size of the first dimension
    // assert shape_x[1] == 300 # Size of the second dimension
    // assert shape_x[2] == 500 # Size of the third dimension
    // arrayData = x[1+int(round(x[0])):]
    OP_API void saveFloatArray(const Array<float>& array, const std::string& fullFilePath);

    // Save/load json, xml, yaml, yml
    OP_API void saveData(
        const std::vector<Matrix>& opMats, const std::vector<std::string>& cvMatNames,
        const std::string& fileNameNoExtension, const DataFormat dataFormat);

    OP_API void saveData(
        const Matrix& opMat, const std::string cvMatName, const std::string& fileNameNoExtension,
        const DataFormat dataFormat);

    OP_API std::vector<Matrix> loadData(
        const std::vector<std::string>& cvMatNames, const std::string& fileNameNoExtension,
        const DataFormat dataFormat);

    OP_API Matrix loadData(
        const std::string& cvMatName, const std::string& fileNameNoExtension, const DataFormat dataFormat);

    // Json - Saving as *.json not available in OpenCV versions < 3.0, this function is a quick fix
    OP_API void savePeopleJson(
        const Array<float>& keypoints, const std::vector<std::vector<std::array<float,3>>>& candidates,
        const std::string& keypointName, const std::string& fileName, const bool humanReadable);

    // It will save a bunch of Array<float> elements
    OP_API void savePeopleJson(
        const std::vector<std::pair<Array<float>, std::string>>& keypointVector,
        const std::vector<std::vector<std::array<float,3>>>& candidates, const std::string& fileName,
        const bool humanReadable);

    // Save/load image
    OP_API void saveImage(
        const Matrix& matrix, const std::string& fullFilePath,
        const std::vector<int>& openCvCompressionParams
            = {getCvImwriteJpegQuality(), 100, getCvImwritePngCompression(), 9});

    OP_API Matrix loadImage(const std::string& fullFilePath, const int openCvFlags = getCvLoadImageAnydepth());

    OP_API std::vector<std::array<Rectangle<float>, 2>> loadHandDetectorTxt(const std::string& txtFilePath);
}

#endif // OPENPOSE_FILESTREAM_FILE_STREAM_HPP
