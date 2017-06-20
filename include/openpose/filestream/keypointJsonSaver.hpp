#ifndef OPENPOSE_FILESTREAM_KEYPOINT_JSON_SAVER_HPP
#define OPENPOSE_FILESTREAM_KEYPOINT_JSON_SAVER_HPP

#include <string>
#include <vector>
#include <openpose/core/array.hpp>
#include "fileSaver.hpp"

namespace op
{
    class KeypointJsonSaver : public FileSaver
    {
    public:
        KeypointJsonSaver(const std::string& directoryPath);

        void save(const std::vector<Array<float>>& keypointVector, const std::string& fileName, const std::string& keypointName) const;
    };
}

#endif // OPENPOSE_FILESTREAM_KEYPOINT_JSON_SAVER_HPP
