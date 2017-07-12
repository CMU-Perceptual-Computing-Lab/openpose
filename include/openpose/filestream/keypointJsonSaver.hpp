#ifndef OPENPOSE_FILESTREAM_KEYPOINT_JSON_SAVER_HPP
#define OPENPOSE_FILESTREAM_KEYPOINT_JSON_SAVER_HPP

#include <string>
#include <vector>
#include <openpose/core/array.hpp>
#include <openpose/core/macros.hpp>
#include "fileSaver.hpp"

namespace op
{
    class OP_API KeypointJsonSaver : public FileSaver
    {
    public:
        KeypointJsonSaver(const std::string& directoryPath);

        void save(const std::vector<std::pair<Array<float>, std::string>>& keypointVector,
                  const std::string& fileName, const bool humanReadable = true) const;
    };
}

#endif // OPENPOSE_FILESTREAM_KEYPOINT_JSON_SAVER_HPP
