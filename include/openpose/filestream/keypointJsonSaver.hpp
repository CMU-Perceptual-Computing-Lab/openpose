#ifndef OPENPOSE_FILESTREAM_KEYPOINT_JSON_SAVER_HPP
#define OPENPOSE_FILESTREAM_KEYPOINT_JSON_SAVER_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/fileSaver.hpp>

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
