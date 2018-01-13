#ifndef OPENPOSE_FILESTREAM_PEOPLE_JSON_SAVER_HPP
#define OPENPOSE_FILESTREAM_PEOPLE_JSON_SAVER_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/fileSaver.hpp>

namespace op
{
    class OP_API PeopleJsonSaver : public FileSaver
    {
    public:
        PeopleJsonSaver(const std::string& directoryPath);

        void save(const std::vector<std::pair<Array<float>, std::string>>& keypointVector,
                  const std::vector<std::vector<std::array<float,3>>>& candidates,
                  const std::string& fileName, const bool humanReadable = true) const;
    };
}

#endif // OPENPOSE_FILESTREAM_PEOPLE_JSON_SAVER_HPP
