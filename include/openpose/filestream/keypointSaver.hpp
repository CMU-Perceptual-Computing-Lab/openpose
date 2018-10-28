#ifndef OPENPOSE_FILESTREAM_KEYPOINT_SAVER_HPP
#define OPENPOSE_FILESTREAM_KEYPOINT_SAVER_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/enumClasses.hpp>
#include <openpose/filestream/fileSaver.hpp>

namespace op
{
    class OP_API KeypointSaver : public FileSaver
    {
    public:
        KeypointSaver(const std::string& directoryPath, const DataFormat format);

        virtual ~KeypointSaver();

        void saveKeypoints(const std::vector<Array<float>>& keypointVector, const std::string& fileName,
                           const std::string& keypointName) const;

    private:
        const DataFormat mFormat;
    };
}

#endif // OPENPOSE_FILESTREAM_KEYPOINT_SAVER_HPP
