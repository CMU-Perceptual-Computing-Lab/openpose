#ifndef OPENPOSE_FILESTREAM_KEYPOINT_SAVER_HPP
#define OPENPOSE_FILESTREAM_KEYPOINT_SAVER_HPP

#include <openpose/core/array.hpp>
#include "enumClasses.hpp"
#include "fileSaver.hpp"

namespace op
{
    class KeypointSaver : public FileSaver
    {
    public:
        KeypointSaver(const std::string& directoryPath, const DataFormat format);

        void saveKeypoints(const std::vector<Array<float>>& keypointVector, const std::string& fileName, const std::string& keypointName) const;

    private:
        const DataFormat mFormat;
    };
}

#endif // OPENPOSE_FILESTREAM_KEYPOINT_SAVER_HPP
