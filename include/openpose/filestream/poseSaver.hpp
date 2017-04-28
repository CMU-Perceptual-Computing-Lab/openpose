#ifndef OPENPOSE__FILESTREAM__POSE_SAVER_HPP
#define OPENPOSE__FILESTREAM__POSE_SAVER_HPP

#include "../core/array.hpp"
#include "enumClasses.hpp"
#include "fileSaver.hpp"

namespace op
{
    class PoseSaver : public FileSaver
    {
    public:
        PoseSaver(const std::string& directoryPath, const DataFormat format);

        void savePoseKeyPoints(const std::vector<Array<float>>& poseKeyPointsVector, const std::string& fileName) const;

    private:
        const DataFormat mFormat;
    };
}

#endif // OPENPOSE__FILESTREAM__POSE_SAVER_HPP
