#ifndef OPENPOSE_FILESTREAM_HEAT_MAP_SAVER_HPP
#define OPENPOSE_FILESTREAM_HEAT_MAP_SAVER_HPP

#include <string>
#include <vector>
#include "fileSaver.hpp"

namespace op
{
    class HeatMapSaver : public FileSaver
    {
    public:
        HeatMapSaver(const std::string& directoryPath, const std::string& imageFormat);

        void saveHeatMaps(const std::vector<Array<float>>& heatMaps, const std::string& fileName) const;

    private:
        const std::string mImageFormat;
    };
}

#endif // OPENPOSE_FILESTREAM_HEAT_MAP_SAVER_HPP
