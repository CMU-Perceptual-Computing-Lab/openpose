#ifndef OPENPOSE__FILESTREAM__HEAT_MAP_SAVER_HPP
#define OPENPOSE__FILESTREAM__HEAT_MAP_SAVER_HPP

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
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

#endif // OPENPOSE__FILESTREAM__HEAT_MAP_SAVER_HPP
