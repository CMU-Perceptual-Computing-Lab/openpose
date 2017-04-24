#ifndef OPENPOSE__FILESTREAM__DATA_SAVER_HPP
#define OPENPOSE__FILESTREAM__DATA_SAVER_HPP

#include <string>
#include "../utilities/string.hpp"

namespace op
{
    class FileSaver
    {
    protected:
        explicit FileSaver(const std::string& directoryPath);

        std::string getNextFileName(const unsigned long long index) const;

        std::string getNextFileName(const std::string& fileNameNoExtension) const;

    private:
        const std::string mDirectoryPath;
    };
}

#endif // OPENPOSE__FILESTREAM__DATA_SAVER_HPP
