#include <openpose/utilities/fileSystem.hpp>
#include <openpose/filestream/fileSaver.hpp>

namespace op
{
    FileSaver::FileSaver(const std::string& directoryPath) :
        mDirectoryPath{formatAsDirectory(directoryPath)}
    {
        try
        {
            makeDirectory(mDirectoryPath);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::string FileSaver::getNextFileName(const unsigned long long index) const
    {
        try
        {
            const auto stringLength = 12u;
            return mDirectoryPath + toFixedLengthString(index, stringLength);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    std::string FileSaver::getNextFileName(const std::string& fileName) const
    {
        try
        {
            return mDirectoryPath + fileName;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }
}
