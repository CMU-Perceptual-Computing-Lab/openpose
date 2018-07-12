#include <openpose/filestream/fileStream.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/hand/handDetectorFromTxt.hpp>
 
namespace op
{
    std::vector<std::string> getTxtPathsOnDirectory(const std::string& txtDirectoryPath)
    {
        try
        {
            // Get files on directory with JSON extension
            const auto txtPaths = getFilesOnDirectory(txtDirectoryPath, ".txt");
            // Check #files > 0
            if (txtPaths.empty())
                error("No txt files were found on " + txtDirectoryPath, __LINE__, __FUNCTION__, __FILE__);
            // Return file names
            return txtPaths;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    HandDetectorFromTxt::HandDetectorFromTxt(const std::string& txtDirectoryPath) :
        mTxtDirectoryPath{txtDirectoryPath},
        mFilePaths{getTxtPathsOnDirectory(txtDirectoryPath)},
        mFrameNameCounter{0}
    {
    }

    std::vector<std::array<Rectangle<float>, 2>> HandDetectorFromTxt::detectHands()
    {
        try
        {
            return loadHandDetectorTxt(mFilePaths.at(mFrameNameCounter++));
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::vector<std::array<Rectangle<float>, 2>>{};
        }
    }
}
