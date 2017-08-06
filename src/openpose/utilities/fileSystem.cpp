#include <boost/filesystem.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/utilities/string.hpp>

namespace op
{
    void mkdir(const std::string& directoryPath)
    {
        try
        {
            if (!directoryPath.empty())
            {
                // Create folder if it does not exist
                const boost::filesystem::path directory{directoryPath};
                if (!boost::filesystem::is_directory(directory) && !boost::filesystem::create_directory(directory))
                    error("Could not write to or create directory to save processed frames.", __LINE__, __FUNCTION__, __FILE__);
            };
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    bool exist(const std::string& directoryPath)
    {
        try
        {
            return boost::filesystem::exists(directoryPath);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    bool isDirectory(const std::string& directoryPath)
    {
        try
        {
            return (!directoryPath.empty() && boost::filesystem::is_directory(directoryPath));
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    std::string formatAsDirectory(const std::string& directoryPathString)
    {
        try
        {
            std::string directoryPath = directoryPathString;
            if (!directoryPath.empty())
            {
                std::replace(directoryPath.begin(), directoryPath.end(), '\\', '/'); // replace all '\\' to '/';
                if (directoryPath.back() != '/')
                    directoryPath = directoryPath + "/";
            }
            return directoryPath;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    std::string getFileNameAndExtension(const std::string& fullPath)
    {
        try
        {
            return boost::filesystem::path{fullPath}.filename().string();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    std::string getFileNameNoExtension(const std::string& fullPath)
    {
        try
        {
            return boost::filesystem::path{fullPath}.stem().string();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    std::string getFileExtension(const std::string& fullPath)
    {
        try
        {
            return boost::filesystem::path{fullPath}.extension().string();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    // This function just removes the initial '.' in the std::string (if any)
    // To avoid errors for not finding extensions because of comparing ".jpg" vs "jpg"
    std::string removeExtensionDot(const std::string& extension)
    {
        try
        {
            // Extension is empty
            if (extension.empty())
                return "";
            // Return string without initial character
            else if (*extension.cbegin() == '.')
                return extension.substr(1, extension.size() - 1);
            // Return string itself
            else
                return extension;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    bool extensionIsDesired(const std::string& extension, const std::vector<std::string>& extensions)
    {
        try
        {
            const auto cleanedExtension = toLower(removeExtensionDot(extension));
            for (auto& extensionI : extensions)
                if (cleanedExtension == toLower(removeExtensionDot(extensionI)))
                    return true;
            return false;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    std::vector<std::string> getFilesOnDirectory(const std::string& directoryPath, const std::vector<std::string>& extensions)
    {
        try
        {
            // Check folder exits
            if (!exist(directoryPath))
                error("Folder " + directoryPath + " does not exist.", __LINE__, __FUNCTION__, __FILE__);
            // Read images
            std::vector<std::string> filePaths;
            for (auto& file : boost::make_iterator_range(boost::filesystem::directory_iterator{directoryPath}, {}))
                if (!boost::filesystem::is_directory(file.status()))                // Skip directories
                    filePaths.emplace_back(file.path().string());
            // Check #files > 0
            if (filePaths.empty())
                error("No files were found on " + directoryPath, __LINE__, __FUNCTION__, __FILE__);

            // If specific extensions specified
            if (!extensions.empty())
            {
                // Read images
                std::vector<std::string> specificExtensionPaths;
                specificExtensionPaths.reserve(filePaths.size());
                for (const auto& filePath : filePaths)
                    if (extensionIsDesired(getFileExtension(filePath), extensions))
                        specificExtensionPaths.emplace_back(filePath);
                std::swap(filePaths, specificExtensionPaths);
            }

            // Sort alphabetically
            std::sort(filePaths.begin(), filePaths.end());

            return filePaths;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<std::string> getFilesOnDirectory(const std::string& directoryPath, const std::string& extension)
    {
        try
        {
            return getFilesOnDirectory(directoryPath, std::vector<std::string>{extension});
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }
}
