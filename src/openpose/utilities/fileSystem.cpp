#include <cstdio> // fopen
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <openpose/utilities/string.hpp>
#include <openpose/utilities/fileSystem.hpp>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <dirent.h> //opendir
#include <cstring>
#include <memory>
#include <unistd.h>

// The only remaining boost dependency part
// We could use the mkdir method to create dir
// in unix or windows

//#if defined _MSC_VER
//#include <direct.h>
//#elif defined __GNUC__
//#include <sys/types.h>
//#include <sys/stat.h>
//#endif

namespace op
{
    void mkdir(const std::string& directoryPath)
    {
        try
        {
            if (!directoryPath.empty())
            {

                //#if defined _MSC_VER
                //_mkdir(directoryPath.c_str());
                //#elif defined __GNUC__
                //mkdir(directoryPath.data(), S_IRUSR | S_IWUSR | S_IXUSR);
                //#endif

                // Create folder if it does not exist
                const boost::filesystem::path directory{directoryPath};
                if (!isDirectory(directoryPath) && !boost::filesystem::create_directory(directory))
                    error("Could not write to or create directory to save processed frames.",
                          __LINE__, __FUNCTION__, __FILE__);
            };
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    bool existDir(const std::string& directoryPath)
    {
        try
        {
            // Maybe existFile also works for directories in Ubuntu/Windows/Mac?
            DIR* dir = opendir(directoryPath.c_str());

            if (dir != NULL)
            {
                closedir(dir);
                return true;
            }
            else
            {
                return false;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    bool existFile(const std::string& filePath)
    {
        try
        {
            if (auto* file = fopen(filePath.c_str(), "r"))
            {
                fclose(file);
                return true;
            }
            else
                return false;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    // What's the difference with existDir
    bool isDirectory(const std::string& directoryPath)
    {
        try
        {
            DIR* dir = opendir(directoryPath.c_str());

            if (dir != NULL)
            {
                closedir(dir);
                return true;
            }
            else
            {
                return false;
            }
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
            size_t sep = fullPath.find_last_of("\\/");
            if (sep != std::string::npos)
                return fullPath.substr(sep+1, fullPath.size() - sep - 1);
            else
                return fullPath;
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
            std::string nameExt = getFileNameAndExtension(fullPath);
            size_t dot = nameExt.find_last_of(".");
            if (dot != std::string::npos)
                return nameExt.substr(0, dot);
            else
                return nameExt;
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
            std::string nameExt = getFileNameAndExtension(fullPath);

            size_t dot = nameExt.find_last_of(".");
            if (dot != std::string::npos)
                return nameExt.substr(dot+1, nameExt.size()-dot-1);
            else
                return "";
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

    std::vector<std::string> getFilesOnDirectory(const std::string& directoryPath,
                                                 const std::vector<std::string>& extensions)
    {
        try
        {
            // Format the path first
            std::string format_path = formatAsDirectory(directoryPath);
            // Check folder exits
            if (!existDir(directoryPath))
                error("Folder " + directoryPath + " does not exist.", __LINE__, __FUNCTION__, __FILE__);
            // Read images
            std::vector<std::string> filePaths;
            std::shared_ptr<DIR> directory_ptr(opendir(format_path.c_str()), [](DIR* format_path)
					       { format_path && closedir(format_path); });
            struct dirent *dirent_ptr;

            while ((dirent_ptr = readdir(directory_ptr.get())) != nullptr)
            {
                std::string tmp_path = format_path + dirent_ptr->d_name;
                if ((strncmp(dirent_ptr->d_name, ".", 1) ==0)||(isDirectory(tmp_path) == 1))
                        continue;
                filePaths.push_back(tmp_path);
            }

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
