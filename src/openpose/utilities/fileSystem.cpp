#include <openpose/utilities/fileSystem.hpp>
#include <algorithm> // std::replace
#include <cctype> // std::isdigit
#include <cstdio> // std::fopen
#include <cstring> // std::strncmp
#ifdef _WIN32
    #include <direct.h> // _mkdir
    #include <windows.h> // DWORD, GetFileAttributesA
#elif defined __unix__ || defined __APPLE__
    #include <dirent.h> // opendir
    #include <sys/stat.h> // mkdir
#else
    #error Unknown environment!
#endif
#include <openpose/utilities/string.hpp>

namespace op
{
    bool compareNat(const std::string& a, const std::string& b)
    {
        // If any is empty, that one is shorter
        if (a.empty())
            return true;
        else if (b.empty())
            return false;
        // If one is digit, return that one
        else if (std::isdigit(a[0]) && !std::isdigit(b[0]))
            return true;
        else if (!std::isdigit(a[0]) && std::isdigit(b[0]))
            return false;
        // If none is digit, compare those strings
        else if (!std::isdigit(a[0]) && !std::isdigit(b[0]))
        {
            if (std::toupper(a[0]) == std::toupper(b[0]))
                return compareNat(a.substr(1), b.substr(1));
            return (std::toupper(a[0]) < std::toupper(b[0]));
        }
        // Both strings begin with digit
        else
        {
            // Remove 0s at the beginning
            const std::string aNo0s = remove0sFromString(a);
            const std::string bNo0s = remove0sFromString(b);
            // Get only number
            const std::string aNumberAsString = getFirstNumberOnString(aNo0s);
            const std::string bNumberAsString = getFirstNumberOnString(bNo0s);
            // 1 number is longer --> it is bigger
            if (aNumberAsString.size() != bNumberAsString.size())
                return aNumberAsString.size() < bNumberAsString.size();
            // Both same length --> Compare them
            for (auto i = 0 ; i < aNumberAsString.size() ; ++i)
                if (aNumberAsString[i] != bNumberAsString[i])
                    return aNumberAsString[i] < bNumberAsString[i];
            // Both same number --> Remove and return compareNat after removing those characters
            const std::string aWithoutFirstNumber = aNo0s.substr(aNumberAsString.size(), aNo0s.size());
            const std::string bWithoutFirstNumber = bNo0s.substr(bNumberAsString.size(), bNo0s.size());
            // Number removed, compared the rest of the string
            return compareNat(aWithoutFirstNumber, bWithoutFirstNumber);
        }
    }

    void makeDirectory(const std::string& directoryPath)
    {
        try
        {
            if (!directoryPath.empty())
            {
                // Format the path first
                const auto formatedPath = formatAsDirectory(directoryPath);
                // Create dir if it doesn't exist yet
                if (!existDirectory(formatedPath))
                {
                    #ifdef _WIN32
                        const auto status = _mkdir(formatedPath.c_str());
                    #elif defined __unix__ || defined __APPLE__
                        // Create folder
                        // Access permission - 775 (7, 7, 4+1)
                        // https://www.gnu.org/software/libc/manual/html_node/Permission-Bits.html
                        const auto status = mkdir(formatedPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                    #endif
                    // Error if folder cannot be created
                    if (status != 0)
                        error("Could not create directory: " + formatedPath + ". Status error = "
                              + std::to_string(status) + ". Does the parent folder exist and/or do you have writting"
                              " access to that path?", __LINE__, __FUNCTION__, __FILE__);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    bool existDirectory(const std::string& directoryPath)
    {
        try
        {
            // Format the path first
            const auto formatedPath = formatAsDirectory(directoryPath);
            #ifdef _WIN32
                DWORD status = GetFileAttributesA(formatedPath.c_str());
                // It is not a directory
                if (status == INVALID_FILE_ATTRIBUTES)
                    return false;
                // It is a directory
                else if (status & FILE_ATTRIBUTE_DIRECTORY)
                    return true;
                // It is not a directory
                return false;    // this is not a directory!
            #elif defined __unix__ || defined __APPLE__
                // It is a directory
                if (auto* directory = opendir(formatedPath.c_str()))
                {
                    closedir(directory);
                    return true;
                }
                // It is not a directory
                else
                    return false;
            #else
                #error Unknown environment!
            #endif
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

    std::string formatAsDirectory(const std::string& directoryPathString)
    {
        try
        {
            std::string directoryPath = directoryPathString;
            if (!directoryPath.empty())
            {
                // Replace all '\\' to '/'
                std::replace(directoryPath.begin(), directoryPath.end(), '\\', '/');
                if (directoryPath.back() != '/')
                    directoryPath = directoryPath + "/";
                // Windows - Replace all '/' to '\\'
                #ifdef _WIN32
                    std::replace(directoryPath.begin(), directoryPath.end(), '/', '\\');
                #endif
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
            size_t lastSlashPos = fullPath.find_last_of("\\/");
            if (lastSlashPos != std::string::npos)
                return fullPath.substr(lastSlashPos+1, fullPath.size() - lastSlashPos - 1);
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
            // Name + extension
            const std::string nameExt = getFileNameAndExtension(fullPath);
            // Name
            return getFullFilePathNoExtension(nameExt);
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
            // Name + extension
            const std::string nameExt = getFileNameAndExtension(fullPath);
            // Extension
            const size_t dotPos = nameExt.find_last_of(".");
            if (dotPos != std::string::npos)
                return nameExt.substr(dotPos + 1, nameExt.size() - dotPos - 1);
            else
                return "";
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    std::string getFullFilePathNoExtension(const std::string& fullPath)
    {
        try
        {
            // Name
            const size_t dotPos = fullPath.find_last_of(".");
            if (dotPos != std::string::npos)
                return fullPath.substr(0, dotPos);
            else
                return fullPath;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    std::string getFileParentFolderPath(const std::string& fullPath)
    {
        try
        {
            if (fullPath.size() > 0)
            {
                // Clean string
                std::string fullPathAux = fullPath;
                if (fullPathAux.at(fullPathAux.size() - 1) == '/'
                        || fullPathAux.at(fullPathAux.size() - 1) == '\\')
                    fullPathAux = {fullPathAux.substr(0, fullPathAux.size() - 1)};
                // Find last `/` (Unix) or `\` (Windows)
                const std::size_t posFound{fullPathAux.find_last_of("/\\")};
                // Return substring
                if (posFound != std::string::npos)
                    return fullPathAux.substr(0, posFound+1);
                else
                    return fullPathAux;
            }
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
            const auto formatedPath = formatAsDirectory(directoryPath);
            // Check folder exits
            if (!existDirectory(formatedPath))
                error("Folder " + formatedPath + " does not exist.", __LINE__, __FUNCTION__, __FILE__);
            // Read all files in folder
            std::vector<std::string> filePaths;
            #ifdef _WIN32
                auto formatedPathWindows = formatedPath;
                formatedPathWindows.append("\\*");
                WIN32_FIND_DATA data;
                HANDLE hFind;
                if ((hFind = FindFirstFile(formatedPathWindows.c_str(), &data)) != INVALID_HANDLE_VALUE)
                {
                    do
                        filePaths.emplace_back(formatedPath + data.cFileName);
                    while (FindNextFile(hFind, &data) != 0);
                    FindClose(hFind);
                }
            #elif defined __unix__ || defined __APPLE__
                std::shared_ptr<DIR> directoryPtr(
                    opendir(formatedPath.c_str()),
                    [](DIR* formatedPath){ formatedPath && closedir(formatedPath); }
                );
                struct dirent* direntPtr;
                while ((direntPtr = readdir(directoryPtr.get())) != nullptr)
                {
                    std::string currentPath = formatedPath + direntPtr->d_name;
                    if ((strncmp(direntPtr->d_name, ".", 1) == 0) || existDirectory(currentPath))
                            continue;
                    filePaths.emplace_back(currentPath);
                }
            #else
                #error Unknown environment!
            #endif
            // Check #files > 0
            if (filePaths.empty())
                error("No files were found on " + formatedPath, __LINE__, __FUNCTION__, __FILE__);
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
            // // Sort alphabetically
            // std::sort(filePaths.begin(), filePaths.end());
            // Natural sort
            std::sort(filePaths.begin(), filePaths.end(), compareNat);
            // Return result
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

    std::vector<std::string> getFilesOnDirectory(const std::string& directoryPath, const Extensions extensions)
    {
        try
        {
            // Get files on directory with the desired extensions
            if (extensions == Extensions::Images)
            {
                const std::vector<std::string> extensionNames{
                    // Completely supported by OpenCV
                    "bmp", "dib", "pbm", "pgm", "ppm", "sr", "ras",
                    // Most of them supported by OpenCV
                    "jpg", "jpeg", "png"};
                return getFilesOnDirectory(directoryPath, extensionNames);
            }
            // Unknown kind of extensions
            else
            {
                error("Unknown kind of extensions (id = " + std::to_string(int(extensions))
                      + "). Notify us of this error.", __LINE__, __FUNCTION__, __FILE__);
                return {};
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::string removeSpecialsCharacters(const std::string& stringToVariate)
    {
        try
        {
            auto result(stringToVariate);

            auto i = 0u;
            auto len = result.length();
            while (i < len)
            {
                const char c=result.at(i);
                if (((c>='0')&&(c<='9'))||((c>='A')&&(c<='Z'))||((c>='a')&&(c<='z')))
                {
                    // Assuming dictionary contains small letters only.
                    if ((c>='A')&&(c<='Z')) result.at(i) += 32;
                        ++i;
                }
                else
                {
                    result.erase(i,1);
                    --len;
                }
            }

            return result;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    void removeAllOcurrencesOfSubString(std::string& stringToModify, const std::string& substring)
    {
        try
        {
            auto pos(stringToModify.find(substring));
            while (pos != std::string::npos)
            {
                stringToModify.erase(pos, substring.size());
                pos = {stringToModify.find(substring)};
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void replaceAll(std::string& stringText, const char charToChange, const char charToAdd)
    {
        try
        {
            // replace all charToChange to charToAdd
            std::replace( stringText.begin(), stringText.end(), charToChange, charToAdd);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
