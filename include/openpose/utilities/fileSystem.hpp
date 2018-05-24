#ifndef OPENPOSE_UTILITIES_FILE_SYSTEM_HPP
#define OPENPOSE_UTILITIES_FILE_SYSTEM_HPP

#include <openpose/core/common.hpp>

namespace op
{
    OP_API void makeDirectory(const std::string& directoryPath);

    OP_API bool existDirectory(const std::string& directoryPath);

    OP_API bool existFile(const std::string& filePath);

    /**
     * This function makes sure that the directoryPathString is properly formatted. I.e., it
     * changes all '\' by '/', and it makes sure that the string finishes with '/'.
     * @param directoryPathString std::string with the directory path to be formatted.
     * @return std::string with the formatted directory path.
     */
    OP_API std::string formatAsDirectory(const std::string& directoryPathString);

    /**
     * This function extracts the file name and extension from a full path.
     * @param fullPath std::string with the full path.
     * @return std::string with the file name with extension.
     */
    OP_API std::string getFileNameAndExtension(const std::string& fullPath);

    /**
     * This function extracts the file name (without extension) from a full path.
     * @param fullPath std::string with the full path.
     * @return std::string with the file name without extension.
     */
    OP_API std::string getFileNameNoExtension(const std::string& fullPath);

    /**
     * This function extracts the extension from a full path.
     * @param fullPath std::string with the full path.
     * @return std::string with the file extension.
     */
    OP_API std::string getFileExtension(const std::string& fullPath);

    /**
     * This function extracts the full file path without its extension from a full file path.
     * @param fullPath std::string with the full path.
     * @return std::string with the full file path without extension.
     */
    OP_API std::string getFullFilePathNoExtension(const std::string& fullPath);

    /**
     * This function extracts the full file path of the folder where it is contained.
     * @param fullPath std::string with the full path.
     * @return std::string with the full file path of the folder.
     */
    OP_API std::string getFileParentFolderPath(const std::string& fullPath);

    /**
     * This function extracts all the files in a directory path with the desired
     * extensions. If no extensions is specified, then all the file names are returned.
     * @param directoryPath std::string with the directory path.
     * @param extensions std::vector<std::string> with the extensions of the desired files.
     * @return std::vector<std::string> with the existing file names.
     */
    OP_API std::vector<std::string> getFilesOnDirectory(const std::string& directoryPath,
                                                        const std::vector<std::string>& extensions = {});

    /**
     * Analogous to getFilesOnDirectory(const std::string& directoryPath, const std::vector<std::string>& extensions)
     * for 1 specific extension.
     * @param directoryPath std::string with the directory path.
     * @param extension std::string with the extension of the desired files.
     * @return std::vector<std::string> with the existing file names.
     */
    OP_API std::vector<std::string> getFilesOnDirectory(const std::string& directoryPath,
                                                        const std::string& extension);

    OP_API std::string removeSpecialsCharacters(const std::string& stringToVariate);

    OP_API void removeAllOcurrencesOfSubString(std::string& stringToModify, const std::string& substring);

    OP_API void replaceAll(std::string& stringText, const char charToChange, const char charToAdd);
}

#endif // OPENPOSE_UTILITIES_FILE_SYSTEM_HPP
