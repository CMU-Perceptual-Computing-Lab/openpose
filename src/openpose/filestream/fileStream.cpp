#include <fstream> // std::ifstream, std::ofstream
#include <opencv2/highgui/highgui.hpp> // cv::imread
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/utilities/string.hpp>
#include <openpose/filestream/jsonOfstream.hpp>
#include <openpose/filestream/fileStream.hpp>

namespace op
{
    // Private class (on *.cpp)
    const auto errorMessage = "Json format only implemented in OpenCV for versions >= 3.0. Check savePoseJson"
                              " instead.";

    std::string getFullName(const std::string& fileNameNoExtension, const DataFormat dataFormat)
    {
        return fileNameNoExtension + "." + dataFormatToString(dataFormat);
    }

    void addKeypointsToJson(
        JsonOfstream& jsonOfstream, const std::vector<std::pair<Array<float>, std::string>>& keypointVector)
    {
        try
        {
            // Sanity check
            for (const auto& keypointPair : keypointVector)
                if (!keypointPair.first.empty() && keypointPair.first.getNumberDimensions() != 3
                    && keypointPair.first.getNumberDimensions() != 1)
                    error("keypointVector.getNumberDimensions() != 1 && != 3.", __LINE__, __FUNCTION__, __FILE__);
            // Add people keypoints
            jsonOfstream.key("people");
            jsonOfstream.arrayOpen();
            // Ger max numberPeople
            auto numberPeople = 0;
            for (auto vectorIndex = 0u ; vectorIndex < keypointVector.size() ; vectorIndex++)
                numberPeople = fastMax(numberPeople, keypointVector[vectorIndex].first.getSize(0));
            for (auto person = 0 ; person < numberPeople ; person++)
            {
                jsonOfstream.objectOpen();
                for (auto vectorIndex = 0u ; vectorIndex < keypointVector.size() ; vectorIndex++)
                {
                    const auto& keypoints = keypointVector[vectorIndex].first;
                    const auto& keypointName = keypointVector[vectorIndex].second;
                    const auto numberElementsPerRaw = keypoints.getSize(1) * keypoints.getSize(2);
                    jsonOfstream.key(keypointName);
                    jsonOfstream.arrayOpen();
                    // Body parts
                    if (numberElementsPerRaw > 0)
                    {
                        const auto finalIndex = person*numberElementsPerRaw;
                        for (auto element = 0 ; element < numberElementsPerRaw - 1 ; element++)
                        {
                            jsonOfstream.plainText(keypoints[finalIndex + element]);
                            jsonOfstream.comma();
                        }
                        // Last element (no comma)
                        jsonOfstream.plainText(keypoints[finalIndex + numberElementsPerRaw - 1]);
                    }
                    // Close array
                    jsonOfstream.arrayClose();
                    if (vectorIndex < keypointVector.size()-1)
                        jsonOfstream.comma();
                }
                jsonOfstream.objectClose();
                if (person < numberPeople-1)
                {
                    jsonOfstream.comma();
                    jsonOfstream.enter();
                }
            }
            // Close bodies array
            jsonOfstream.arrayClose();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void addCandidatesToJson(
        JsonOfstream& jsonOfstream, const std::vector<std::vector<std::array<float,3>>>& candidates)
    {
        try
        {
            // Add body part candidates
            jsonOfstream.key("part_candidates");
            jsonOfstream.arrayOpen();
            // Ger max numberParts
            const auto numberParts = candidates.size();
            jsonOfstream.objectOpen();
            for (auto part = 0u ; part < numberParts ; part++)
            {
                // Open array
                jsonOfstream.key(std::to_string(part));
                jsonOfstream.arrayOpen();
                // Iterate over part candidates
                const auto& partCandidates = candidates[part];
                const auto numberPartCandidates = partCandidates.size();
                // Body part candidates
                for (auto bodyPart = 0u ; bodyPart < numberPartCandidates ; bodyPart++)
                {
                    const auto& candidate = partCandidates[bodyPart];
                    jsonOfstream.plainText(candidate[0]);
                    jsonOfstream.comma();
                    jsonOfstream.plainText(candidate[1]);
                    jsonOfstream.comma();
                    jsonOfstream.plainText(candidate[2]);
                    if (bodyPart < numberPartCandidates-1)
                        jsonOfstream.comma();
                }
                jsonOfstream.arrayClose();
                if (part < numberParts-1)
                    jsonOfstream.comma();
            }
            jsonOfstream.objectClose();
            // Close array
            jsonOfstream.arrayClose();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }





    // Public classes (on *.hpp)
    std::string dataFormatToString(const DataFormat dataFormat)
    {
        try
        {
            if (dataFormat == DataFormat::Json)
                return "json";
            else if (dataFormat == DataFormat::Xml)
                return "xml";
            else if (dataFormat == DataFormat::Yaml)
                return "yaml";
            else if (dataFormat == DataFormat::Yml)
                return "yml";
            else
            {
                error("Undefined DataFormat.", __LINE__, __FUNCTION__, __FILE__);
                return "";
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    DataFormat stringToDataFormat(const std::string& dataFormat)
    {
        try
        {
            if (dataFormat == "json")
                return DataFormat::Json;
            else if (dataFormat == "xml")
                return DataFormat::Xml;
            else if (dataFormat == "yaml")
                return DataFormat::Yaml;
            else if (dataFormat == "yml")
                return DataFormat::Yml;
            else
            {
                error("String does not correspond to any known format (json, xml, yaml, yml)",
                      __LINE__, __FUNCTION__, __FILE__);
                return DataFormat::Json;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return DataFormat::Json;
        }
    }

    void saveFloatArray(const Array<float>& array, const std::string& fullFilePath)
    {
        try
        {
            // Open file
            std::ofstream outputFile;
            outputFile.open(fullFilePath, std::ios::binary);
            // Save #dimensions
            const auto numberDimensions = (float)(array.getNumberDimensions());
            outputFile.write((char*)&numberDimensions, sizeof(float));
            // Save dimensions
            for (const auto& sizeI : array.getSize())
            {
                const float sizeIFloat = (float) sizeI;
                outputFile.write((char*)&sizeIFloat, sizeof(float));
            }
            // Save each value
            outputFile.write((char*)&array[0], array.getVolume() * sizeof(float));
            // Close file
            outputFile.close();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void saveData(const std::vector<cv::Mat>& cvMats, const std::vector<std::string>& cvMatNames,
                  const std::string& fileNameNoExtension, const DataFormat dataFormat)
    {
        try
        {
            // Sanity checks
            if (dataFormat == DataFormat::Json && CV_MAJOR_VERSION < 3)
                error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
            if (cvMats.size() != cvMatNames.size())
                error("cvMats.size() != cvMatNames.size() (" + std::to_string(cvMats.size())
                      + " vs. " + std::to_string(cvMatNames.size()) + ")", __LINE__, __FUNCTION__, __FILE__);
            // Save cv::Mat data
            cv::FileStorage fileStorage{getFullName(fileNameNoExtension, dataFormat), cv::FileStorage::WRITE};
            for (auto i = 0u ; i < cvMats.size() ; i++)
                fileStorage << cvMatNames[i] << (cvMats[i].empty() ? cv::Mat() : cvMats[i]);
            // Release file
            fileStorage.release();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void saveData(const cv::Mat& cvMat, const std::string cvMatName, const std::string& fileNameNoExtension,
                  const DataFormat dataFormat)
    {
        try
        {
            saveData(std::vector<cv::Mat>{cvMat}, std::vector<std::string>{cvMatName}, fileNameNoExtension,
                     dataFormat);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::vector<cv::Mat> loadData(const std::vector<std::string>& cvMatNames, const std::string& fileNameNoExtension,
                                  const DataFormat dataFormat)
    {
        try
        {
            // Sanity check
            if (dataFormat == DataFormat::Json && CV_MAJOR_VERSION < 3)
                error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
            // File name
            const auto fileName = getFullName(fileNameNoExtension, dataFormat);
            // Sanity check
            if (!existFile(fileName))
                error("File to be read does not exist: " + fileName + ".", __LINE__, __FUNCTION__, __FILE__);
            // Read file
            cv::FileStorage fileStorage{fileName, cv::FileStorage::READ};
            std::vector<cv::Mat> cvMats(cvMatNames.size());
            for (auto i = 0u ; i < cvMats.size() ; i++)
                fileStorage[cvMatNames[i]] >> cvMats[i];
            fileStorage.release();
            return cvMats;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    cv::Mat loadData(const std::string& cvMatName, const std::string& fileNameNoExtension, const DataFormat dataFormat)
    {
        try
        {
            return loadData(std::vector<std::string>{cvMatName}, fileNameNoExtension, dataFormat)[0];
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    void savePeopleJson(
        const Array<float>& keypoints, const std::vector<std::vector<std::array<float,3>>>& candidates,
        const std::string& keypointName, const std::string& fileName, const bool humanReadable)
    {
        try
        {
            savePeopleJson(
                std::vector<std::pair<Array<float>, std::string>>{std::make_pair(keypoints, keypointName)},
                candidates, fileName, humanReadable
            );
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void savePeopleJson(
        const std::vector<std::pair<Array<float>, std::string>>& keypointVector,
        const std::vector<std::vector<std::array<float,3>>>& candidates, const std::string& fileName,
        const bool humanReadable)
    {
        try
        {
            // Sanity check
            for (const auto& keypointPair : keypointVector)
                if (!keypointPair.first.empty() && keypointPair.first.getNumberDimensions() != 3
                    && keypointPair.first.getNumberDimensions() != 1)
                    error("keypointVector.getNumberDimensions() != 1 && != 3.", __LINE__, __FUNCTION__, __FILE__);
            // Record frame on desired path
            JsonOfstream jsonOfstream{fileName, humanReadable};
            jsonOfstream.objectOpen();
            // Add version
            // Version 0.1: Body keypoints (2-D)
            // Version 1.0: Added face and hands (2-D)
            // Version 1.1: Added candidates
            // Version 1.2: Added body, face, and hands (3-D)
            // Version 1.3: Added person ID (for temporal consistency)
            jsonOfstream.version("1.3");
            jsonOfstream.comma();
            // Add people keypoints
            addKeypointsToJson(jsonOfstream, keypointVector);
            // Add body part candidates
            if (!candidates.empty())
            {
                jsonOfstream.comma();
                addCandidatesToJson(jsonOfstream, candidates);
            }
            // Close object
            jsonOfstream.objectClose();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void saveImage(const cv::Mat& cvMat, const std::string& fullFilePath,
                   const std::vector<int>& openCvCompressionParams)
    {
        try
        {
            if (!cv::imwrite(fullFilePath, cvMat, openCvCompressionParams))
                error("Image could not be saved on " + fullFilePath + ".", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    cv::Mat loadImage(const std::string& fullFilePath, const int openCvFlags)
    {
        try
        {
            cv::Mat cvMat = cv::imread(fullFilePath, openCvFlags);
            if (cvMat.empty())
                log("Empty image on path: " + fullFilePath + ".", Priority::Max, __LINE__, __FUNCTION__, __FILE__);
            return cvMat;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return cv::Mat();
        }
    }

    std::vector<std::array<Rectangle<float>, 2>> loadHandDetectorTxt(const std::string& txtFilePath)
    {
        try
        {
            std::vector<std::array<Rectangle<float>, 2>> handRectangles;

            std::string line;
            std::ifstream jsonFile{txtFilePath};
            if (jsonFile.is_open())
            {
                while (std::getline(jsonFile, line))
                {
                    const auto splittedStrings = splitString(line, " ");
                    std::vector<float> splittedInts;
                    for (auto splittedString : splittedStrings)
                        splittedInts.emplace_back(std::stof(splittedString));
                    if (splittedInts.size() != 4u)
                        error("splittedInts.size() != 4, but splittedInts.size() = "
                              + std::to_string(splittedInts.size()) + ".", __LINE__, __FUNCTION__, __FILE__);
                    const Rectangle<float> handRectangleZero;
                    const Rectangle<float> handRectangle{splittedInts[0], splittedInts[1], splittedInts[2],
                                                         splittedInts[3]};
                    if (getFileNameNoExtension(txtFilePath).back() == 'l')
                        handRectangles.emplace_back(std::array<Rectangle<float>, 2>{handRectangle, handRectangleZero});
                    else
                        handRectangles.emplace_back(std::array<Rectangle<float>, 2>{handRectangleZero, handRectangle});
                }
                jsonFile.close();
            }
            else
                error("Unable to open file " + txtFilePath + ".", __LINE__, __FUNCTION__, __FILE__);

            return handRectangles;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }
}
