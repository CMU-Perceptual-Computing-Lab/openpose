#include <openpose/filestream/cocoJsonSaver.hpp>
#include <numeric> // std::iota
#include <openpose/pose/poseParametersRender.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/utilities/string.hpp>

namespace op
{
    int getLastNumberWithErrorMessage(const std::string& imageName, const CocoJsonFormat cocoJsonFormat)
    {
        try
        {
            return (int)getLastNumber(imageName);
        }
        catch (const std::exception& e)
        {
            const std::string errorMessage = "`--write_coco_json` is to be used with the original "
                + std::string(cocoJsonFormat == CocoJsonFormat::Car ? "car" : "COCO")
                + " dataset images. If you are not"
                " applying those, OpenPose cannot obtain the ID from their file names. Image name:\n"
                + imageName + "\n Error details: "
                + e.what();
            error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    CocoJsonSaver::CocoJsonSaver(const std::string& filePathToSave, const PoseModel poseModel,
                                 const bool humanReadable, const int cocoJsonVariants,
                                 const CocoJsonFormat cocoJsonFormat, const int cocoJsonVariant) :
        mPoseModel{poseModel},
        mCocoJsonVariant{cocoJsonVariant}
    {
        try
        {
            // Sanity checks
            if (filePathToSave.empty())
                error("Empty path given as output file path for saving COCO JSON format.",
                      __LINE__, __FUNCTION__, __FILE__);
            if (cocoJsonVariants >= 32)
                error("Unkown value for cocoJsonFormat (flag `--write_coco_json_variants`).",
                      __LINE__, __FUNCTION__, __FILE__);
            // Open mJsonOfstreams
            const auto filePath = getFullFilePathNoExtension(filePathToSave);
            const auto extension = getFileExtension(filePathToSave);
            // Body/cars
            if (cocoJsonVariants % 2 == 1 || cocoJsonVariants < 1)
                mJsonOfstreams.emplace_back(
                    std::make_tuple(JsonOfstream{filePathToSave, humanReadable}, cocoJsonFormat, false));
            // Foot
            if ((cocoJsonVariants/2) % 2 == 1 || cocoJsonVariants < 1)
                mJsonOfstreams.emplace_back(
                    std::make_tuple(JsonOfstream{filePath+"_foot."+extension, humanReadable}, CocoJsonFormat::Foot,
                        false));
            // Face
            if ((cocoJsonVariants/4) % 2 == 1 || cocoJsonVariants < 1)
                mJsonOfstreams.emplace_back(
                    std::make_tuple(JsonOfstream{filePath+"_face."+extension, humanReadable}, CocoJsonFormat::Face,
                        false));
            // Hand21
            if ((cocoJsonVariants/8) % 2 == 1 || cocoJsonVariants < 1)
                mJsonOfstreams.emplace_back(
                    std::make_tuple(JsonOfstream{filePath+"_hand21."+extension, humanReadable}, CocoJsonFormat::Hand21,
                        false));
            // Hand42
            if ((cocoJsonVariants/16) % 2 == 1 || cocoJsonVariants < 1)
                mJsonOfstreams.emplace_back(
                    std::make_tuple(JsonOfstream{filePath+"_hand42."+extension, humanReadable}, CocoJsonFormat::Hand42,
                        false));
            // Open array
            for (auto& jsonOfstream : mJsonOfstreams)
                std::get<0>(jsonOfstream).arrayOpen();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    CocoJsonSaver::~CocoJsonSaver()
    {
        try
        {
            for (auto& jsonOfstream : mJsonOfstreams)
                std::get<0>(jsonOfstream).arrayClose();
        }
        catch (const std::exception& e)
        {
            errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void CocoJsonSaver::record(
        const Array<float>& poseKeypoints, const Array<float>& poseScores, const std::string& imageName,
        const unsigned long long frameNumber)
    {
        try
        {
            // Sanity check
            if ((size_t)poseKeypoints.getSize(0) != poseScores.getVolume())
                error("Dimension mismatch between poseKeypoints and poseScores.", __LINE__, __FUNCTION__, __FILE__);
            // Fixed variables
            const auto numberPeople = poseKeypoints.getSize(0);
            if (numberPeople > 0)
            {
                const auto numberBodyParts = poseKeypoints.getSize(1);
                // Iterate over all JsonOfstreams
                for (auto& jsonOfstreamAndFormat : mJsonOfstreams)
                {
                    auto& jsonOfstream = std::get<0>(jsonOfstreamAndFormat);
                    const auto cocoJsonFormat = std::get<1>(jsonOfstreamAndFormat);
                    auto& firstElementAdded = std::get<2>(jsonOfstreamAndFormat);
                    // Get indexesInCocoOrder
                    std::vector<int> indexesInCocoOrder;
                    // Body/car
                    auto imageId = frameNumber;
                    if (cocoJsonFormat == CocoJsonFormat::Body)
                    {
                        imageId = getLastNumberWithErrorMessage(imageName, CocoJsonFormat::Body);
                        // Body
                        if (numberBodyParts == 23)
                            indexesInCocoOrder = std::vector<int>{
                                0, 14,13,16,15,    4,1,5,2,6,    3,10,7,11, 8,    12, 9};
                        else if (numberBodyParts == 18)
                            indexesInCocoOrder = std::vector<int>{
                                0, 15,14,17,16,    5,2,6,3,7,    4,11,8,12, 9,    13,10};
                        else if (mPoseModel == PoseModel::BODY_25B || mPoseModel == PoseModel::BODY_135)
                        {
                            indexesInCocoOrder = std::vector<int>(17);
                            std::iota(indexesInCocoOrder.begin(), indexesInCocoOrder.end(), 0);
                        }
                        else if (numberBodyParts == 19 || numberBodyParts == 25 || numberBodyParts == 59)
                            indexesInCocoOrder = std::vector<int>{
                                0, 16,15,18,17,    5,2,6,3,7,    4,12,9,13,10,    14,11};
                        // else if (numberBodyParts == 23)
                        //     indexesInCocoOrder = std::vector<int>{
                        //         18,21,19,22,20,    4,1,5,2,6,    3,13,8,14, 9,    15,10};
                    }
                    // Foot
                    else if (cocoJsonFormat == CocoJsonFormat::Foot)
                    {
                        imageId = getLastNumberWithErrorMessage(imageName, CocoJsonFormat::Foot);
                        if (numberBodyParts == 25 || numberBodyParts > 60)
                            indexesInCocoOrder = std::vector<int>{19,20,21, 22,23,24};
                        else if (numberBodyParts == 23)
                            indexesInCocoOrder = std::vector<int>{17,18,19, 20,21,22};
                    }
                    // Face
                    else if (cocoJsonFormat == CocoJsonFormat::Face)
                    {
                        if (numberBodyParts == 135)
                        {
                            indexesInCocoOrder = std::vector<int>(68);
                            std::iota(indexesInCocoOrder.begin(), indexesInCocoOrder.end(), F135);
                        }
                    }
                    // Hand21
                    else if (cocoJsonFormat == CocoJsonFormat::Hand21)
                    {
                        if (numberBodyParts == 135)
                        {
                            indexesInCocoOrder = std::vector<int>(21);
                            indexesInCocoOrder[0] = 10;
                            std::iota(indexesInCocoOrder.begin()+1, indexesInCocoOrder.end(), H135+20);
                        }
                    }
                    // Hand42
                    else if (cocoJsonFormat == CocoJsonFormat::Hand42)
                    {
                        if (numberBodyParts == 135)
                        {
                            indexesInCocoOrder = std::vector<int>(42);
                            indexesInCocoOrder[0] = 9;
                            std::iota(indexesInCocoOrder.begin()+1, indexesInCocoOrder.end(), H135);
                            indexesInCocoOrder[21] = 10;
                            std::iota(indexesInCocoOrder.begin()+22, indexesInCocoOrder.end(), H135+20);
                        }
                    }
                    // Car
                    else if (cocoJsonFormat == CocoJsonFormat::Car)
                    {
                        imageId = getLastNumberWithErrorMessage(imageName, CocoJsonFormat::Car);
                        // Car12
                        if (numberBodyParts == 12)
                            indexesInCocoOrder = std::vector<int>{0,1,2,3, 4,5,6,7, 8, 8,9,10,11, 11};
                        // Car22
                        else if (numberBodyParts == 22)
                        {
                            // Dataset 1
                            if (mCocoJsonVariant == 0)
                                indexesInCocoOrder = std::vector<int>{0,1,2,3, 6,7, 12,13,14,15, 16,17};
                            // Dataset 2
                            else if (mCocoJsonVariant == 1)
                                indexesInCocoOrder = std::vector<int>{0,1,2,3, 6,7, 12,13,14,15, 20,21};
                            // Dataset 3
                            else if (mCocoJsonVariant == 2)
                                for (auto i = 0 ; i < 20 ; i++)
                                    indexesInCocoOrder.emplace_back(i);
                        }
                    }
                    // Sanity check
                    if (indexesInCocoOrder.empty())
                        error("Invalid number of body parts (" + std::to_string(numberBodyParts) + ").",
                              __LINE__, __FUNCTION__, __FILE__);
                    // Save on JSON file
                    for (auto person = 0 ; person < numberPeople ; person++)
                    {
                        // At least 1 valid keypoint?
                        // Reason: When saving any combination of Body + Foot + Face + Hand, the others might be empty
                        bool foundAtLeast1Keypoint = false;
                        for (auto bodyPart = 0u ; bodyPart < indexesInCocoOrder.size() ; bodyPart++)
                        {
                            const auto finalIndex = 3*(person*numberBodyParts + indexesInCocoOrder.at(bodyPart));
                            const auto validPoint = (poseKeypoints[finalIndex+2] > 0.f);
                            if (validPoint)
                            {
                                foundAtLeast1Keypoint = true;
                                break;
                            }
                        }

                        if (foundAtLeast1Keypoint)
                        {
                            // Comma at any moment but first element
                            if (firstElementAdded)
                            {
                                jsonOfstream.comma();
                                jsonOfstream.enter();
                            }
                            else
                                firstElementAdded = true;

                            // New element
                            jsonOfstream.objectOpen();

                            // image_id
                            jsonOfstream.key("image_id");
                            jsonOfstream.plainText(imageId);
                            jsonOfstream.comma();

                            // category_id
                            jsonOfstream.key("category_id");
                            jsonOfstream.plainText("1");
                            jsonOfstream.comma();

                            // keypoints - i.e., poseKeypoints
                            jsonOfstream.key("keypoints");
                            jsonOfstream.arrayOpen();
                            for (auto bodyPart = 0u ; bodyPart < indexesInCocoOrder.size() ; bodyPart++)
                            {
                                const auto finalIndex = 3*(person*numberBodyParts + indexesInCocoOrder.at(bodyPart));
                                const auto validPoint = (poseKeypoints[finalIndex+2] > 0.f);
                                jsonOfstream.plainText(validPoint ? poseKeypoints[finalIndex] : -1.f);
                                jsonOfstream.comma();
                                jsonOfstream.plainText(validPoint ? poseKeypoints[finalIndex+1] : -1.f);
                                jsonOfstream.comma();
                                jsonOfstream.plainText(validPoint ? 1 : 0);
                                // jsonOfstream.plainText(poseKeypoints[finalIndex+2]); // For debugging
                                if (bodyPart < indexesInCocoOrder.size() - 1u)
                                    jsonOfstream.comma();
                            }
                            jsonOfstream.arrayClose();
                            jsonOfstream.comma();

                            // score
                            jsonOfstream.key("score");
                            jsonOfstream.plainText(poseScores[person]);

                            jsonOfstream.objectClose();
                        }
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
