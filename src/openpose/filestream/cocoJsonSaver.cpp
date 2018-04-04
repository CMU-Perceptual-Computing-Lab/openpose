#include <openpose/utilities/string.hpp>
#include <openpose/filestream/cocoJsonSaver.hpp>

namespace op
{
    CocoJsonSaver::CocoJsonSaver(const std::string& filePathToSave, const bool humanReadable) :
        mJsonOfstream{filePathToSave, humanReadable},
        mFirstElementAdded{false}
    {
        try
        {
            mJsonOfstream.arrayOpen();
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
            mJsonOfstream.arrayClose();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void CocoJsonSaver::record(const Array<float>& poseKeypoints, const Array<float>& poseScores,
                               const std::string& imageName)
    {
        try
        {
            // Security checks
            if ((size_t)poseKeypoints.getSize(0) != poseScores.getVolume())
                error("Dimension mismatch between poseKeypoints and poseScores.", __LINE__, __FUNCTION__, __FILE__);
            const auto numberPeople = poseKeypoints.getSize(0);
            const auto numberBodyParts = poseKeypoints.getSize(1);
            const auto imageId = getLastNumber(imageName);
            for (auto person = 0 ; person < numberPeople ; person++)
            {
                // Comma at any moment but first element
                if (mFirstElementAdded)
                {
                    mJsonOfstream.comma();
                    mJsonOfstream.enter();
                }
                else
                    mFirstElementAdded = true;

                // New element
                mJsonOfstream.objectOpen();

                // image_id
                mJsonOfstream.key("image_id");
                mJsonOfstream.plainText(imageId);
                mJsonOfstream.comma();

                // category_id
                mJsonOfstream.key("category_id");
                mJsonOfstream.plainText("1");
                mJsonOfstream.comma();

                // keypoints - i.e. poseKeypoints
                mJsonOfstream.key("keypoints");
                mJsonOfstream.arrayOpen();
                std::vector<int> indexesInCocoOrder;
                if (numberBodyParts == 18)
                    indexesInCocoOrder = std::vector<int>{0, 15,14,17,16,    5,2,6,3,7,    4,11,8,12, 9,    13,10};
                else if (numberBodyParts == 19 || numberBodyParts == 59)
                    indexesInCocoOrder = std::vector<int>{0, 16,15,18,17,    5,2,6,3,7,    4,12,9,13,10,    14,11};
                else if (numberBodyParts == 23)
                    indexesInCocoOrder = std::vector<int>{18,21,19,22,20,    4,1,5,2,6,    3,13,8,14, 9,    15,10};
                else
                    error("Invalid number of body parts (" + std::to_string(numberBodyParts) + ").",
                          __LINE__, __FUNCTION__, __FILE__);
                for (auto bodyPart = 0u ; bodyPart < indexesInCocoOrder.size() ; bodyPart++)
                {
                    const auto finalIndex = 3*(person*numberBodyParts + indexesInCocoOrder.at(bodyPart));
                    mJsonOfstream.plainText(poseKeypoints[finalIndex]);
                    mJsonOfstream.comma();
                    mJsonOfstream.plainText(poseKeypoints[finalIndex+1]);
                    mJsonOfstream.comma();
                    mJsonOfstream.plainText((poseKeypoints[finalIndex+2] > 0.f ? 1 : 0));
                    // mJsonOfstream.plainText(poseKeypoints[finalIndex+2]); // For debugging
                    if (bodyPart < indexesInCocoOrder.size() - 1u)
                        mJsonOfstream.comma();
                }
                mJsonOfstream.arrayClose();
                mJsonOfstream.comma();

                // score
                mJsonOfstream.key("score");
                mJsonOfstream.plainText(poseScores[person]);

                mJsonOfstream.objectClose();
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
