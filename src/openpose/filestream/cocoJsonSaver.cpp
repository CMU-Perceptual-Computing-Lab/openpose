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

    void CocoJsonSaver::record(const Array<float>& poseKeypoints, const std::string& imageName)
    {
        try
        {
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
                const std::vector<int> indexesInCocoOrder{0, 15, 14, 17, 16,        5, 2, 6, 3, 7,        4, 11, 8, 12, 9,        13, 10};
                for (auto bodyPart = 0 ; bodyPart < indexesInCocoOrder.size() ; bodyPart++)
                {
                    const auto finalIndex = 3*(person*numberBodyParts + indexesInCocoOrder.at(bodyPart));
                    mJsonOfstream.plainText(poseKeypoints[finalIndex]);
                    mJsonOfstream.comma();
                    mJsonOfstream.plainText(poseKeypoints[finalIndex+1]);
                    mJsonOfstream.comma();
                    mJsonOfstream.plainText(1);
                    if (bodyPart < indexesInCocoOrder.size() - 1)
                        mJsonOfstream.comma();
                }
                mJsonOfstream.arrayClose();
                mJsonOfstream.comma();

                // score
                mJsonOfstream.key("score");
                mJsonOfstream.plainText("0.0");

                mJsonOfstream.objectClose();
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
