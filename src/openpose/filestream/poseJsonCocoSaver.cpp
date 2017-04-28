#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/filestream/poseJsonCocoSaver.hpp"

namespace op
{
    PoseJsonCocoSaver::PoseJsonCocoSaver(const std::string& filePathToSave, const bool humanReadable) :
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
    PoseJsonCocoSaver::~PoseJsonCocoSaver()
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

    void PoseJsonCocoSaver::record(const Array<float>& poseKeyPoints, const int imageId)
    {
        try
        {
            const auto numberPeople = poseKeyPoints.getSize(0);
            const auto numberBodyParts = poseKeyPoints.getSize(1);
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

                // keypoints - i.e. poseKeyPoints
                mJsonOfstream.key("keypoints");
                mJsonOfstream.arrayOpen();
                const std::vector<int> indexesInCocoOrder{0, 15, 14, 17, 16,        5, 2, 6, 3, 7, 4,       11, 8, 12, 9, 13, 10};
                for (auto bodyPart = 0 ; bodyPart < indexesInCocoOrder.size() ; bodyPart++)
                {
                    const auto finalIndex = 3*(person*numberBodyParts + indexesInCocoOrder.at(bodyPart));
                    mJsonOfstream.plainText(poseKeyPoints[finalIndex]);
                    mJsonOfstream.comma();
                    mJsonOfstream.plainText(poseKeyPoints[finalIndex+1]);
                    mJsonOfstream.comma();
                    mJsonOfstream.plainText(1);
                    if (bodyPart < numberBodyParts-1)
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
