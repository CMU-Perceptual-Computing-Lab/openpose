#ifndef OPENPOSE_HAND_HAND_DETECTOR_FROM_TXT_HPP
#define OPENPOSE_HAND_HAND_DETECTOR_FROM_TXT_HPP

#include <openpose/core/common.hpp>
#include <openpose/pose/enumClasses.hpp>

namespace op
{
    class OP_API HandDetectorFromTxt
    {
    public:
        explicit HandDetectorFromTxt(const std::string& txtDirectoryPath);

        std::vector<std::array<Rectangle<float>, 2>> detectHands();

    private:
        const std::string mTxtDirectoryPath;
        const std::vector<std::string> mFilePaths;
        long long mFrameNameCounter;

        DELETE_COPY(HandDetectorFromTxt);
    };
}

#endif // OPENPOSE_HAND_HAND_DETECTOR_FROM_TXT_HPP
