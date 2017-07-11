#ifndef OPENPOSE_HAND_HAND_DETECTOR_FROM_TXT_HPP
#define OPENPOSE_HAND_HAND_DETECTOR_FROM_TXT_HPP

#include <array>
#include <string>
#include <vector>
#include <openpose/core/array.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/core/macros.hpp>

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
