#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/utilities/fastMath.hpp"
#include "openpose/utilities/openCv.hpp"
#include "openpose/core/cvMatToOpInput.hpp"

namespace op
{
    CvMatToOpInput::CvMatToOpInput(const cv::Size& netInputResolution, const int scaleNumber, const float scaleGap) :
        mScaleNumber{scaleNumber},
        mScaleGap{scaleGap},
        mInputNetSize4D{{mScaleNumber, 3, netInputResolution.height, netInputResolution.width}}
    {
    }

    Array<float> CvMatToOpInput::format(const cv::Mat& cvInputData) const
    {
        try
        {
            // Security checks
            if (cvInputData.empty())
                error("Wrong input element (empty cvInputData).", __LINE__, __FUNCTION__, __FILE__);

            // inputNetData - Reescale keeping aspect ratio and transform to float the input deep net image
            Array<float> inputNetData{mInputNetSize4D};
            const auto inputNetDataOffset = inputNetData.getVolume(1, 3);
            for (auto i = 0; i < mScaleNumber; i++)
            {
                const auto requestedScale = 1.f - i*mScaleGap;
                if (requestedScale > 1.f)
                    error("All scales must be <= 1, i.e. 1-num_scales*scale_gap <= 1", __LINE__, __FUNCTION__, __FILE__);

                const auto netInputWidth = inputNetData.getSize(3);
                const auto targetWidth  = fastTruncate(16 * intRound(netInputWidth * requestedScale / 16.), 1, netInputWidth/16*16);
                const auto netInputHeight = inputNetData.getSize(2);
                const auto targetHeight  = fastTruncate(16 * intRound(netInputHeight * requestedScale / 16.), 1, netInputHeight/16*16);
                const cv::Size targetSize{targetWidth, targetHeight};
                const auto scale = resizeGetScaleFactor(cvInputData.size(), targetSize);
                const cv::Mat frameWithNetSize = resizeFixedAspectRatio(cvInputData, scale, cv::Size{netInputWidth, netInputHeight});
                uCharCvMatToFloatPtr(inputNetData.getPtr() + i * inputNetDataOffset, frameWithNetSize, true);
            }

            return inputNetData;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }
}
