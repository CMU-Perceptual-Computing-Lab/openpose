#include <openpose/utilities/check.hpp>
#include <openpose/face/faceExtractorNet.hpp>

namespace op
{
    FaceExtractorNet::FaceExtractorNet(const Point<int>& netInputSize, const Point<int>& netOutputSize,
                                       const std::vector<HeatMapType>& heatMapTypes, const ScaleMode heatMapScale) :
        mNetOutputSize{netOutputSize},
        mFaceImageCrop{{1, 3, mNetOutputSize.y, mNetOutputSize.x}},
        mHeatMapScaleMode{heatMapScale},
        mHeatMapTypes{heatMapTypes}
    {
        try
        {
            // Error check
            if (mHeatMapScaleMode != ScaleMode::ZeroToOne && mHeatMapScaleMode != ScaleMode::PlusMinusOne
                && mHeatMapScaleMode != ScaleMode::UnsignedChar)
                error("The ScaleMode heatMapScale must be ZeroToOne, PlusMinusOne or UnsignedChar.",
                      __LINE__, __FUNCTION__, __FILE__);
            checkE(netOutputSize.x, netInputSize.x, "Net input and output size must be equal.",
                   __LINE__, __FUNCTION__, __FILE__);
            checkE(netOutputSize.y, netInputSize.y, "Net input and output size must be equal.",
                   __LINE__, __FUNCTION__, __FILE__);
            checkE(netInputSize.x, netInputSize.y, "Net input size must be squared.",
                   __LINE__, __FUNCTION__, __FILE__);
            // Warnings
            if (!mHeatMapTypes.empty())
                log("Note only keypoint heatmaps are available with face heatmaps (no background nor PAFs).",
                    Priority::High);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    FaceExtractorNet::~FaceExtractorNet()
    {
    }

    void FaceExtractorNet::initializationOnThread()
    {
        try
        {
            // Get thread id
            mThreadId = {std::this_thread::get_id()};
            // Deep net initialization
            netInitializationOnThread();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Array<float> FaceExtractorNet::getFaceKeypoints() const
    {
        try
        {
            checkThread();
            return mFaceKeypoints;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }

    Array<float> FaceExtractorNet::getHeatMaps() const
    {
        try
        {
            checkThread();
            return mHeatMaps;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }

    void FaceExtractorNet::checkThread() const
    {
        try
        {
            if (mThreadId != std::this_thread::get_id())
                error("The CPU/GPU pointer data cannot be accessed from a different thread.",
                      __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
