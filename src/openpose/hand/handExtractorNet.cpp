#include <openpose/utilities/check.hpp>
#include <openpose/hand/handExtractorNet.hpp>

namespace op
{
    HandExtractorNet::HandExtractorNet(const Point<int>& netInputSize, const Point<int>& netOutputSize,
                                       const unsigned short numberScales, const float rangeScales,
                                       const std::vector<HeatMapType>& heatMapTypes, const ScaleMode heatMapScale) :
        mMultiScaleNumberAndRange{std::make_pair(numberScales, rangeScales)},
        mNetOutputSize{netOutputSize},
        mHandImageCrop{{1, 3, mNetOutputSize.y, mNetOutputSize.x}},
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
                log("Note only keypoint heatmaps are available with hand heatmaps (no background nor PAFs).",
                    Priority::High);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    HandExtractorNet::~HandExtractorNet()
    {
    }

    void HandExtractorNet::initializationOnThread()
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

    std::array<Array<float>, 2> HandExtractorNet::getHandKeypoints() const
    {
        try
        {
            checkThread();
            return mHandKeypoints;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::array<Array<float>, 2>(); // Parentheses instead of braces to avoid error in GCC 4.8
        }
    }

    std::array<Array<float>, 2> HandExtractorNet::getHeatMaps() const
    {
        try
        {
            checkThread();
            return mHeatMaps;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::array<Array<float>, 2>(); // Parentheses instead of braces to avoid error in GCC 4.8
        }
    }

    void HandExtractorNet::checkThread() const
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
