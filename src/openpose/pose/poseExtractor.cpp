#ifdef USE_CUDA
    #include <cuda_runtime_api.h>
#endif
#include <openpose/core/enumClasses.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/pose/poseExtractor.hpp>

namespace op
{
    bool heatMapTypesHas(const std::vector<HeatMapType>& heatMapTypes, const HeatMapType heatMapType)
    {
        try
        {
            for (auto heatMapTypeVector : heatMapTypes)
                if (heatMapTypeVector == heatMapType)
                    return true;
            return false;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    int getNumberHeatMapChannels(const std::vector<HeatMapType>& heatMapTypes, const PoseModel poseModel)
    {
        try
        {
            auto numberHeatMapChannels = 0;
            if (heatMapTypesHas(heatMapTypes, HeatMapType::Parts))
                numberHeatMapChannels += getPoseNumberBodyParts(poseModel);
            if (heatMapTypesHas(heatMapTypes, HeatMapType::Background))
                numberHeatMapChannels += 1;
            if (heatMapTypesHas(heatMapTypes, HeatMapType::PAFs))
                numberHeatMapChannels += (int)getPosePartPairs(poseModel).size();
            return numberHeatMapChannels;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0;
        }
    }

    PoseExtractor::PoseExtractor(const PoseModel poseModel, const std::vector<HeatMapType>& heatMapTypes,
                                 const ScaleMode heatMapScale) :
        mPoseModel{poseModel},
        mNetOutputSize{0,0},
        mHeatMapTypes{heatMapTypes},
        mHeatMapScaleMode{heatMapScale}
    {
        try
        {
            // Error check
            if (mHeatMapScaleMode != ScaleMode::ZeroToOne && mHeatMapScaleMode != ScaleMode::PlusMinusOne
                && mHeatMapScaleMode != ScaleMode::UnsignedChar && mHeatMapScaleMode != ScaleMode::NoScale)
                error("The ScaleMode heatMapScale must be ZeroToOne, PlusMinusOne, UnsignedChar, or NoScale.",
                      __LINE__, __FUNCTION__, __FILE__);

            // Properties
            for (auto& property : mProperties)
                property = 0.;
            mProperties[(int)PoseProperty::NMSThreshold] = getPoseDefaultNmsThreshold(mPoseModel);
            mProperties[(int)PoseProperty::ConnectInterMinAboveThreshold]
                = getPoseDefaultConnectInterMinAboveThreshold(mPoseModel);
            mProperties[(int)PoseProperty::ConnectInterThreshold] = getPoseDefaultConnectInterThreshold(mPoseModel);
            mProperties[(int)PoseProperty::ConnectMinSubsetCnt] = getPoseDefaultMinSubsetCnt(mPoseModel);
            mProperties[(int)PoseProperty::ConnectMinSubsetScore] = getPoseDefaultConnectMinSubsetScore(mPoseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    PoseExtractor::~PoseExtractor()
    {
    }

    void PoseExtractor::initializationOnThread()
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

    Array<float> PoseExtractor::getHeatMaps() const
    {
        try
        {
            checkThread();
            Array<float> heatMaps;
            if (!mHeatMapTypes.empty())
            {
                // Get heatmaps size
                const auto heatMapSize = getHeatMapSize();

                // Allocate memory
                const auto numberHeatMapChannels = getNumberHeatMapChannels(mHeatMapTypes, mPoseModel);
                heatMaps.reset({numberHeatMapChannels, heatMapSize[2], heatMapSize[3]});

                // Copy memory
                const auto channelOffset = heatMaps.getVolume(1, 2);
                const auto volumeBodyParts = getPoseNumberBodyParts(mPoseModel) * channelOffset;
                const auto volumePAFs = getPosePartPairs(mPoseModel).size() * channelOffset;
                auto totalOffset = 0u;
                // Body parts
                if (heatMapTypesHas(mHeatMapTypes, HeatMapType::Parts))
                {
                    #ifdef USE_CUDA
                        cudaMemcpy(heatMaps.getPtr(), getHeatMapGpuConstPtr(),
                                   volumeBodyParts * sizeof(float), cudaMemcpyDeviceToHost);
                    #else
                        const auto* heatMapCpuPtr = getHeatMapCpuConstPtr();
                        std::copy(heatMapCpuPtr, heatMapCpuPtr+volumeBodyParts, heatMaps.getPtr());
                    #endif
                    if (mHeatMapScaleMode != ScaleMode::NoScale)
                    {
                        // Change from [0,1] to [-1,1]
                        if (mHeatMapScaleMode == ScaleMode::PlusMinusOne)
                            for (auto i = 0u ; i < volumeBodyParts ; i++)
                                heatMaps[i] = fastTruncate(heatMaps[i]) * 2.f - 1.f;
                        // [0, 255]
                        else if (mHeatMapScaleMode == ScaleMode::UnsignedChar)
                            for (auto i = 0u ; i < volumeBodyParts ; i++)
                                heatMaps[i] = (float)intRound(fastTruncate(heatMaps[i]) * 255.f);
                        // Avoid values outside original range
                        else
                            for (auto i = 0u ; i < volumeBodyParts ; i++)
                                heatMaps[i] = fastTruncate(heatMaps[i]);
                    }
                    totalOffset += (unsigned int)volumeBodyParts;
                }
                // Background
                if (heatMapTypesHas(mHeatMapTypes, HeatMapType::Background))
                {
                    auto* heatMapsPtr = heatMaps.getPtr() + totalOffset;
                    #ifdef USE_CUDA
                        cudaMemcpy(heatMapsPtr, getHeatMapGpuConstPtr() + volumeBodyParts,
                                   channelOffset * sizeof(float), cudaMemcpyDeviceToHost);
                    #else
                        const auto* heatMapCpuPtr = getHeatMapCpuConstPtr();
                        std::copy(heatMapCpuPtr + volumeBodyParts, heatMapCpuPtr + volumeBodyParts + channelOffset,
                                  heatMapsPtr);
                    #endif
                    if (mHeatMapScaleMode != ScaleMode::NoScale)
                    {
                        // Change from [0,1] to [-1,1]
                        if (mHeatMapScaleMode == ScaleMode::PlusMinusOne)
                            for (auto i = 0u ; i < channelOffset ; i++)
                                heatMapsPtr[i] = fastTruncate(heatMapsPtr[i]) * 2.f - 1.f;
                        // [0, 255]
                        else if (mHeatMapScaleMode == ScaleMode::UnsignedChar)
                            for (auto i = 0u ; i < channelOffset ; i++)
                                heatMapsPtr[i] = (float)intRound(fastTruncate(heatMapsPtr[i]) * 255.f);
                        // Avoid values outside original range
                        else
                            for (auto i = 0u ; i < channelOffset ; i++)
                                heatMapsPtr[i] = fastTruncate(heatMapsPtr[i]);
                    }
                    totalOffset += (unsigned int)channelOffset;
                }
                // PAFs
                if (heatMapTypesHas(mHeatMapTypes, HeatMapType::PAFs))
                {
                    auto* heatMapsPtr = heatMaps.getPtr() + totalOffset;
                    #ifdef USE_CUDA
                        cudaMemcpy(heatMapsPtr,
                                   getHeatMapGpuConstPtr() + volumeBodyParts + channelOffset,
                                   volumePAFs * sizeof(float), cudaMemcpyDeviceToHost);
                    #else
                        const auto* heatMapCpuPtr = getHeatMapCpuConstPtr();
                        std::copy(heatMapCpuPtr + volumeBodyParts + channelOffset,
                                  heatMapCpuPtr + volumeBodyParts + channelOffset + volumePAFs,
                                  heatMapsPtr);
                    #endif
                    if (mHeatMapScaleMode != ScaleMode::NoScale)
                    {
                        // Change from [-1,1] to [0,1]. Note that PAFs are in [-1,1]
                        if (mHeatMapScaleMode == ScaleMode::ZeroToOne)
                            for (auto i = 0u ; i < volumePAFs ; i++)
                                heatMapsPtr[i] = fastTruncate(heatMapsPtr[i], -1.f) * 0.5f + 0.5f;
                        // [0, 255]
                        else if (mHeatMapScaleMode == ScaleMode::UnsignedChar)
                            for (auto i = 0u ; i < volumePAFs ; i++)
                                heatMapsPtr[i] = (float)intRound(
                                    fastTruncate(heatMapsPtr[i], -1.f) * 128.5f + 128.5f
                                );
                        // Avoid values outside original range
                        else
                            for (auto i = 0u ; i < volumePAFs ; i++)
                                heatMapsPtr[i] = fastTruncate(heatMapsPtr[i], -1.f);
                    }
                    totalOffset += (unsigned int)volumePAFs;
                }
                // Copy all at once
                // cudaMemcpy(heatMaps.getPtr(), getHeatMapGpuConstPtr(), heatMaps.getVolume() * sizeof(float),
                //            cudaMemcpyDeviceToHost);
            }
            return heatMaps;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }

    Array<float> PoseExtractor::getPoseKeypoints() const
    {
        try
        {
            checkThread();
            return mPoseKeypoints;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }

    Array<float> PoseExtractor::getPoseScores() const
    {
        try
        {
            checkThread();
            return mPoseScores;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }

    float PoseExtractor::getScaleNetToOutput() const
    {
        try
        {
            checkThread();
            return mScaleNetToOutput;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.;
        }
    }

    double PoseExtractor::get(const PoseProperty property) const
    {
        try
        {
            return mProperties.at((int)property);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.;
        }
    }

    void PoseExtractor::set(const PoseProperty property, const double value)
    {
        try
        {
            mProperties.at((int)property) = {value};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PoseExtractor::increase(const PoseProperty property, const double value)
    {
        try
        {
            mProperties[(int)property] = mProperties.at((int)property) + value;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PoseExtractor::checkThread() const
    {
        try
        {
            if(mThreadId != std::this_thread::get_id())
                error("The CPU/GPU pointer data cannot be accessed from a different thread.",
                      __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
