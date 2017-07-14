#include <cuda_runtime_api.h>
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
                numberHeatMapChannels += POSE_NUMBER_BODY_PARTS[(int)poseModel];
            if (heatMapTypesHas(heatMapTypes, HeatMapType::Background))
                numberHeatMapChannels += 1;
            if (heatMapTypesHas(heatMapTypes, HeatMapType::PAFs))
                numberHeatMapChannels += (int)POSE_BODY_PART_PAIRS[(int)poseModel].size();
            return numberHeatMapChannels;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0;
        }
    }

    PoseExtractor::PoseExtractor(const Point<int>& netOutputSize, const Point<int>& outputSize, const PoseModel poseModel,
                                 const std::vector<HeatMapType>& heatMapTypes, const ScaleMode heatMapScale) :
        mPoseModel{poseModel},
        mNetOutputSize{netOutputSize},
        mOutputSize{outputSize},
        mHeatMapTypes{heatMapTypes},
        mHeatMapScaleMode{heatMapScale}
    {
        try
        {
            // Error check
            if (mHeatMapScaleMode != ScaleMode::ZeroToOne && mHeatMapScaleMode != ScaleMode::PlusMinusOne && mHeatMapScaleMode != ScaleMode::UnsignedChar)
                error("The ScaleMode heatMapScale must be ZeroToOne, PlusMinusOne or UnsignedChar.", __LINE__, __FUNCTION__, __FILE__);

            // Properties
            for (auto& property : mProperties)
                property = 0.;
            mProperties[(int)PoseProperty::NMSThreshold] = POSE_DEFAULT_NMS_THRESHOLD[(int)mPoseModel];
            mProperties[(int)PoseProperty::ConnectInterMinAboveThreshold] = POSE_DEFAULT_CONNECT_INTER_MIN_ABOVE_THRESHOLD[(int)mPoseModel];
            mProperties[(int)PoseProperty::ConnectInterThreshold] = POSE_DEFAULT_CONNECT_INTER_THRESHOLD[(int)mPoseModel];
            mProperties[(int)PoseProperty::ConnectMinSubsetCnt] = POSE_DEFAULT_CONNECT_MIN_SUBSET_CNT[(int)mPoseModel];
            mProperties[(int)PoseProperty::ConnectMinSubsetScore] = POSE_DEFAULT_CONNECT_MIN_SUBSET_SCORE[(int)mPoseModel];
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
        // Get thread id
        mThreadId = {std::this_thread::get_id()};

        // Deep net initialization
        netInitializationOnThread();
    }

    Array<float> PoseExtractor::getHeatMaps() const
    {
        try
        {
            checkThread();
            Array<float> poseHeatMaps;
            if (!mHeatMapTypes.empty())
            {
                // Allocate memory
                const auto numberHeatMapChannels = getNumberHeatMapChannels(mHeatMapTypes, mPoseModel);
                poseHeatMaps.reset({numberHeatMapChannels, mNetOutputSize.y, mNetOutputSize.x});

                // Copy memory
                const auto channelOffset = poseHeatMaps.getVolume(1, 2);
                const auto volumeBodyParts = POSE_NUMBER_BODY_PARTS[(int)mPoseModel] * channelOffset;
                const auto volumePAFs = POSE_BODY_PART_PAIRS[(int)mPoseModel].size() * channelOffset;
                unsigned int totalOffset = 0u;
                if (heatMapTypesHas(mHeatMapTypes, HeatMapType::Parts))
                {
                    cudaMemcpy(poseHeatMaps.getPtr(), getHeatMapGpuConstPtr(), volumeBodyParts * sizeof(float), cudaMemcpyDeviceToHost);
                    // Change from [0,1] to [-1,1]
                    if (mHeatMapScaleMode == ScaleMode::PlusMinusOne)
                        for (auto i = 0u ; i < volumeBodyParts ; i++)
                            poseHeatMaps[i] = fastTruncate(poseHeatMaps[i]) * 2.f - 1.f;
                    // [0, 255]
                    else if (mHeatMapScaleMode == ScaleMode::UnsignedChar)
                        for (auto i = 0u ; i < volumeBodyParts ; i++)
                            poseHeatMaps[i] = (float)intRound(fastTruncate(poseHeatMaps[i]) * 255.f);
                    // Avoid values outside original range
                    else
                        for (auto i = 0u ; i < volumeBodyParts ; i++)
                            poseHeatMaps[i] = fastTruncate(poseHeatMaps[i]);
                    totalOffset += (unsigned int)volumeBodyParts;
                }
                if (heatMapTypesHas(mHeatMapTypes, HeatMapType::Background))
                {
                    cudaMemcpy(poseHeatMaps.getPtr() + totalOffset, getHeatMapGpuConstPtr() + volumeBodyParts, channelOffset * sizeof(float), cudaMemcpyDeviceToHost);
                    // Change from [0,1] to [-1,1]
                    auto* poseHeatMapsPtr = poseHeatMaps.getPtr() + totalOffset;
                    if (mHeatMapScaleMode == ScaleMode::PlusMinusOne)
                        for (auto i = 0u ; i < channelOffset ; i++)
                            poseHeatMapsPtr[i] = fastTruncate(poseHeatMapsPtr[i]) * 2.f - 1.f;
                    // [0, 255]
                    else if (mHeatMapScaleMode == ScaleMode::UnsignedChar)
                        for (auto i = 0u ; i < channelOffset ; i++)
                            poseHeatMapsPtr[i] = (float)intRound(fastTruncate(poseHeatMapsPtr[i]) * 255.f);
                    // Avoid values outside original range
                    else
                        for (auto i = 0u ; i < channelOffset ; i++)
                            poseHeatMapsPtr[i] = fastTruncate(poseHeatMapsPtr[i]);
                    totalOffset += (unsigned int)channelOffset;
                }
                if (heatMapTypesHas(mHeatMapTypes, HeatMapType::PAFs))
                {
                    cudaMemcpy(poseHeatMaps.getPtr() + totalOffset, getHeatMapGpuConstPtr() + volumeBodyParts + channelOffset, volumePAFs * sizeof(float), cudaMemcpyDeviceToHost);
                    // Change from [-1,1] to [0,1]. Note that PAFs are in [-1,1]
                    auto* poseHeatMapsPtr = poseHeatMaps.getPtr() + totalOffset;
                    if (mHeatMapScaleMode == ScaleMode::ZeroToOne)
                        for (auto i = 0u ; i < volumePAFs ; i++)
                            poseHeatMapsPtr[i] = fastTruncate(poseHeatMapsPtr[i], -1.f) * 0.5f + 0.5f;
                    // [0, 255]
                    else if (mHeatMapScaleMode == ScaleMode::UnsignedChar)
                        for (auto i = 0u ; i < volumePAFs ; i++)
                            poseHeatMapsPtr[i] = (float)intRound(fastTruncate(poseHeatMapsPtr[i], -1.f) * 128.5f + 128.5f);
                    // Avoid values outside original range
                    else
                        for (auto i = 0u ; i < volumePAFs ; i++)
                            poseHeatMapsPtr[i] = fastTruncate(poseHeatMapsPtr[i], -1.f);
                    totalOffset += (unsigned int)volumePAFs;
                }
                // Copy all at once
                // cudaMemcpy(poseHeatMaps.getPtr(), getHeatMapGpuConstPtr(), poseHeatMaps.getVolume() * sizeof(float), cudaMemcpyDeviceToHost);
            }
            return poseHeatMaps;
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
                error("The CPU/GPU pointer data cannot be accessed from a different thread.", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
