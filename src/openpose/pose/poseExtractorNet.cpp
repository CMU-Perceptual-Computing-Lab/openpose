#ifdef USE_CUDA
    #include <cuda_runtime_api.h>
#endif
#include <openpose/core/enumClasses.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/pose/poseExtractorNet.hpp>

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

    PoseExtractorNet::PoseExtractorNet(const PoseModel poseModel, const std::vector<HeatMapType>& heatMapTypes,
                                       const ScaleMode heatMapScale, const bool addPartCandidates) :
        mPoseModel{poseModel},
        mNetOutputSize{0,0},
        mHeatMapTypes{heatMapTypes},
        mHeatMapScaleMode{heatMapScale},
        mAddPartCandidates{addPartCandidates}
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

    PoseExtractorNet::~PoseExtractorNet()
    {
    }

    void PoseExtractorNet::initializationOnThread()
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

    Array<float> PoseExtractorNet::getHeatMapsCopy() const
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

    std::vector<std::vector<std::array<float,3>>> PoseExtractorNet::getCandidatesCopy() const
    {
        try
        {
            // Security check
            checkThread();
            // Initialization
            std::vector<std::vector<std::array<float,3>>> candidates;
            // Fill candidates
            if (mAddPartCandidates)
            {
                const auto numberBodyParts = getPoseNumberBodyParts(mPoseModel);
                candidates.resize(numberBodyParts);
                const auto peaksArea = (POSE_MAX_PEOPLE+1) * 3;
                // Memory copy
                const auto* candidatesCpuPtr = getCandidatesCpuConstPtr();
                for (auto part = 0u ; part < numberBodyParts ; part++)
                {
                    const auto numberPartCandidates = candidatesCpuPtr[part*peaksArea];
                    candidates[part].resize(numberPartCandidates);
                    const auto* partCandidatesPtr = &candidatesCpuPtr[part*peaksArea+3];
                    for (auto candidate = 0 ; candidate < numberPartCandidates ; candidate++)
                        candidates[part][candidate] = {partCandidatesPtr[3*candidate] * mScaleNetToOutput,
                                                       partCandidatesPtr[3*candidate+1] * mScaleNetToOutput,
                                                       partCandidatesPtr[3*candidate+2]};
                }
            }
            // Return
            return candidates;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::vector<std::vector<std::array<float,3>>>{};
        }
    }

    Array<float> PoseExtractorNet::getPoseKeypoints() const
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

    Array<float> PoseExtractorNet::getPoseScores() const
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

    float PoseExtractorNet::getScaleNetToOutput() const
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

    double PoseExtractorNet::get(const PoseProperty property) const
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

    void PoseExtractorNet::set(const PoseProperty property, const double value)
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

    void PoseExtractorNet::increase(const PoseProperty property, const double value)
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

    void PoseExtractorNet::clear()
    {
        try
        {
            mPoseKeypoints.reset();
            mPoseScores.reset();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PoseExtractorNet::checkThread() const
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
