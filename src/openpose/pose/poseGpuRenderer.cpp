#include <openpose/pose/poseGpuRenderer.hpp>
#ifdef USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#endif
#include <openpose/pose/poseParameters.hpp>
#include <openpose/pose/renderPose.hpp>
#include <openpose/gpu/cuda.hpp>
#include <openpose/utilities/keypoint.hpp>

namespace op
{
    PoseGpuRenderer::PoseGpuRenderer(const PoseModel poseModel,
                                     const std::shared_ptr<PoseExtractorNet>& poseExtractorNet,
                                     const float renderThreshold, const bool blendOriginalFrame,
                                     const float alphaKeypoint, const float alphaHeatMap,
                                     const unsigned int elementToRender) :
        // #body elements to render = #body parts (size()) + #body part pair connections
        //                          + 3 (+whole pose +whole heatmaps +PAFs)
        // POSE_BODY_PART_MAPPING crashes on Windows, replaced by getPoseBodyPartMapping
        GpuRenderer{renderThreshold, alphaKeypoint, alphaHeatMap, blendOriginalFrame, elementToRender,
                    getNumberElementsToRender(poseModel)}, // mNumberElementsToRender
        PoseRenderer{poseModel},
        spPoseExtractorNet{poseExtractorNet},
        pGpuPose{nullptr},
        pMaxPtr{nullptr},
        pMinPtr{nullptr},
        pScalePtr{nullptr}
    {
    }

    PoseGpuRenderer::~PoseGpuRenderer()
    {
        try
        {
            // Free CUDA pointers - Note that if pointers are 0 (i.e., nullptr), no operation is performed.
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            #ifdef USE_CUDA
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                if (pGpuPose != nullptr)
                {
                    cudaFree(pGpuPose);
                    pGpuPose = nullptr;
                }
                if (pMaxPtr != nullptr)
                {
                    cudaFree(pMaxPtr);
                    pMaxPtr = nullptr;
                }
                if (pMinPtr != nullptr)
                {
                    cudaFree(pMinPtr);
                    pMinPtr = nullptr;
                }
                if (pScalePtr != nullptr)
                {
                    cudaFree(pScalePtr);
                    pScalePtr = nullptr;
                }
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #endif
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PoseGpuRenderer::initializationOnThread()
    {
        try
        {
            opLog("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // GPU memory allocation for rendering
            #ifdef USE_CUDA
                const auto gpuPoseVolume = POSE_MAX_PEOPLE * getPoseNumberBodyParts(mPoseModel) * 3 * sizeof(float);
                cudaMalloc((void**)(&pGpuPose), gpuPoseVolume);
                cudaMalloc((void**)&pMaxPtr, sizeof(float) * 2 * POSE_MAX_PEOPLE);
                cudaMalloc((void**)&pMinPtr, sizeof(float) * 2 * POSE_MAX_PEOPLE);
                cudaMalloc((void**)&pScalePtr, sizeof(float) * POSE_MAX_PEOPLE);
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #endif
            opLog("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::pair<int, std::string> PoseGpuRenderer::renderPose(
        Array<float>& outputData, const Array<float>& poseKeypoints, const float scaleInputToOutput,
        const float scaleNetToOutput)
    {
        try
        {
            // Sanity check
            if (outputData.empty())
                error("Empty Array<float> outputData.", __LINE__, __FUNCTION__, __FILE__);
            // GPU rendering
            const auto elementRendered = spElementToRender->load();
            std::string elementRenderedName;
            #ifdef USE_CUDA
                const auto numberPeople = poseKeypoints.getSize(0);
                if (numberPeople > 0 || elementRendered != 0 || !mBlendOriginalFrame)
                {
                    cpuToGpuMemoryIfNotCopiedYet(outputData.getPtr(), outputData.getVolume());
                    const auto numberBodyParts = getPoseNumberBodyParts(mPoseModel);
                    const auto hasBkg = addBkgChannel(mPoseModel);
                    const auto numberBodyPartsPlusBkg = numberBodyParts + (hasBkg ? 1 : 0);
                    const auto numberBodyPAFChannels = getPosePartPairs(mPoseModel).size();
                    const Point<unsigned int> frameSize{(unsigned int)outputData.getSize(1), (unsigned int)outputData.getSize(0)};
                    // Draw poseKeypoints
                    if (elementRendered == 0)
                    {
                        // Rescale keypoints to output size
                        auto poseKeypointsRescaled = poseKeypoints.clone();
                        scaleKeypoints(poseKeypointsRescaled, scaleInputToOutput);
                        // Render keypoints
                        if (!poseKeypoints.empty())
                        {
                            const auto gpuPoseVolume = numberPeople * numberBodyParts * 3 * sizeof(float);
                            cudaMemcpy(
                                pGpuPose, poseKeypointsRescaled.getConstPtr(), gpuPoseVolume, cudaMemcpyHostToDevice);
                        }
                        renderPoseKeypointsGpu(
                            *spGpuMemory, pMaxPtr, pMinPtr, pScalePtr, mPoseModel, numberPeople, frameSize, pGpuPose,
                            mRenderThreshold, mShowGooglyEyes, mBlendOriginalFrame, getAlphaKeypoint());
                    }
                    else
                    {
                        // If resized to input resolution: Replace scaleNetToOutput * scaleInputToOutput by
                        // scaleInputToOutput, and comment the sanity check.
                        // Sanity check
                        if (scaleNetToOutput == -1.f)
                            error("Non valid scaleNetToOutput.", __LINE__, __FUNCTION__, __FILE__);
                        // Parameters
                        const auto& heatMapSizes = spPoseExtractorNet->getHeatMapSize();
                        const Point<int> heatMapSize{heatMapSizes[3], heatMapSizes[2]};
                        const auto lastPAFChannel = numberBodyPartsPlusBkg+2+numberBodyPAFChannels/2;
                        // Add all heatmaps
                        if (elementRendered == 2)
                        // if (elementRendered == numberBodyPartsPlusBkg+1)
                        {
                            elementRenderedName = "Heatmaps";
                            renderPoseHeatMapsGpu(
                                *spGpuMemory, mPoseModel, frameSize, spPoseExtractorNet->getHeatMapGpuConstPtr(),
                                heatMapSize, scaleNetToOutput * scaleInputToOutput,
                                (mBlendOriginalFrame ? getAlphaHeatMap() : 1.f));
                        }
                        // Draw PAFs (Part Affinity Fields)
                        else if (elementRendered == 3)
                        // else if (elementRendered == numberBodyPartsPlusBkg+2)
                        {
                            elementRenderedName = "PAFs (Part Affinity Fields)";
                            renderPosePAFsGpu(
                                *spGpuMemory, mPoseModel, frameSize, spPoseExtractorNet->getHeatMapGpuConstPtr(),
                                heatMapSize, scaleNetToOutput * scaleInputToOutput,
                                (mBlendOriginalFrame ? getAlphaHeatMap() : 1.f));
                        }
                        // Draw specific body part or background
                        else if (elementRendered <= numberBodyPartsPlusBkg+2)
                        {
                            const auto realElementRendered = (elementRendered == 1
                                                                ? (hasBkg ? numberBodyParts : 0)
                                                                : elementRendered - 3 - (hasBkg ? 1:0));
                            elementRenderedName = mPartIndexToName.at(realElementRendered);
                            renderPoseHeatMapGpu(
                                *spGpuMemory, frameSize, spPoseExtractorNet->getHeatMapGpuConstPtr(), heatMapSize,
                                scaleNetToOutput * scaleInputToOutput, realElementRendered,
                                (mBlendOriginalFrame ? getAlphaHeatMap() : 1.f));
                        }
                        // Draw affinity between 2 body parts
                        else if (elementRendered <= lastPAFChannel)
                        {
                            const auto affinityPart = (elementRendered-numberBodyPartsPlusBkg-3)*2;
                            const auto affinityPartMapped = numberBodyPartsPlusBkg
                                                          + getPoseMapIndex(mPoseModel).at(affinityPart);
                            elementRenderedName = mPartIndexToName.at(affinityPartMapped);
                            elementRenderedName = elementRenderedName.substr(0, elementRenderedName.find("("));
                            renderPosePAFGpu(
                                *spGpuMemory, mPoseModel, frameSize, spPoseExtractorNet->getHeatMapGpuConstPtr(),
                                heatMapSize, scaleNetToOutput * scaleInputToOutput, affinityPartMapped,
                                (mBlendOriginalFrame ? getAlphaHeatMap() : 1.f));
                        }
                        // Draw neck-part distance channel
                        else
                        {
                            if (mPoseModel != PoseModel::BODY_25D)
                                error("Neck-part distance channel only for BODY_25D.",
                                      __LINE__, __FUNCTION__, __FILE__);
                            const auto distancePart = (elementRendered - lastPAFChannel - 1);
                            const auto distancePartMapped = (unsigned int)(
                                numberBodyPartsPlusBkg + numberBodyPAFChannels + distancePart);
                            elementRenderedName = mPartIndexToName.at(distancePartMapped);
                            renderPoseDistanceGpu(
                                *spGpuMemory, frameSize, spPoseExtractorNet->getHeatMapGpuConstPtr(), heatMapSize,
                                scaleNetToOutput * scaleInputToOutput, distancePartMapped,
                                (mBlendOriginalFrame ? getAlphaHeatMap() : 1.f));
                        }
                    }
                }
                // GPU memory to CPU if last renderer
                gpuToCpuMemoryIfLastRenderer(outputData.getPtr(), outputData.getVolume());
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #else
                UNUSED(outputData);
                UNUSED(poseKeypoints);
                UNUSED(scaleInputToOutput);
                UNUSED(scaleNetToOutput);
                error("OpenPose must be compiled with the `USE_CUDA` macro definitions in order to run this"
                      " functionality. You can alternatively use CPU rendering (flag `--render_pose 1`).",
                      __LINE__, __FUNCTION__, __FILE__);
            #endif
            // Return result
            return std::make_pair(elementRendered, elementRenderedName);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_pair(-1, "");
        }
    }
}
