#ifndef CPU_ONLY
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#endif
#include <openpose/pose/poseParameters.hpp>
#include <openpose/pose/renderPose.hpp>
#include <openpose/utilities/cuda.hpp>
#include <openpose/pose/poseRenderer.hpp>

namespace op
{
    std::map<unsigned int, std::string> createPartToName(const PoseModel poseModel)
    {
        try
        {
            // POSE_BODY_PART_MAPPING crashes on Windows, replaced by getPoseBodyPartMapping
            // auto partToName = POSE_BODY_PART_MAPPING[(int)poseModel];
            auto partToName = getPoseBodyPartMapping(poseModel);
            const auto& bodyPartPairs = POSE_BODY_PART_PAIRS[(int)poseModel];
            const auto& mapIdx = POSE_MAP_IDX[(int)poseModel];

            for (auto bodyPart = 0; bodyPart < bodyPartPairs.size(); bodyPart+=2)
            {
                const auto bodyPartPairsA = bodyPartPairs.at(bodyPart);
                const auto bodyPartPairsB = bodyPartPairs.at(bodyPart+1);
                const auto mapIdxA = mapIdx.at(bodyPart);
                const auto mapIdxB = mapIdx.at(bodyPart+1);
                partToName[mapIdxA] = partToName.at(bodyPartPairsA) + "->" + partToName.at(bodyPartPairsB) + "(X)";
                partToName[mapIdxB] = partToName.at(bodyPartPairsA) + "->" + partToName.at(bodyPartPairsB) + "(Y)";
            }

            return partToName;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    PoseRenderer::PoseRenderer(const Point<int>& heatMapsSize, const Point<int>& outputSize, const PoseModel poseModel,
                               const std::shared_ptr<PoseExtractor>& poseExtractor, const float renderThreshold,
                               const bool blendOriginalFrame, const float alphaKeypoint, const float alphaHeatMap,
                               const unsigned int elementToRender, const RenderMode renderMode) :
        // #body elements to render = #body parts (size()) + #body part pair connections + 3 (+whole pose +whole heatmaps +PAFs)
        // POSE_BODY_PART_MAPPING crashes on Windows, replaced by getPoseBodyPartMapping
        Renderer{(unsigned long long)(outputSize.area() * 3), alphaKeypoint, alphaHeatMap, elementToRender,
                 (unsigned int)(getPoseBodyPartMapping(poseModel).size() + POSE_BODY_PART_PAIRS[(int)poseModel].size()/2 + 3)}, // mNumberElementsToRender
        mRenderThreshold{renderThreshold},
        mHeatMapsSize{heatMapsSize},
        mOutputSize{outputSize},
        mPoseModel{poseModel},
        mPartIndexToName{createPartToName(poseModel)},
        spPoseExtractor{poseExtractor},
        mRenderMode{renderMode},
        mBlendOriginalFrame{blendOriginalFrame},
        mShowGooglyEyes{false},
        pGpuPose{nullptr}
    {
    }

    PoseRenderer::~PoseRenderer()
    {
        try
        {
            // Free CUDA pointers - Note that if pointers are 0 (i.e. nullptr), no operation is performed.
            #ifndef CPU_ONLY
                cudaFree(pGpuPose);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PoseRenderer::initializationOnThread()
    {
        try
        {
            log("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            Renderer::initializationOnThread();
            // GPU memory allocation for rendering
            #ifndef CPU_ONLY
                cudaMalloc((void**)(&pGpuPose), POSE_MAX_PEOPLE * POSE_NUMBER_BODY_PARTS[(int)mPoseModel] * 3 * sizeof(float));
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #endif
            log("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    bool PoseRenderer::getBlendOriginalFrame() const
    {
        try
        {
            return mBlendOriginalFrame;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    bool PoseRenderer::getShowGooglyEyes() const
    {
        try
        {
            return mShowGooglyEyes;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    void PoseRenderer::setBlendOriginalFrame(const bool blendOriginalFrame)
    {
        try
        {
            mBlendOriginalFrame = blendOriginalFrame;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PoseRenderer::setShowGooglyEyes(const bool showGooglyEyes)
    {
        try
        {
            mShowGooglyEyes = showGooglyEyes;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::pair<int, std::string> PoseRenderer::renderPose(Array<float>& outputData, const Array<float>& poseKeypoints,
                                                         const float scaleNetToOutput)
    {
        try
        {
            // Security checks
            if (outputData.empty())
                error("Empty Array<float> outputData.", __LINE__, __FUNCTION__, __FILE__);

            // CPU rendering
            if (mRenderMode == RenderMode::Cpu)
                return renderPoseCpu(outputData, poseKeypoints, scaleNetToOutput);

            // GPU rendering
            else
                return renderPoseGpu(outputData, poseKeypoints, scaleNetToOutput);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_pair(-1, "");
        }
    }

    std::pair<int, std::string> PoseRenderer::renderPoseCpu(Array<float>& outputData, const Array<float>& poseKeypoints,
                                                            const float scaleNetToOutput)
    {
        try
        {
            const auto elementRendered = spElementToRender->load();

            std::string elementRenderedName;
            // CPU rendering
            // Draw poseKeypoints
            if (elementRendered == 0)
                renderPoseKeypointsCpu(outputData, poseKeypoints, mPoseModel, mRenderThreshold, mBlendOriginalFrame);
            // Draw heat maps / PAFs
            else
            {
                UNUSED(scaleNetToOutput);
                error("CPU rendering only available for drawing keypoints, no heat maps nor PAFs.", __LINE__, __FUNCTION__, __FILE__);    
            }
            // Return result
            return std::make_pair(elementRendered, elementRenderedName);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_pair(-1, "");
        }
    }

    std::pair<int, std::string> PoseRenderer::renderPoseGpu(Array<float>& outputData, const Array<float>& poseKeypoints,
                                                            const float scaleNetToOutput)
    {
        try
        {
            const auto elementRendered = spElementToRender->load();

            std::string elementRenderedName;
            // GPU rendering
            #ifndef CPU_ONLY
                const auto numberPeople = poseKeypoints.getSize(0);
                if (numberPeople > 0 || elementRendered != 0 || !mBlendOriginalFrame)
                {
                    cpuToGpuMemoryIfNotCopiedYet(outputData.getPtr());
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                    const auto numberBodyParts = POSE_NUMBER_BODY_PARTS[(int)mPoseModel];
                    const auto numberBodyPartsPlusBkg = numberBodyParts+1;
                    // Draw poseKeypoints
                    if (elementRendered == 0)
                    {
                        if (!poseKeypoints.empty())
                            cudaMemcpy(pGpuPose, poseKeypoints.getConstPtr(), numberPeople * numberBodyParts * 3 * sizeof(float),
                                       cudaMemcpyHostToDevice);
                        renderPoseKeypointsGpu(*spGpuMemoryPtr, mPoseModel, numberPeople, mOutputSize, pGpuPose,
                                               mRenderThreshold, mShowGooglyEyes, mBlendOriginalFrame, getAlphaKeypoint());
                    }
                    else
                    {
                        if (scaleNetToOutput == -1.f)
                            error("Non valid scaleNetToOutput.", __LINE__, __FUNCTION__, __FILE__);
                        // Draw specific body part or bkg
                        if (elementRendered <= numberBodyPartsPlusBkg)
                        {
                            elementRenderedName = mPartIndexToName.at(elementRendered-1);
                            renderPoseHeatMapGpu(*spGpuMemoryPtr, mPoseModel, mOutputSize, spPoseExtractor->getHeatMapCpuConstPtr(),
                                                 mHeatMapsSize, scaleNetToOutput, elementRendered,
                                                 (mBlendOriginalFrame ? getAlphaHeatMap() : 1.f));
                        }
                        // Draw PAFs (Part Affinity Fields)
                        else if (elementRendered == numberBodyPartsPlusBkg+1)
                        {
                            elementRenderedName = "Heatmaps";
                            renderPoseHeatMapsGpu(*spGpuMemoryPtr, mPoseModel, mOutputSize, spPoseExtractor->getHeatMapCpuConstPtr(),
                                                  mHeatMapsSize, scaleNetToOutput, (mBlendOriginalFrame ? getAlphaHeatMap() : 1.f));
                        }
                        // Draw PAFs (Part Affinity Fields)
                        else if (elementRendered == numberBodyPartsPlusBkg+2)
                        {
                            elementRenderedName = "PAFs (Part Affinity Fields)";
                            renderPosePAFsGpu(*spGpuMemoryPtr, mPoseModel, mOutputSize, spPoseExtractor->getHeatMapCpuConstPtr(),
                                              mHeatMapsSize, scaleNetToOutput, (mBlendOriginalFrame ? getAlphaHeatMap() : 1.f));
                        }
                        // Draw affinity between 2 body parts
                        else
                        {
                            const auto affinityPart = (elementRendered-numberBodyPartsPlusBkg-3)*2;
                            const auto affinityPartMapped = POSE_MAP_IDX[(int)mPoseModel].at(affinityPart);
                            elementRenderedName = mPartIndexToName.at(affinityPartMapped);
                            elementRenderedName = elementRenderedName.substr(0, elementRenderedName.find("("));
                            renderPosePAFGpu(*spGpuMemoryPtr, mPoseModel, mOutputSize, spPoseExtractor->getHeatMapCpuConstPtr(),
                                             mHeatMapsSize, scaleNetToOutput, affinityPartMapped,
                                             (mBlendOriginalFrame ? getAlphaHeatMap() : 1.f));
                        }
                    }
                }
                // GPU memory to CPU if last renderer
                gpuToCpuMemoryIfLastRenderer(outputData.getPtr());
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            // CPU_ONLY mode
            #else
                error("GPU rendering not available if `CPU_ONLY` is set.", __LINE__, __FUNCTION__, __FILE__);
                UNUSED(elementRendered);
                UNUSED(outputData);
                UNUSED(poseKeypoints);
                UNUSED(scaleNetToOutput);
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
