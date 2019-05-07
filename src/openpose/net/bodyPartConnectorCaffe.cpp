#ifdef USE_CAFFE
    #include <caffe/blob.hpp>
#endif
#ifdef USE_CUDA
    #include <openpose/gpu/cuda.hpp>
#endif
#ifdef USE_OPENCL
    #include <openpose/gpu/opencl.hcl>
    #include <openpose/gpu/cl2.hpp>
#endif
#include <openpose/net/bodyPartConnectorBase.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/net/bodyPartConnectorCaffe.hpp>

namespace op
{
    template <typename T>
    BodyPartConnectorCaffe<T>::BodyPartConnectorCaffe() :
        mPoseModel{PoseModel::Size},
        mMaximizePositives{false},
        pBodyPartPairsGpuPtr{nullptr},
        pMapIdxGpuPtr{nullptr},
        pFinalOutputGpuPtr{nullptr}
    {
        try
        {
            #ifndef USE_CAFFE
                error("OpenPose must be compiled with the `USE_CAFFE` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    BodyPartConnectorCaffe<T>::~BodyPartConnectorCaffe()
    {
        try
        {
            #if defined USE_CAFFE && defined USE_CUDA
                cudaFree(pBodyPartPairsGpuPtr);
                cudaFree(pMapIdxGpuPtr);
                cudaFree(pFinalOutputGpuPtr);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::Reshape(const std::vector<ArrayCpuGpu<T>*>& bottom, const int gpuID)
    {
        try
        {
            #ifdef USE_CAFFE
                auto heatMapsBlob = bottom.at(0);
                auto peaksBlob = bottom.at(1);
                // Top shape
                const auto maxPeaks = peaksBlob->shape(2) - 1;
                const auto numberBodyParts = peaksBlob->shape(1);
                // Array sizes
                mTopSize = std::array<int, 4>{1, maxPeaks, numberBodyParts, 3};
                mHeatMapsSize = std::array<int, 4>{
                    heatMapsBlob->shape(0), heatMapsBlob->shape(1), heatMapsBlob->shape(2), heatMapsBlob->shape(3)};
                mPeaksSize = std::array<int, 4>{
                    peaksBlob->shape(0), peaksBlob->shape(1), peaksBlob->shape(2), peaksBlob->shape(3)};
                // GPU ID
                mGpuID = gpuID;
            #else
                UNUSED(bottom);
                UNUSED(gpuID);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::setPoseModel(const PoseModel poseModel)
    {
        try
        {
            mPoseModel = {poseModel};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::setMaximizePositives(const bool maximizePositives)
    {
        try
        {
            mMaximizePositives = {maximizePositives};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::setInterMinAboveThreshold(const T interMinAboveThreshold)
    {
        try
        {
            mInterMinAboveThreshold = {interMinAboveThreshold};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::setInterThreshold(const T interThreshold)
    {
        try
        {
            mInterThreshold = {interThreshold};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::setMinSubsetCnt(const int minSubsetCnt)
    {
        try
        {
            mMinSubsetCnt = {minSubsetCnt};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::setMinSubsetScore(const T minSubsetScore)
    {
        try
        {
            mMinSubsetScore = {minSubsetScore};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::setScaleNetToOutput(const T scaleNetToOutput)
    {
        try
        {
            mScaleNetToOutput = {scaleNetToOutput};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::Forward(const std::vector<ArrayCpuGpu<T>*>& bottom, Array<T>& poseKeypoints,
                                            Array<T>& poseScores)
    {
        try
        {
            // CUDA
            #ifdef USE_CUDA
                Forward_gpu(bottom, poseKeypoints, poseScores);
            // OpenCL
            #elif defined USE_OPENCL
                Forward_ocl(bottom, poseKeypoints, poseScores);
            // CPU
            #else
                Forward_cpu(bottom, poseKeypoints, poseScores);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::Forward_cpu(const std::vector<ArrayCpuGpu<T>*>& bottom, Array<T>& poseKeypoints,
                                                Array<T>& poseScores)
    {
        try
        {
            #ifdef USE_CAFFE
                const auto heatMapsBlob = bottom.at(0);
                const auto* const heatMapsPtr = heatMapsBlob->cpu_data();                 // ~8.5 ms COCO, ~35ms BODY_135
                const auto* const peaksPtr = bottom.at(1)->cpu_data();                    // ~0.02ms
                const auto maxPeaks = mTopSize[1];
                connectBodyPartsCpu(poseKeypoints, poseScores, heatMapsPtr, peaksPtr, mPoseModel,
                                    Point<int>{heatMapsBlob->shape(3), heatMapsBlob->shape(2)},
                                    maxPeaks, mInterMinAboveThreshold, mInterThreshold,
                                    mMinSubsetCnt, mMinSubsetScore, mScaleNetToOutput, mMaximizePositives);
            #else
                UNUSED(bottom);
                UNUSED(poseKeypoints);
                UNUSED(poseScores);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::Forward_ocl(const std::vector<ArrayCpuGpu<T>*>& bottom, Array<T>& poseKeypoints,
                                                Array<T>& poseScores)
    {
        try
        {
            #if defined USE_CAFFE && defined USE_OPENCL
                // Global data
                const auto heatMapsBlob = bottom.at(0);
                const auto* const heatMapsGpuPtr = heatMapsBlob->gpu_data();
                const auto* const peaksPtr = bottom.at(1)->cpu_data();
                const auto maxPeaks = mTopSize[1];
                const auto* const peaksGpuPtr = bottom.at(1)->gpu_data();

                // Initialize fixed pointers (1-time task) - It must be done in the same thread than Forward_gpu
                if (pBodyPartPairsGpuPtr == nullptr || pMapIdxGpuPtr == nullptr)
                {
                    // Data
                    const auto& bodyPartPairs = getPosePartPairs(mPoseModel);
                    const auto numberBodyParts = getPoseNumberBodyParts(mPoseModel);
                    const auto& mapIdxOffset = getPoseMapIndex(mPoseModel);
                    // Update mapIdx
                    const auto offset = (mPoseModel != PoseModel::BODY_25B ? 1 : 0);
                    auto mapIdx = mapIdxOffset;
                    for (auto& i : mapIdx)
                        i += (numberBodyParts+offset);
                    // Re-allocate memory
                    pBodyPartPairsGpuPtr = (unsigned int*)clCreateBuffer(
                        OpenCL::getInstance(mGpuID)->getContext().operator()(), CL_MEM_READ_WRITE,
                        bodyPartPairs.size() * sizeof(unsigned int), NULL, NULL);
                    clEnqueueWriteBuffer(
                        OpenCL::getInstance(mGpuID)->getQueue().operator()(), (cl_mem)pBodyPartPairsGpuPtr, CL_TRUE,
                        0, bodyPartPairs.size() * sizeof(unsigned int), &bodyPartPairs[0], NULL, NULL, NULL);
                    pMapIdxGpuPtr = (unsigned int*)clCreateBuffer(
                        OpenCL::getInstance(mGpuID)->getContext().operator()(), CL_MEM_READ_WRITE,
                        mapIdx.size() * sizeof(unsigned int), NULL, NULL);
                    clEnqueueWriteBuffer(
                        OpenCL::getInstance(mGpuID)->getQueue().operator()(), (cl_mem)pMapIdxGpuPtr, CL_TRUE,
                        0, mapIdx.size() * sizeof(unsigned int), &mapIdx[0], NULL, NULL, NULL);
                }
                // Initialize auxiliary pointers (1-time task)
                if (mFinalOutputCpu.empty()) // if (pFinalOutputGpuPtr == nullptr)
                {
                    // Data
                    const auto& bodyPartPairs = getPosePartPairs(mPoseModel);
                    const auto numberBodyPartPairs = bodyPartPairs.size() / 2;
                    // Allocate memory
                    mFinalOutputCpu.reset({(int)numberBodyPartPairs, maxPeaks, maxPeaks});
                    const auto totalComputations = mFinalOutputCpu.getVolume();
                    if (pFinalOutputGpuPtr == nullptr)
                        pFinalOutputGpuPtr = (T*)clCreateBuffer(
                            OpenCL::getInstance(mGpuID)->getContext().operator()(), CL_MEM_READ_WRITE,
                            totalComputations * sizeof(T), NULL, NULL);
                }

                // Run body part connector
                connectBodyPartsOcl(poseKeypoints, poseScores, heatMapsGpuPtr, peaksPtr, mPoseModel,
                                    Point<int>{heatMapsBlob->shape(3), heatMapsBlob->shape(2)},
                                    maxPeaks, mInterMinAboveThreshold, mInterThreshold,
                                    mMinSubsetCnt, mMinSubsetScore, mScaleNetToOutput, mMaximizePositives,
                                    mFinalOutputCpu, pFinalOutputGpuPtr, pBodyPartPairsGpuPtr, pMapIdxGpuPtr,
                                    peaksGpuPtr, mGpuID);
            #else
                UNUSED(bottom);
                UNUSED(poseKeypoints);
                UNUSED(poseScores);
                error("OpenPose must be compiled with the `USE_CAFFE` & `USE_CUDA` macro definitions in order to run"
                      " this functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::Forward_gpu(const std::vector<ArrayCpuGpu<T>*>& bottom, Array<T>& poseKeypoints,
                                                Array<T>& poseScores)
    {
        try
        {
            #if defined USE_CAFFE && defined USE_CUDA
                // Global data
                const auto heatMapsBlob = bottom.at(0);
                const auto* const heatMapsGpuPtr = heatMapsBlob->gpu_data();
                const auto* const peaksPtr = bottom.at(1)->cpu_data();
                const auto maxPeaks = mTopSize[1];
                const auto* const peaksGpuPtr = bottom.at(1)->gpu_data();

                // Initialize fixed pointers (1-time task) - It must be done in the same thread than Forward_gpu
                if (pBodyPartPairsGpuPtr == nullptr || pMapIdxGpuPtr == nullptr)
                {
                    // Free previous memory
                    cudaFree(pBodyPartPairsGpuPtr);
                    cudaFree(pMapIdxGpuPtr);
                    // Data
                    const auto& bodyPartPairs = getPosePartPairs(mPoseModel);
                    const auto numberBodyParts = getPoseNumberBodyParts(mPoseModel);
                    const auto& mapIdxOffset = getPoseMapIndex(mPoseModel);
                    // Update mapIdx
                    const auto offset = (addBkgChannel(mPoseModel) ? 1 : 0);
                    auto mapIdx = mapIdxOffset;
                    for (auto& i : mapIdx)
                        i += (numberBodyParts+offset);
                    // Re-allocate memory
                    cudaMalloc((void **)&pBodyPartPairsGpuPtr, bodyPartPairs.size() * sizeof(unsigned int));
                    cudaMemcpy(pBodyPartPairsGpuPtr, &bodyPartPairs[0], bodyPartPairs.size() * sizeof(unsigned int),
                               cudaMemcpyHostToDevice);
                    cudaMalloc((void **)&pMapIdxGpuPtr, mapIdx.size() * sizeof(unsigned int));
                    cudaMemcpy(pMapIdxGpuPtr, &mapIdx[0], mapIdx.size() * sizeof(unsigned int),
                               cudaMemcpyHostToDevice);
                    // Sanity check
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                }
                // Initialize auxiliary pointers (1-time task)
                if (mFinalOutputCpu.empty()) // if (pFinalOutputGpuPtr == nullptr)
                {
                    // Data
                    const auto& bodyPartPairs = getPosePartPairs(mPoseModel);
                    const auto numberBodyPartPairs = bodyPartPairs.size() / 2;
                    // Allocate memory
                    mFinalOutputCpu.reset({(int)numberBodyPartPairs, maxPeaks, maxPeaks});
                    const auto totalComputations = mFinalOutputCpu.getVolume();
                    if (pFinalOutputGpuPtr == nullptr)
                        cudaMalloc((void **)&pFinalOutputGpuPtr, totalComputations * sizeof(float));
                    // Sanity check
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                }

                // Run body part connector
                connectBodyPartsGpu(poseKeypoints, poseScores, heatMapsGpuPtr, peaksPtr, mPoseModel,
                                    Point<int>{heatMapsBlob->shape(3), heatMapsBlob->shape(2)},
                                    maxPeaks, mInterMinAboveThreshold, mInterThreshold,
                                    mMinSubsetCnt, mMinSubsetScore, mScaleNetToOutput, mMaximizePositives,
                                    mFinalOutputCpu, pFinalOutputGpuPtr, pBodyPartPairsGpuPtr, pMapIdxGpuPtr,
                                    peaksGpuPtr);
            #else
                UNUSED(bottom);
                UNUSED(poseKeypoints);
                UNUSED(poseScores);
                error("OpenPose must be compiled with the `USE_CAFFE` & `USE_CUDA` macro definitions in order to run"
                      " this functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::Backward_cpu(const std::vector<ArrayCpuGpu<T>*>& top,
                                                 const std::vector<bool>& propagate_down,
                                                 const std::vector<ArrayCpuGpu<T>*>& bottom)
    {
        try
        {
            UNUSED(top);
            UNUSED(propagate_down);
            UNUSED(bottom);
            #ifdef USE_CAFFE
                NOT_IMPLEMENTED;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::Backward_gpu(const std::vector<ArrayCpuGpu<T>*>& top,
                                                 const std::vector<bool>& propagate_down,
                                                 const std::vector<ArrayCpuGpu<T>*>& bottom)
    {
        try
        {
            UNUSED(top);
            UNUSED(propagate_down);
            UNUSED(bottom);
            #ifdef USE_CAFFE
                NOT_IMPLEMENTED;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_FLOATING_TYPES_CLASS(BodyPartConnectorCaffe);
}
