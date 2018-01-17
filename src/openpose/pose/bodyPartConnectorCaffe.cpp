#ifdef USE_CAFFE
    #include <caffe/blob.hpp>
#endif
#include <openpose/pose/bodyPartConnectorBase.hpp>
#include <openpose/pose/bodyPartConnectorCaffe.hpp>

namespace op
{
    template <typename T>
    BodyPartConnectorCaffe<T>::BodyPartConnectorCaffe()
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
    void BodyPartConnectorCaffe<T>::Reshape(const std::vector<caffe::Blob<T>*>& bottom)
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
                mHeatMapsSize = std::array<int, 4>{heatMapsBlob->shape(0), heatMapsBlob->shape(1),
                                                   heatMapsBlob->shape(2), heatMapsBlob->shape(3)};
                mPeaksSize = std::array<int, 4>{peaksBlob->shape(0), peaksBlob->shape(1), peaksBlob->shape(2),
                                                peaksBlob->shape(3)};
            #else
                UNUSED(bottom);
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
    void BodyPartConnectorCaffe<T>::Forward_cpu(const std::vector<caffe::Blob<T>*>& bottom, Array<T>& poseKeypoints,
                                                Array<T>& poseScores)
    {
        try
        {
            #ifdef USE_CAFFE
                const auto heatMapsBlob = bottom.at(0);
                const auto* const heatMapsPtr = heatMapsBlob->cpu_data();                 // ~8.5 ms COCO, 27ms BODY_59
                const auto* const peaksPtr = bottom.at(1)->cpu_data();                    // ~0.02ms
                const auto maxPeaks = mTopSize[1];
                connectBodyPartsCpu(poseKeypoints, poseScores, heatMapsPtr, peaksPtr, mPoseModel,
                                    Point<int>{heatMapsBlob->shape(3), heatMapsBlob->shape(2)},
                                    maxPeaks, mInterMinAboveThreshold, mInterThreshold,
                                    mMinSubsetCnt, mMinSubsetScore, mScaleNetToOutput);
            #else
                UNUSED(bottom);
                UNUSED(poseKeypoints);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::Forward_gpu(const std::vector<caffe::Blob<T>*>& bottom, Array<T>& poseKeypoints,
                                                Array<T>& poseScores)
    {
        try
        {
            #if defined USE_CAFFE && defined USE_CUDA
                const auto heatMapsBlob = bottom.at(0);
                const auto* const heatMapsPtr = heatMapsBlob->cpu_data();
                const auto* const peaksPtr = bottom.at(1)->cpu_data();
                const auto* const heatMapsGpuPtr = heatMapsBlob->gpu_data();
                const auto* const peaksGpuPtr = bottom.at(1)->gpu_data();
                const auto maxPeaks = mTopSize[1];
                connectBodyPartsGpu(poseKeypoints, poseScores, heatMapsPtr, peaksPtr, mPoseModel,
                                    Point<int>{heatMapsBlob->shape(3), heatMapsBlob->shape(2)},
                                    maxPeaks, mInterMinAboveThreshold, mInterThreshold,
                                    mMinSubsetCnt, mMinSubsetScore, mScaleNetToOutput,
                                    heatMapsGpuPtr, peaksGpuPtr);
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
    void BodyPartConnectorCaffe<T>::Backward_cpu(const std::vector<caffe::Blob<T>*>& top,
                                                 const std::vector<bool>& propagate_down,
                                                 const std::vector<caffe::Blob<T>*>& bottom)
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
    void BodyPartConnectorCaffe<T>::Backward_gpu(const std::vector<caffe::Blob<T>*>& top,
                                                 const std::vector<bool>& propagate_down,
                                                 const std::vector<caffe::Blob<T>*>& bottom)
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
