#ifdef USE_CAFFE
    #include <caffe/blob.hpp>
#endif
#ifdef USE_OPENCL
    #include <openpose/gpu/opencl.hcl>
    #include <openpose/gpu/cl2.hpp>
#endif
#include <openpose/net/resizeAndMergeBase.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/net/resizeAndMergeCaffe.hpp>

namespace op
{
    template <typename T>
    ResizeAndMergeCaffe<T>::ResizeAndMergeCaffe() :
        mScaleRatios{T(1)}
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
    ResizeAndMergeCaffe<T>::~ResizeAndMergeCaffe()
    {
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::LayerSetUp(const std::vector<caffe::Blob<T>*>& bottom,
                                            const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            #ifdef USE_CAFFE
                if (top.size() != 1)
                    error("top.size() != 1.", __LINE__, __FUNCTION__, __FILE__);
                if (bottom.size() != 1)
                    error("bottom.size() != 1.", __LINE__, __FUNCTION__, __FILE__);
            #else
                UNUSED(bottom);
                UNUSED(top);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Reshape(const std::vector<caffe::Blob<T>*>& bottom,
                                         const std::vector<caffe::Blob<T>*>& top,
                                         const T netFactor,
                                         const T scaleFactor,
                                         const bool mergeFirstDimension,
                                         const int gpuID)
    {
        try
        {
            #ifdef USE_CAFFE
                // Sanity checks
                if (top.size() != 1)
                    error("top.size() != 1", __LINE__, __FUNCTION__, __FILE__);
                if (bottom.empty())
                    error("bottom cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
                // Data
                auto* topBlob = top.at(0);
                const auto* bottomBlob = bottom.at(0);
                // Set top shape
                auto topShape = bottomBlob->shape();
                topShape[0] = (mergeFirstDimension ? 1 : bottomBlob->shape(0));
                // -1 and later +1 to take into account that we are using 0-based index
                // E.g., 100x100 image --> 200x200 --> 0-99 to 0-199 --> scale = 199/99 (not 2!)
                // E.g., 101x101 image --> 201x201 --> scale = 2
                // Test: pixel 0 --> 0, pixel 99 (ex 1) --> 199, pixel 100 (ex 2) --> 200
                topShape[2] = intRound((topShape[2]*netFactor - 1.f) * scaleFactor) + 1;
                topShape[3] = intRound((topShape[3]*netFactor - 1.f) * scaleFactor) + 1;
                topBlob->Reshape(topShape);
                // Array sizes
                mTopSize = std::array<int, 4>{topBlob->shape(0), topBlob->shape(1), topBlob->shape(2),
                                              topBlob->shape(3)};
                mBottomSizes.resize(bottom.size());
                for (auto i = 0u ; i < mBottomSizes.size() ; i++)
                    mBottomSizes[i] = std::array<int, 4>{bottom[i]->shape(0), bottom[i]->shape(1),
                                                         bottom[i]->shape(2), bottom[i]->shape(3)};
                #ifdef USE_OPENCL
                    // GPU ID
                    mGpuID = gpuID;
                    mTempGPUData.resize(mBottomSizes.size(), nullptr);
                #else
                    UNUSED(gpuID);
                #endif
            #else
                UNUSED(bottom);
                UNUSED(top);
                UNUSED(netFactor);
                UNUSED(scaleFactor);
                UNUSED(mergeFirstDimension);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::setScaleRatios(const std::vector<T>& scaleRatios)
    {
        try
        {
            mScaleRatios = {scaleRatios};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Forward(const std::vector<caffe::Blob<T>*>& bottom,
                                         const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            // CUDA
            #ifdef USE_CUDA
                Forward_gpu(bottom, top);
            // OpenCL
            #elif defined USE_OPENCL
                Forward_ocl(bottom, top);
            // CPU
            #else
                Forward_cpu(bottom, top);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Forward_cpu(const std::vector<caffe::Blob<T>*>& bottom,
                                             const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            #ifdef USE_CAFFE
                std::vector<const T*> sourcePtrs(bottom.size());
                for (auto i = 0u ; i < sourcePtrs.size() ; i++)
                    sourcePtrs[i] = bottom[i]->cpu_data();
                resizeAndMergeCpu(top.at(0)->mutable_cpu_data(), sourcePtrs, mTopSize, mBottomSizes,
                                  mScaleRatios);
            #else
                UNUSED(bottom);
                UNUSED(top);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Forward_gpu(const std::vector<caffe::Blob<T>*>& bottom,
                                             const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            #if defined USE_CAFFE && defined USE_CUDA
                std::vector<const T*> sourcePtrs(bottom.size());
                for (auto i = 0u ; i < sourcePtrs.size() ; i++)
                    sourcePtrs[i] = bottom[i]->gpu_data();
                resizeAndMergeGpu(top.at(0)->mutable_gpu_data(), sourcePtrs, mTopSize, mBottomSizes,
                                  mScaleRatios);
            #else
                UNUSED(bottom);
                UNUSED(top);
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
    void ResizeAndMergeCaffe<T>::Forward_ocl(const std::vector<caffe::Blob<T>*>& bottom,
                                             const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            #if defined USE_CAFFE && defined USE_OPENCL
                std::vector<const T*> sourcePtrs(bottom.size());
                for (auto i = 0u ; i < sourcePtrs.size() ; i++)
                    sourcePtrs[i] = bottom[i]->gpu_data();
                resizeAndMergeOcl(top.at(0)->mutable_gpu_data(), sourcePtrs, mTempGPUData, mTopSize, mBottomSizes,
                                  mScaleRatios, mGpuID);
            #else
                UNUSED(bottom);
                UNUSED(top);
                error("OpenPose must be compiled with the `USE_CAFFE` & `USE_OPENCL` macro definitions in order to run"
                      " this functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Backward_cpu(const std::vector<caffe::Blob<T>*>& top,
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
    void ResizeAndMergeCaffe<T>::Backward_gpu(const std::vector<caffe::Blob<T>*>& top,
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

    COMPILE_TEMPLATE_FLOATING_TYPES_CLASS(ResizeAndMergeCaffe);
}
