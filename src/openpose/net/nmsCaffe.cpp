#ifdef USE_CAFFE
    #include <caffe/blob.hpp>
#endif
#ifdef USE_OPENCL
    #include <openpose/gpu/opencl.hcl>
    #include <openpose/gpu/cl2.hpp>
#endif
#include <openpose/net/nmsBase.hpp>
#include <openpose/net/nmsCaffe.hpp>

namespace op
{
    template <typename T>
    struct NmsCaffe<T>::ImplNmsCaffe
    {
        #ifdef USE_CAFFE
            ArrayCpuGpu<int> mKernelBlob;
            std::array<int, 4> mBottomSize;
            std::array<int, 4> mTopSize;
            // Special Kernel for OpenCL NMS
            #if defined USE_CAFFE && defined USE_OPENCL
                //std::shared_ptr<ArrayCpuGpu<uint8_t>> mKernelBlobT;
                uint8_t* mKernelGpuPtr;
                uint8_t* mKernelCpuPtr;
            #endif
        #endif

        ImplNmsCaffe()
        {
            #if defined USE_CAFFE && defined USE_OPENCL
                try
                {
                    mKernelGpuPtr = nullptr;
                    mKernelCpuPtr = nullptr;
                }
                catch (const std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                }
            #endif
        }

        ~ImplNmsCaffe()
        {
            #if defined USE_CAFFE && defined USE_OPENCL
                try
                {
                    if(mKernelGpuPtr != nullptr)
                        clReleaseMemObject((cl_mem)mKernelGpuPtr);
                    if(mKernelCpuPtr != nullptr)
                        delete mKernelCpuPtr;
                }
                catch (const std::exception& e)
                {
                    errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
                }
            #endif
        }
    };

    template <typename T>
    NmsCaffe<T>::NmsCaffe() :
        upImpl{new ImplNmsCaffe{}}
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
    NmsCaffe<T>::~NmsCaffe()
    {
    }

    template <typename T>
    void NmsCaffe<T>::LayerSetUp(const std::vector<ArrayCpuGpu<T>*>& bottom, const std::vector<ArrayCpuGpu<T>*>& top)
    {
        try
        {
            #ifdef USE_CAFFE
                if (top.size() != 1)
                    error("top.size() != 1", __LINE__, __FUNCTION__, __FILE__);
                if (bottom.size() != 1)
                    error("bottom.size() != 1", __LINE__, __FUNCTION__, __FILE__);
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
    void NmsCaffe<T>::Reshape(const std::vector<ArrayCpuGpu<T>*>& bottom, const std::vector<ArrayCpuGpu<T>*>& top,
                              const int maxPeaks, const int outputChannels, const int gpuID)
    {
        try
        {
            #ifdef USE_CAFFE
                auto bottomBlob = bottom.at(0);
                auto topBlob = top.at(0);

                // Bottom shape
                std::vector<int> bottomShape = bottomBlob->shape();

                // Top shape
                std::vector<int> topShape{bottomShape};
                topShape[1] = (outputChannels > 0 ? outputChannels : bottomShape[1]);
                topShape[2] = maxPeaks+1; // # maxPeaks + 1
                topShape[3] = 3;  // X, Y, score
                topBlob->Reshape(topShape);
                upImpl->mKernelBlob.Reshape(bottomShape);

                // Special Kernel for OpenCL NMS
                #if defined USE_CAFFE && defined USE_OPENCL
                    int bottomShapeVolume = bottomShape[0] * bottomShape[1] * bottomShape[2] * bottomShape[3];
                    upImpl->mKernelGpuPtr = (uint8_t*)clCreateBuffer(
                        OpenCL::getInstance(gpuID)->getContext().operator()(), CL_MEM_READ_WRITE,
                        sizeof(uint8_t) * bottomShapeVolume, NULL, NULL);
                    upImpl->mKernelCpuPtr = new uint8_t[bottomShapeVolume];
                    // GPU ID
                    mGpuID = gpuID;
                #else
                    UNUSED(gpuID);
                #endif
                // Array sizes
                upImpl->mTopSize = std::array<int, 4>{topBlob->shape(0), topBlob->shape(1),
                                                      topBlob->shape(2), topBlob->shape(3)};
                upImpl->mBottomSize = std::array<int, 4>{bottomBlob->shape(0), bottomBlob->shape(1),
                                                         bottomBlob->shape(2), bottomBlob->shape(3)};
            #else
                UNUSED(bottom);
                UNUSED(top);
                UNUSED(maxPeaks);
                UNUSED(outputChannels);
                UNUSED(gpuID);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void NmsCaffe<T>::setThreshold(const T threshold)
    {
        try
        {
            mThreshold = {threshold};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void NmsCaffe<T>::setOffset(const Point<T>& offset)
    {
        try
        {
            mOffset = {offset};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void NmsCaffe<T>::Forward(const std::vector<ArrayCpuGpu<T>*>& bottom, const std::vector<ArrayCpuGpu<T>*>& top)
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
    void NmsCaffe<T>::Forward_cpu(const std::vector<ArrayCpuGpu<T>*>& bottom, const std::vector<ArrayCpuGpu<T>*>& top)
    {
        try
        {
            #ifdef USE_CAFFE
                nmsCpu(top.at(0)->mutable_cpu_data(), upImpl->mKernelBlob.mutable_cpu_data(), bottom.at(0)->cpu_data(),
                       mThreshold, upImpl->mTopSize, upImpl->mBottomSize, mOffset);
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
    void NmsCaffe<T>::Forward_gpu(const std::vector<ArrayCpuGpu<T>*>& bottom, const std::vector<ArrayCpuGpu<T>*>& top)
    {
        try
        {
            #if defined USE_CAFFE && defined USE_CUDA
                nmsGpu(top.at(0)->mutable_gpu_data(), upImpl->mKernelBlob.mutable_gpu_data(),
                       bottom.at(0)->gpu_data(), mThreshold, upImpl->mTopSize, upImpl->mBottomSize, mOffset);
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
    void NmsCaffe<T>::Forward_ocl(const std::vector<ArrayCpuGpu<T>*>& bottom, const std::vector<ArrayCpuGpu<T>*>& top)
    {
        try
        {
            #if defined USE_CAFFE && defined USE_OPENCL
                nmsOcl(top.at(0)->mutable_gpu_data(), upImpl->mKernelGpuPtr, upImpl->mKernelCpuPtr,
                       bottom.at(0)->gpu_data(), mThreshold, upImpl->mTopSize, upImpl->mBottomSize, mOffset,
                       mGpuID);
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
    void NmsCaffe<T>::Backward_cpu(const std::vector<ArrayCpuGpu<T>*>& top, const std::vector<bool>& propagate_down,
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
    void NmsCaffe<T>::Backward_gpu(const std::vector<ArrayCpuGpu<T>*>& top, const std::vector<bool>& propagate_down,
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

    COMPILE_TEMPLATE_FLOATING_TYPES_CLASS(NmsCaffe);
}
