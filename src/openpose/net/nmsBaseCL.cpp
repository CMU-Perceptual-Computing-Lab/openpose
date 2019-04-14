#include <algorithm>
#include <bitset>
#include <numeric>
#include <opencv2/opencv.hpp>
#ifdef USE_OPENCL
    #include <openpose/gpu/opencl.hcl>
    #include <openpose/gpu/cl2.hpp>
#endif
#include <openpose/core/common.hpp>
#include <openpose/net/nmsBase.hpp>

namespace op
{
    #ifdef USE_OPENCL
        const std::string nmsOclCommonFunctions = MULTI_LINE_STRING(
            void nmsAccuratePeakPosition(__global const Type* sourcePtr, Type* fx, Type* fy, Type* fscore,
                                         const int peakLocX, const int peakLocY, const int width, const int height,
                                         const Type offsetX, const Type offsetY)
            {
                Type xAcc = 0.f;
                Type yAcc = 0.f;
                Type scoreAcc = 0.f;
                const int dWidth = 3;
                const int dHeight = 3;
                for (int dy = -dHeight ; dy <= dHeight ; dy++)
                {
                    const int y = peakLocY + dy;
                    if (0 <= y && y < height) // Default height = 368
                    {
                        for (int dx = -dWidth ; dx <= dWidth ; dx++)
                        {
                            const int x = peakLocX + dx;
                            if (0 <= x && x < width) // Default width = 656
                            {
                                const Type score = sourcePtr[y * width + x];
                                if (score > 0)
                                {
                                    xAcc += (Type)x*score;
                                    yAcc += (Type)y*score;
                                    scoreAcc += score;
                                }
                            }
                        }
                    }
                }

                // Offset to keep Matlab format (empirically higher acc)
                // Best results for 1 scale: x + 0, y + 0.5
                // +0.5 to both to keep Matlab format
                *fx = xAcc / scoreAcc + offsetX;
                *fy = yAcc / scoreAcc + offsetY;
                *fscore = sourcePtr[peakLocY*width + peakLocX];
            }

            union DS {
              struct {
                short x;
                short y;
                float score;
              } ds;
              double dbl;
            };
        );

        typedef cl::KernelFunctor<cl::Buffer, int, int> PartialSumKernelFunctor;
        const std::string partialSumKernel = MULTI_LINE_STRING(
            __kernel void partialSumKernel(__global int* kernelFullPtr,
                                               const int w, const int h)
            {
                int c = get_global_id(0);
                __global int* kernelPtr = kernelFullPtr + (c*w*h);

                int incr = 0;
                for(int y=0; y<h; y++){
                    for(int x=0; x<w; x++){
                        int index = y*w + x;
                        if(kernelPtr[index]) incr += 1;
                        kernelPtr[index] = incr;
                    }
                }

            }
        );

        typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, int, int, float, int> NMSFullRegisterKernelFunctor;
        const std::string nmsFullRegisterKernel = MULTI_LINE_STRING(
            __kernel void nmsFullRegisterKernel(__global uchar* kernelFullPtr, __global const Type* sourceFullPtr,
                                               const int w, const int h, const Type threshold, const int debug)
            {
                int c = get_global_id(0);
                int x = get_global_id(1);
                int y = get_global_id(2);

                __global const Type* sourcePtr = sourceFullPtr + (c*w*h);
                __global uchar* kernelPtr = kernelFullPtr + (c*w*h);
                int index = y*w + x;

                if (0 < x && x < (w-1) && 0 < y && y < (h-1))
                {
                    const Type value = sourcePtr[index];

                    if (value > threshold)
                    {

                        const Type topLeft     = sourcePtr[(y-1)*w + x-1];
                        const Type top         = sourcePtr[(y-1)*w + x];
                        const Type topRight    = sourcePtr[(y-1)*w + x+1];
                        const Type left        = sourcePtr[    y*w + x-1];
                        const Type right       = sourcePtr[    y*w + x+1];
                        const Type bottomLeft  = sourcePtr[(y+1)*w + x-1];
                        const Type bottom      = sourcePtr[(y+1)*w + x];
                        const Type bottomRight = sourcePtr[(y+1)*w + x+1];

                        if (value > topLeft && value > top && value > topRight
                            && value > left && value > right
                            && value > bottomLeft && value > bottom && value > bottomRight)
                        {
                            kernelPtr[index] = 1;
                        }
                        else
                            kernelPtr[index] = 0;
                    }
                    else
                        kernelPtr[index] = 0;
                }
                else if (x == 0 || x == (w-1) || y == 0 || y == (h-1))
                    kernelPtr[index] = 0;
            }
        );

        typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, int, int, float, int> NMSRegisterKernelFunctor;
        const std::string nmsRegisterKernel = MULTI_LINE_STRING(
            __kernel void nmsRegisterKernel(__global int* kernelPtr, __global const Type* sourcePtr,
                                               const int w, const int h, const Type threshold, const int debug)
            {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int index = y*w + x;      

                if (0 < x && x < (w-1) && 0 < y && y < (h-1))
                {
                    const Type value = sourcePtr[index];
                    if (value > threshold)
                    {
                        const Type topLeft     = sourcePtr[(y-1)*w + x-1];
                        const Type top         = sourcePtr[(y-1)*w + x];
                        const Type topRight    = sourcePtr[(y-1)*w + x+1];
                        const Type left        = sourcePtr[    y*w + x-1];
                        const Type right       = sourcePtr[    y*w + x+1];
                        const Type bottomLeft  = sourcePtr[(y+1)*w + x-1];
                        const Type bottom      = sourcePtr[(y+1)*w + x];
                        const Type bottomRight = sourcePtr[(y+1)*w + x+1];

                        if (value > topLeft && value > top && value > topRight
                            && value > left && value > right
                            && value > bottomLeft && value > bottom && value > bottomRight)
                        {
                            kernelPtr[index] = 1;
                        }
                        else
                            kernelPtr[index] = 0;
                    }
                    else
                        kernelPtr[index] = 0;
                }
                else if (x == 0 || x == (w-1) || y == 0 || y == (h-1))
                    kernelPtr[index] = 0;
            }
        );

        typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int, float, float> NMSWriteKernelFunctor;
        const std::string nmsWriteKernel = MULTI_LINE_STRING(
            __kernel void nmsWriteKernel(__global Type* targetPtr, __global int* kernelPtr, __global const Type* sourcePtr,
                                         const int w, const int h, const int maxPeaks, const int debug,
                                         const Type offsetX, const Type offsetY)
            {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int index = y*w + x;

                if (index != 0){
                    int prev = kernelPtr[index-1];
                    int curr = kernelPtr[index];
                    if (curr < maxPeaks)
                    {
                        if (prev - curr)
                        {
                            Type fx = 0; Type fy = 0; Type fscore = 0;
                            nmsAccuratePeakPosition(sourcePtr, &fx, &fy, &fscore, x, y, w, h, offsetX, offsetY);
                            //if (debug) printf("C %d %d %d \n", x,y,kernelPtr[index]);
                            __global Type* output = &targetPtr[curr*3];
                            output[0] = fx; output[1] = fy; output[2] = fscore;
                        }
                        if (index + 1 == w*h)
                        {
                            __global Type* output = &targetPtr[0*3];
                            output[0] = curr;
                        }
                    }
                    else
                    {
                        if (index + 1 == w*h)
                        {
                            __global Type* output = &targetPtr[0*3];
                            output[0] = maxPeaks;
                        }
                    }
                }
            }
        );

        typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int, float, float> NMSFullWriteKernelFunctor;
        const std::string nmsFullWriteKernel = MULTI_LINE_STRING(
            __kernel void nmsFullWriteKernel(__global Type* targetPtrFull, __global uchar* kernelPtrFull, __global const Type* sourcePtrFull,
                                         const int w, const int h, const int maxPeaks, const int debug,
                                         const Type offsetX, const Type offsetY)
            {
                int c = get_global_id(0);
                int x = get_global_id(1);
                int y = get_global_id(2);

                __global Type* targetPtr = targetPtrFull + c*(maxPeaks+1)*3;
                __global uchar* kernelPtr = kernelPtrFull + c*w*h;
                __global const Type* sourcePtr = sourcePtrFull + c*w*h;

                int index = y*w + x;

                if (index != 0){
                    uchar prev = kernelPtr[index-1];
                    uchar curr = kernelPtr[index];
                    if (curr < maxPeaks)
                    {
                        if (prev - curr)
                        {
                            Type fx = 0; Type fy = 0; Type fscore = 0;
                            nmsAccuratePeakPosition(sourcePtr, &fx, &fy, &fscore, x, y, w, h, offsetX, offsetY);
                            //printf("C %d %d %d \n", x,y,kernelPtr[index]);
                            __global Type* output = &targetPtr[curr*3];
                            output[0] = fx; output[1] = fy; output[2] = fscore;
                        }
                        if (index + 1 == w*h)
                        {
                            __global Type* output = &targetPtr[0*3];
                            output[0] = curr;
                        }
                    }
                    else
                    {
                        if (index + 1 == w*h)
                        {
                            __global Type* output = &targetPtr[0*3];
                            output[0] = maxPeaks;
                        }
                    }
                }
            }
        );
    #endif

    template <typename T>
    void nmsOcl(T* targetPtr, uint8_t* kernelGpuPtr, uint8_t* kernelCpuPtr, const T* const sourcePtr, const T threshold,
                const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize, const Point<T>& offset,
                const int gpuID)
    {
        try
        {
            #ifdef USE_OPENCL
                // Sanity checks
                if (sourceSize.empty())
                    error("sourceSize cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
                if (targetSize.empty())
                    error("targetSize cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
                if (threshold < 0 || threshold > 1.0)
                    error("threshold value invalid.", __LINE__, __FUNCTION__, __FILE__);

                //Forward_cpu(bottom, top);
                const auto num = sourceSize[0];
                const auto height = sourceSize[2];
                const auto width = sourceSize[3];
                const auto channels = targetSize[1];
                const auto targetPeaks = targetSize[2]; // 97
                const auto targetPeakVec = targetSize[3]; // 3
                const auto imageOffset = height * width;
                const auto targetChannelOffset = targetPeaks * targetPeakVec;

                //std::cout << targetPeaks << std::endl;
                //std::cout << targetPeakVec << std::endl;

                // Get Kernel
                cl::Buffer sourcePtrBuffer = cl::Buffer((cl_mem)(sourcePtr), true);
                cl::Buffer kernelPtrBuffer = cl::Buffer((cl_mem)(kernelGpuPtr), true);
                cl::Buffer targetPtrBuffer = cl::Buffer((cl_mem)(targetPtr), true);
                auto nmsRegisterKernel = OpenCL::getInstance(gpuID)->getKernelFunctorFromManager
                        <NMSRegisterKernelFunctor, T>(
                         "nmsRegisterKernel", op::nmsOclCommonFunctions + op::nmsRegisterKernel);
                auto nmsFullRegisterKernel = OpenCL::getInstance(gpuID)->getKernelFunctorFromManager
                        <NMSFullRegisterKernelFunctor, T>(
                         "nmsFullRegisterKernel", op::nmsOclCommonFunctions + op::nmsFullRegisterKernel);
                auto nmsWriteKernel = OpenCL::getInstance(gpuID)->getKernelFunctorFromManager
                        <NMSWriteKernelFunctor, T>(
                         "nmsWriteKernel", op::nmsOclCommonFunctions + op::nmsWriteKernel);
                auto partialSumKernel = OpenCL::getInstance(gpuID)->getKernelFunctorFromManager
                        <PartialSumKernelFunctor, T>(
                         "partialSumKernel", op::nmsOclCommonFunctions + op::partialSumKernel);
                auto nmsFullWriteKernel = OpenCL::getInstance(gpuID)->getKernelFunctorFromManager
                        <NMSFullWriteKernelFunctor, T>(
                         "nmsFullWriteKernel", op::nmsOclCommonFunctions + op::nmsFullWriteKernel);

                // Temp DS
                for (auto n = 0; n < num; n++)
                {
                    nmsFullRegisterKernel(cl::EnqueueArgs(OpenCL::getInstance(gpuID)->getQueue(), cl::NDRange((int)channels, (int)width, (int)height)),
                                      kernelPtrBuffer, sourcePtrBuffer, width, height, (float)threshold, false);
                    OpenCL::getInstance(gpuID)->getQueue().enqueueReadBuffer(kernelPtrBuffer, CL_TRUE, 0,
                                                                                 sizeof(uint8_t) * channels * imageOffset, &kernelCpuPtr[0]);
                    for(int c=0; c<channels; c++){
                        uint8_t* currPtr = kernelCpuPtr + c*imageOffset;
                        std::partial_sum(currPtr,currPtr + imageOffset,currPtr);
                    }
                    OpenCL::getInstance(gpuID)->getQueue().enqueueWriteBuffer(kernelPtrBuffer, CL_TRUE, 0,
                                                                                  sizeof(uint8_t) * channels * imageOffset, &kernelCpuPtr[0]);
                    nmsFullWriteKernel(cl::EnqueueArgs(OpenCL::getInstance(gpuID)->getQueue(), cl::NDRange(channels, width, height)),
                                      targetPtrBuffer, kernelPtrBuffer, sourcePtrBuffer, width, height, targetPeaks-1, false,
                                      offset.x, offset.y);
                }
            #else
                UNUSED(targetPtr);
                UNUSED(kernelGpuPtr);
                UNUSED(kernelCpuPtr);
                UNUSED(sourcePtr);
                UNUSED(threshold);
                UNUSED(targetSize);
                UNUSED(sourceSize);
                UNUSED(offset);
                UNUSED(gpuID);
                error("OpenPose must be compiled with the `USE_OPENCL` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        #if defined(USE_OPENCL) && defined(CL_HPP_ENABLE_EXCEPTIONS)
        catch (const cl::Error& e)
        {
            error(std::string(e.what()) + " : " + OpenCL::clErrorToString(e.err()) + " ID: " +
                  std::to_string(gpuID), __LINE__, __FUNCTION__, __FILE__);
        }
        #endif
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void nmsOcl(
        float* targetPtr, uint8_t* kernelGpuPtr, uint8_t* kernelCpuPtr, const float* const sourcePtr, const float threshold,
        const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize, const Point<float>& offset,
        const int gpuID);
    template void nmsOcl(
        double* targetPtr, uint8_t* kernelGpuPtr, uint8_t* kernelCpuPtr, const double* const sourcePtr, const double threshold,
        const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize, const Point<double>& offset,
        const int gpuID);
}

//                    for (auto c = 0; c < channels; c++)
//                    {
//                        // log("channel: " + std::to_string(c));
//                        const auto offsetChannel = (n * channels + c);

//                        // CL Data
//                        cl_buffer_region kernelRegion, sourceRegion, targetRegion;
//                        kernelRegion.origin = sizeof(int) * offsetChannel * imageOffset;
//                        kernelRegion.size = sizeof(int) * imageOffset;
//                        cl::Buffer kernelBuffer = kernelPtrBuffer.createSubBuffer(CL_MEM_READ_WRITE,
//                                                                                  CL_BUFFER_CREATE_TYPE_REGION,
//                                                                                  &kernelRegion);
//                        OpenCL::getBufferRegion<T>(sourceRegion, offsetChannel * imageOffset, imageOffset);
//                        OpenCL::getBufferRegion<T>(targetRegion, offsetChannel * targetChannelOffset, targetChannelOffset);
//                        cl::Buffer sourceBuffer = sourcePtrBuffer.createSubBuffer(CL_MEM_READ_ONLY,
//                                                                                  CL_BUFFER_CREATE_TYPE_REGION,
//                                                                                  &sourceRegion);
//                        cl::Buffer targetBuffer = targetPtrBuffer.createSubBuffer(CL_MEM_READ_WRITE,
//                                                                                  CL_BUFFER_CREATE_TYPE_REGION,
//                                                                                  &targetRegion);

//                        // Run Kernel to get 1-0 map
//                        bool debug = false;
////                        nmsRegisterKernel(cl::EnqueueArgs(OpenCL::getInstance(gpuID)->getQueue(), cl::NDRange(width, height)),
////                                          kernelBuffer, sourceBuffer, width, height, threshold, debug);
//                        // This is a really bad approach. We need to write a custom accumalator to run on gpu
//                        // Download it to CPU
////                        OpenCL::getInstance(gpuID)->getQueue().enqueueReadBuffer(kernelBuffer, CL_TRUE, 0,
////                                                                                     sizeof(int) *  width * height, &kernelCPU[0]);
////                        // Compute partial sum in CPU
////                        std::partial_sum(kernelCPU.begin(),kernelCPU.end(),kernelCPU.begin());
////                        // Reupload to GPU
////                        OpenCL::getInstance(gpuID)->getQueue().enqueueWriteBuffer(kernelBuffer, CL_TRUE, 0,
////                                                                                      sizeof(int) *  width * height, &kernelCPU[0]);

//                        // Write Kernel
//                        nmsWriteKernel(cl::EnqueueArgs(OpenCL::getInstance(gpuID)->getQueue(), cl::NDRange(width, height)),
//                                          targetBuffer, kernelBuffer, sourceBuffer, width, height, targetPeaks-1, false,
//                                          offset.x, offset.y);
//                    }
