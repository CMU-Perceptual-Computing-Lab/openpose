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
                                         const T offsetX, const T offsetY)
            {
                Type xAcc = 0.f;
                Type yAcc = 0.f;
                Type scoreAcc = 0.f;
                const int dWidth = 3;
                const int dHeight = 3;
                for (auto dy = -dHeight ; dy <= dHeight ; dy++)
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

        typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int> NMSWriteKernelFunctor;
        const std::string nmsWriteKernel = MULTI_LINE_STRING(
            __kernel void nmsWriteKernel(__global Type* targetPtr, __global int* kernelPtr, __global const Type* sourcePtr,
                                         const int w, const int h, const int maxPeaks, const int debug,
                                         const T offsetX, const T offsetY)
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
    #endif

    template <typename T>
    void nmsOcl(T* targetPtr, int* kernelPtr, const T* const sourcePtr, const T threshold,
                const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize, const Point<T>& offset,
                const int gpuID)
    {
        try
        {
            #ifdef USE_OPENCL
                // Security checks
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

                // Get Kernel
                cl::Buffer sourcePtrBuffer = cl::Buffer((cl_mem)(sourcePtr), true);
                cl::Buffer kernelPtrBuffer = cl::Buffer((cl_mem)(kernelPtr), true);
                cl::Buffer targetPtrBuffer = cl::Buffer((cl_mem)(targetPtr), true);
                auto nmsRegisterKernel = op::OpenCL::getInstance(gpuID)->getKernelFunctorFromManager
                        <op::NMSRegisterKernelFunctor, T>(
                            "nmsRegisterKernel",op::nmsOclCommonFunctions + op::nmsRegisterKernel);
                auto nmsWriteKernel = op::OpenCL::getInstance(gpuID)->getKernelFunctorFromManager
                        <op::NMSWriteKernelFunctor, T>(
                            "nmsWriteKernel",op::nmsOclCommonFunctions + op::nmsWriteKernel);

                // log("num_b: " + std::to_string(bottom->shape(0)));       // = 1
                // log("channel_b: " + std::to_string(bottom->shape(1)));   // = 57 = 18 body parts + bkg + 19x2 PAFs
                // log("height_b: " + std::to_string(bottom->shape(2)));    // = 368 = height
                // log("width_b: " + std::to_string(bottom->shape(3)));     // = 656 = width
                // log("num_t: " + std::to_string(top->shape(0)));       // = 1
                // log("channel_t: " + std::to_string(top->shape(1)));   // = 18 = numberParts
                // log("height_t: " + std::to_string(top->shape(2)));    // = 97 = maxPeople + 1
                // log("width_t: " + std::to_string(top->shape(3)));     // = 3 = [x, y, score]
                // log("");

                // Temp DS
                //cv::Mat kernelCPU(cv::Size(width, height),CV_32FC1,cv::Scalar(0));
                std::vector<int> kernelCPU(imageOffset);
                for (auto n = 0; n < num; n++)
                {
                    for (auto c = 0; c < channels; c++)
                    {
                        // log("channel: " + std::to_string(c));
                        const auto offsetChannel = (n * channels + c);

                        // CL Data
                        cl_buffer_region kernelRegion, sourceRegion, targetRegion;
                        kernelRegion.origin = sizeof(int) * offsetChannel * imageOffset;
                        kernelRegion.size = sizeof(int) * imageOffset;
                        cl::Buffer kernelBuffer = kernelPtrBuffer.createSubBuffer(CL_MEM_READ_WRITE,
                                                                                  CL_BUFFER_CREATE_TYPE_REGION,
                                                                                  &kernelRegion);
                        op::OpenCL::getBufferRegion<T>(sourceRegion, offsetChannel * imageOffset, imageOffset);
                        op::OpenCL::getBufferRegion<T>(targetRegion, offsetChannel * targetChannelOffset, targetChannelOffset);
                        cl::Buffer sourceBuffer = sourcePtrBuffer.createSubBuffer(CL_MEM_READ_ONLY,
                                                                                  CL_BUFFER_CREATE_TYPE_REGION,
                                                                                  &sourceRegion);
                        cl::Buffer targetBuffer = targetPtrBuffer.createSubBuffer(CL_MEM_READ_WRITE,
                                                                                  CL_BUFFER_CREATE_TYPE_REGION,
                                                                                  &targetRegion);

                        // Run Kernel to get 1-0 map
                        bool debug = false;
                        nmsRegisterKernel(cl::EnqueueArgs(op::OpenCL::getInstance(gpuID)->getQueue(), cl::NDRange(width, height)),
                                          kernelBuffer, sourceBuffer, width, height, threshold, debug);
                        // This is a really bad approach. We need to write a custom accumalator to run on gpu
                        // Download it to CPU
                        op::OpenCL::getInstance(gpuID)->getQueue().enqueueReadBuffer(kernelBuffer, CL_TRUE, 0,
                                                                                     sizeof(int) *  width * height, &kernelCPU[0]);
                        // Compute partial sum in CPU
                        std::partial_sum(kernelCPU.begin(),kernelCPU.end(),kernelCPU.begin());
                        // Reupload to GPU
                        op::OpenCL::getInstance(gpuID)->getQueue().enqueueWriteBuffer(kernelBuffer, CL_TRUE, 0,
                                                                                      sizeof(int) *  width * height, &kernelCPU[0]);
                        // Write Kernel
                        nmsWriteKernel(cl::EnqueueArgs(op::OpenCL::getInstance(gpuID)->getQueue(), cl::NDRange(width, height)),
                                          targetBuffer, kernelBuffer, sourceBuffer, width, height, targetPeaks-1, debug,
                                          offset.x, offset.y);
                    }
                }
            #else
                UNUSED(targetPtr);
                UNUSED(kernelPtr);
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
        #ifdef USE_OPENCL
            catch (const cl::Error& e)
            {
                error(std::string(e.what()) + " : " + op::OpenCL::clErrorToString(e.err()) + " ID: " +
                      std::to_string(gpuID), __LINE__, __FUNCTION__, __FILE__);
            }
        #endif
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void nmsOcl(float* targetPtr, int* kernelPtr, const float* const sourcePtr, const float threshold,
                         const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize,
                         const Point<float>& offset, const int gpuID);
    template void nmsOcl(double* targetPtr, int* kernelPtr, const double* const sourcePtr, const double threshold,
                         const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize,
                         const Point<double>& offset, const int gpuID);
}
