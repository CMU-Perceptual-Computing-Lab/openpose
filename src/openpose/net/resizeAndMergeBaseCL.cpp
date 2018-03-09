#ifdef USE_OPENCL
    #include <openpose/gpu/opencl.hcl>
    #include <openpose/gpu/cl2.hpp>
#endif
#include <openpose/core/common.hpp>
#include <openpose/net/resizeAndMergeBase.hpp>

namespace op
{
    #ifdef USE_OPENCL
        const std::string resizeAndMergeOclCommonFunctions = MULTI_LINE_STRING(
            // Max/min functions
            int fastMax(int a, int b)
            {
                return (a > b ? a : b);
            }

            int fastMin(int a, int b)
            {
                return (a < b ? a : b);
            }

            int fastTruncate(int value, int min, int max)
            {
                return fastMin(max, fastMax(min, value));
            }

            void cubicSequentialData(int* xIntArray, int* yIntArray, Type* dx, Type* dy, const Type xSource,
                                     const Type ySource, const int width, const int height)
            {
                xIntArray[1] = fastTruncate((int)(xSource + 1e-5), 0, width - 1);
                xIntArray[0] = fastMax(0, xIntArray[1] - 1);
                xIntArray[2] = fastMin(width - 1, xIntArray[1] + 1);
                xIntArray[3] = fastMin(width - 1, xIntArray[2] + 1);
                *dx = xSource - xIntArray[1];

                yIntArray[1] = fastTruncate((int)(ySource + 1e-5), 0, height - 1);
                yIntArray[0] = fastMax(0, yIntArray[1] - 1);
                yIntArray[2] = fastMin(height - 1, yIntArray[1] + 1);
                yIntArray[3] = fastMin(height - 1, yIntArray[2] + 1);
                *dy = ySource - yIntArray[1];
            }

            Type cubicInterpolate(const Type v0, const Type v1, const Type v2, const Type v3, const Type dx)
            {
                // http://www.paulinternet.nl/?page=bicubic
                // const auto a = (-0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3);
                // const auto b = (v0 - 2.5f * v1 + 2.0 * v2 - 0.5 * v3);
                // const auto c = (-0.5f * v0 + 0.5f * v2);
                // out = ((a * dx + b) * dx + c) * dx + v1;
                return (-0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3) * dx * dx * dx
                        + (v0 - 2.5f * v1 + 2.f * v2 - 0.5f * v3) * dx * dx
                        - 0.5f * (v0 - v2) * dx // + (-0.5f * v0 + 0.5f * v2) * dx
                        + v1;
                // return v1 + 0.5f * dx * (v2 - v0 + dx * (2.f * v0 - 5.f * v1 + 4.f * v2 - v3 + dx * (3.f * (v1 - v2) + v3 - v0)));
            }

            Type bicubicInterpolate(__global const Type* sourcePtr, const Type xSource, const Type ySource, const int widthSource,
                                                   const int heightSource, const int widthSourcePtr)
            {
                int xIntArray[4];
                int yIntArray[4];
                Type dx;
                Type dy;
                cubicSequentialData(xIntArray, yIntArray, &dx, &dy, xSource, ySource, widthSource, heightSource);

                Type temp[4];
                for (unsigned char i = 0; i < 4; i++)
                {
                    const auto offset = yIntArray[i]*widthSourcePtr;
                    temp[i] = cubicInterpolate(sourcePtr[offset + xIntArray[0]], sourcePtr[offset + xIntArray[1]],
                                               sourcePtr[offset + xIntArray[2]], sourcePtr[offset + xIntArray[3]], dx);
                }
                return cubicInterpolate(temp[0], temp[1], temp[2], temp[3], dy);
            }
        );

        typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, int, int, int, int> ResizeAndMergeFullFunctor;
        const std::string resizeAndMergeFullKernel = MULTI_LINE_STRING(
            __kernel void resizeAndMergeFullKernel(__global Type* targetPtr, __global const Type* sourcePtr,
                                               const int sourceWidth, const int sourceHeight,
                                               const int targetWidth, const int targetHeight)
            {
                int c = get_global_id(0);
                int y = get_global_id(1);
                int x = get_global_id(2);

                Type* targetPtrC = &targetPtr[c*targetWidth*targetHeight];
                const Type* sourcePtrC = &sourcePtr[c*sourceWidth*sourceHeight];

                if (x < targetWidth && y < targetHeight)
                {
                    const Type xSource = (x + 0.5f) * sourceWidth / (Type)targetWidth - 0.5f;
                    const Type ySource = (y + 0.5f) * sourceHeight / (Type)targetHeight - 0.5f;
                    targetPtrC[y*targetWidth+x] = bicubicInterpolate(sourcePtrC, xSource, ySource, sourceWidth,
                            sourceHeight, sourceWidth);
                }
            }
        );

        typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, int, int, int, int> ResizeAndMergeFunctor;
        const std::string resizeAndMergeKernel = MULTI_LINE_STRING(
            __kernel void resizeAndMergeKernel(__global Type* targetPtr, __global const Type* sourcePtr,
                                               const int sourceWidth, const int sourceHeight,
                                               const int targetWidth, const int targetHeight)
            {
                int x = get_global_id(0);
                int y = get_global_id(1);

                if (x < targetWidth && y < targetHeight)
                {
                    const Type xSource = (x + 0.5f) * sourceWidth / (Type)targetWidth - 0.5f;
                    const Type ySource = (y + 0.5f) * sourceHeight / (Type)targetHeight - 0.5f;
                    targetPtr[y*targetWidth+x] = bicubicInterpolate(sourcePtr, xSource, ySource, sourceWidth,
                            sourceHeight, sourceWidth);
                }
            }
        );

        typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, float, float, int, int, int, int> ResizeAndAddFunctor;
        const std::string resizeAndAddKernel = MULTI_LINE_STRING(
            __kernel void resizeAndAddKernel(__global Type* targetPtr, __global const Type* sourcePtr,
                                               const Type scaleWidth, const Type scaleHeight,
                                               const int sourceWidth, const int sourceHeight,
                                               const int targetWidth, const int targetHeight)
            {
                int x = get_global_id(0);
                int y = get_global_id(1);

                if (x < targetWidth && y < targetHeight)
                {
                    const Type xSource = (x + 0.5f) / scaleWidth - 0.5f;
                    const Type ySource = (y + 0.5f) / scaleHeight - 0.5f;
                    targetPtr[y*targetWidth+x] += bicubicInterpolate(sourcePtr, xSource, ySource, sourceWidth,
                            sourceHeight, sourceWidth);
                }
            }
        );

        typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, float, float, int, int, int, int, int> ResizeAndAverageFunctor;
        const std::string resizeAndAverageKernel = MULTI_LINE_STRING(
            __kernel void resizeAndAverageKernel(__global Type* targetPtr, __global const Type* sourcePtr,
                                               const Type scaleWidth, const Type scaleHeight,
                                               const int sourceWidth, const int sourceHeight,
                                               const int targetWidth, const int targetHeight, const int counter)
            {
                int x = get_global_id(0);
                int y = get_global_id(1);

                if (x < targetWidth && y < targetHeight)
                {
                    const Type xSource = (x + 0.5f) / scaleWidth - 0.5f;
                    const Type ySource = (y + 0.5f) / scaleHeight - 0.5f;
                    Type interpolated = bicubicInterpolate(sourcePtr, xSource, ySource, sourceWidth, sourceHeight, sourceWidth);
                    __global Type* targetPixel = &targetPtr[y*targetWidth+x];
                    *targetPixel = (*targetPixel + interpolated) / (Type)(counter);
                }
            }
        );

        typedef cl::KernelFunctor<cl::Buffer, int, int> ZeroBufferFunctor;
        const std::string zeroBufferKernel = MULTI_LINE_STRING(
            __kernel void zeroBufferKernel(__global Type* targetPtr, const int targetWidth, const int targetHeight)
            {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int c = get_global_id(2);

                __global Type* targetPtrC = &targetPtr[c*targetWidth*targetHeight];
                targetPtrC[y*targetWidth+x] = 0;
            }
        );
    #endif

    template <typename T>
    void resizeAndMergeOcl(T* targetPtr, const std::vector<const T*>& sourcePtrs,
                           const std::array<int, 4>& targetSize,
                           const std::vector<std::array<int, 4>>& sourceSizes,
                           const std::vector<T>& scaleInputToNetInputs,
                           const int gpuID)
    {
        try
        {
            #ifdef USE_OPENCL
                // Security checks
                if (sourceSizes.empty())
                    error("sourceSizes cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
                if (sourcePtrs.size() != sourceSizes.size() || sourceSizes.size() != scaleInputToNetInputs.size())
                    error("Size(sourcePtrs) must match size(sourceSizes) and size(scaleInputToNetInputs). Currently: "
                          + std::to_string(sourcePtrs.size()) + " vs. " + std::to_string(sourceSizes.size()) + " vs. "
                          + std::to_string(scaleInputToNetInputs.size()) + ".", __LINE__, __FUNCTION__, __FILE__);

                // Get Kernels
                cl::Buffer targetPtrBuffer = cl::Buffer((cl_mem)(targetPtr), true);
                auto resizeAndMergeKernel = op::OpenCL::getInstance(gpuID)->getKernelFunctorFromManager
                        <op::ResizeAndMergeFunctor, T>(
                            "resizeAndMergeKernel",op::resizeAndMergeOclCommonFunctions+op::resizeAndMergeKernel);
                auto resizeAndAddKernel = op::OpenCL::getInstance(gpuID)->getKernelFunctorFromManager
                        <op::ResizeAndAddFunctor, T>(
                            "resizeAndAddKernel",op::resizeAndMergeOclCommonFunctions+op::resizeAndAddKernel);
                auto resizeAndAverageKernel = op::OpenCL::getInstance(gpuID)->getKernelFunctorFromManager
                        <op::ResizeAndAverageFunctor, T>(
                            "resizeAndAverageKernel",op::resizeAndMergeOclCommonFunctions+op::resizeAndAverageKernel);
                auto zeroBufferKernel = op::OpenCL::getInstance(gpuID)->getKernelFunctorFromManager
                        <op::ZeroBufferFunctor, T>(
                            "zeroBufferKernel",op::zeroBufferKernel);

                // Parameters
                const auto channels = targetSize[1];
                const auto targetHeight = targetSize[2];
                const auto targetWidth = targetSize[3];
                const auto& sourceSize = sourceSizes[0];
                const auto sourceHeight = sourceSize[2];
                const auto sourceWidth = sourceSize[3];

                // No multi-scale merging or no merging required
                if (sourceSizes.size() == 10)
                {
                    const auto num = sourceSize[0];
                    if (targetSize[0] > 1 || num == 1)
                    {
                        cl::Buffer sourcePtrBuffer = cl::Buffer((cl_mem)(sourcePtrs.at(0)), true);
                        const auto sourceChannelOffset = sourceHeight * sourceWidth;
                        const auto targetChannelOffset = targetWidth * targetHeight;
                        for (auto n = 0; n < num; n++)
                        {
                            const auto offsetBase = n*channels;
                            for (auto c = 0 ; c < channels ; c++)
                            {
                                const auto offset = offsetBase + c;
                                cl_buffer_region targerRegion, sourceRegion;
                                op::OpenCL::getBufferRegion<T>(targerRegion, offset * targetChannelOffset, targetChannelOffset);
                                cl::Buffer targetBuffer = targetPtrBuffer.createSubBuffer(CL_MEM_READ_WRITE,
                                                                                          CL_BUFFER_CREATE_TYPE_REGION,
                                                                                          &targerRegion);
                                op::OpenCL::getBufferRegion<T>(sourceRegion, offset * sourceChannelOffset, sourceChannelOffset);
                                cl::Buffer sourceBuffer = sourcePtrBuffer.createSubBuffer(CL_MEM_READ_ONLY,
                                                                                          CL_BUFFER_CREATE_TYPE_REGION,
                                                                                          &sourceRegion);
                                resizeAndMergeKernel(cl::EnqueueArgs(op::OpenCL::getInstance(gpuID)->getQueue(),
                                                     cl::NDRange(targetWidth, targetHeight)),
                                                     targetBuffer, sourceBuffer,
                                                     sourceWidth, sourceHeight, targetWidth, targetHeight);
                            }
                        }
                    }
                    // Old inefficient multi-scale merging
                    else
                        error("It should never reache this point. Notify us otherwise.", __LINE__, __FUNCTION__, __FILE__);
                }
                // Multi-scaling merging
                else
                {
                    const auto targetChannelOffset = targetWidth * targetHeight;
                    //cudaMemset(targetPtr, 0.f, channels*targetChannelOffset * sizeof(T));
                    zeroBufferKernel(cl::EnqueueArgs(op::OpenCL::getInstance(gpuID)->getQueue(),
                                                     cl::NDRange(targetWidth, targetHeight, channels)),
                                                     targetPtrBuffer, targetWidth, targetHeight);
                    const auto scaleToMainScaleWidth = targetWidth / T(sourceWidth);
                    const auto scaleToMainScaleHeight = targetHeight / T(sourceHeight);

                    for (auto i = 0u ; i < sourceSizes.size(); i++)
                    {
                        const auto& currentSize = sourceSizes.at(i);
                        const auto currentHeight = currentSize[2];
                        const auto currentWidth = currentSize[3];
                        const auto sourceChannelOffset = currentHeight * currentWidth;
                        const auto scaleInputToNet = scaleInputToNetInputs[i] / scaleInputToNetInputs[0];
                        const auto scaleWidth = scaleToMainScaleWidth / scaleInputToNet;
                        const auto scaleHeight = scaleToMainScaleHeight / scaleInputToNet;
                        cl::Buffer sourcePtrBuffer = cl::Buffer((cl_mem)(sourcePtrs.at(i)), true);
                        // All but last image --> add
                        if (i < sourceSizes.size() - 1)
                        {
                            for (auto c = 0 ; c < channels ; c++)
                            {
                                cl_buffer_region targerRegion, sourceRegion;
                                op::OpenCL::getBufferRegion<T>(targerRegion, c * targetChannelOffset, targetChannelOffset);
                                op::OpenCL::getBufferRegion<T>(sourceRegion, c * sourceChannelOffset, sourceChannelOffset);
                                cl::Buffer targetBuffer = targetPtrBuffer.createSubBuffer(CL_MEM_READ_WRITE,
                                                                                          CL_BUFFER_CREATE_TYPE_REGION,
                                                                                          &targerRegion);
                                cl::Buffer sourceBuffer = sourcePtrBuffer.createSubBuffer(CL_MEM_READ_WRITE,
                                                                                          CL_BUFFER_CREATE_TYPE_REGION,
                                                                                          &sourceRegion);
                                resizeAndAddKernel(cl::EnqueueArgs(op::OpenCL::getInstance(gpuID)->getQueue(),
                                                                   cl::NDRange(targetWidth, targetHeight)),
                                                                   targetBuffer, sourceBuffer,
                                                                   scaleWidth, scaleHeight,
                                                                   currentWidth, currentHeight,
                                                                   targetWidth, targetHeight);
                            }
                        }
                        // Last image --> average all
                        else
                        {
                            for (auto c = 0 ; c < channels ; c++)
                            {
                                cl_buffer_region targerRegion, sourceRegion;
                                op::OpenCL::getBufferRegion<T>(targerRegion, c * targetChannelOffset, targetChannelOffset);
                                op::OpenCL::getBufferRegion<T>(sourceRegion, c * sourceChannelOffset, sourceChannelOffset);
                                cl::Buffer targetBuffer = targetPtrBuffer.createSubBuffer(CL_MEM_READ_WRITE,
                                                                                          CL_BUFFER_CREATE_TYPE_REGION,
                                                                                          &targerRegion);
                                cl::Buffer sourceBuffer = sourcePtrBuffer.createSubBuffer(CL_MEM_READ_WRITE,
                                                                                          CL_BUFFER_CREATE_TYPE_REGION,
                                                                                          &sourceRegion);
                                resizeAndAverageKernel(cl::EnqueueArgs(op::OpenCL::getInstance(gpuID)->getQueue(),
                                                                       cl::NDRange(targetWidth, targetHeight)),
                                                                       targetBuffer, sourceBuffer,
                                                                       scaleWidth, scaleHeight,
                                                                       currentWidth, currentHeight,
                                                                       targetWidth, targetHeight, (int)sourceSizes.size());
                            }
                        }
                    }
                }
            #else
                UNUSED(targetPtr);
                UNUSED(sourcePtrs);
                UNUSED(targetSize);
                UNUSED(sourceSizes);
                UNUSED(scaleInputToNetInputs);
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

    template void resizeAndMergeOcl(float* targetPtr, const std::vector<const float*>& sourcePtrs,
                                    const std::array<int, 4>& targetSize,
                                    const std::vector<std::array<int, 4>>& sourceSizes,
                                    const std::vector<float>& scaleInputToNetInputs,
                                    int gpuID);

    template void resizeAndMergeOcl(double* targetPtr, const std::vector<const double*>& sourcePtrs,
                                    const std::array<int, 4>& targetSize,
                                    const std::vector<std::array<int, 4>>& sourceSizes,
                                    const std::vector<double>& scaleInputToNetInputs,
                                    int gpuID);
}
