#ifndef OPENPOSE_RESIZE_AND_MERGE_CL_HPP
#define OPENPOSE_RESIZE_AND_MERGE_CL_HPP

#include <openpose/core/common.hpp>
#ifdef USE_OPENCL
    #include <CL/cl2.hpp>
#endif

namespace op
{
    const std::string commonKernels = MULTI_LINE_STRING(
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

        void cubicSequentialData(int* xIntArray, int* yIntArray, Type* dx, Type* dy, const Type xSource, const Type ySource, const int width, const int height)
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

        Type bicubicInterpolate(const Type* const sourcePtr, const Type xSource, const Type ySource, const int widthSource,
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
                targetPtrC[y*targetWidth+x] = bicubicInterpolate(sourcePtrC, xSource, ySource, sourceWidth, sourceHeight, sourceWidth);
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
                targetPtr[y*targetWidth+x] = bicubicInterpolate(sourcePtr, xSource, ySource, sourceWidth, sourceHeight, sourceWidth);
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
                targetPtr[y*targetWidth+x] += bicubicInterpolate(sourcePtr, xSource, ySource, sourceWidth, sourceHeight, sourceWidth);
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
                Type* targetPixel = &targetPtr[y*targetWidth+x];
                *targetPixel = (*targetPixel + interpolated) / (Type)(counter);
            }
        }
    );
}

#endif
