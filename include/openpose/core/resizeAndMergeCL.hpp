#ifndef OPENPOSE_RESIZE_AND_MERGE_CL_HPP
#define OPENPOSE_RESIZE_AND_MERGE_CL_HPP

#include <openpose/core/common.hpp>

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

        void cubicSequentialData(int* xIntArray, int* yIntArray, float* dx, float* dy, const float xSource, const float ySource, const int width, const int height)
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

        float cubicInterpolate(const float v0, const float v1, const float v2, const float v3, const float dx)
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

        float bicubicInterpolate(const float* const sourcePtr, const float xSource, const float ySource, const int widthSource,
                                               const int heightSource, const int widthSourcePtr)
        {
            int xIntArray[4];
            int yIntArray[4];
            float dx;
            float dy;
            cubicSequentialData(xIntArray, yIntArray, &dx, &dy, xSource, ySource, widthSource, heightSource);

            float temp[4];
            for (unsigned char i = 0; i < 4; i++)
            {
                const auto offset = yIntArray[i]*widthSourcePtr;
                temp[i] = cubicInterpolate(sourcePtr[offset + xIntArray[0]], sourcePtr[offset + xIntArray[1]],
                                           sourcePtr[offset + xIntArray[2]], sourcePtr[offset + xIntArray[3]], dx);
            }
            return cubicInterpolate(temp[0], temp[1], temp[2], temp[3], dy);
        }
    );

    const std::string resizeAndMergeKernel = MULTI_LINE_STRING(
        __kernel void resizeAndMergeKernel(__global float* targetPtr, const float* sourcePtr,
                                           const float scaleWidth, const float scaleHeight,
                                           const int sourceWidth, const int sourceHeight,
                                           const int targetWidth, const int targetHeight)
        {
            int x = get_global_id(0);
            int y = get_global_id(1);

            if (x < targetWidth && y < targetHeight)
            {
                const float xSource = (x + 0.5f) * sourceWidth / (float)targetWidth - 0.5f;
                const float ySource = (y + 0.5f) * sourceHeight / (float)targetHeight - 0.5f;
                targetPtr[y*targetWidth+x] = bicubicInterpolate(sourcePtr, xSource, ySource, sourceWidth, sourceHeight,
                                                                sourceWidth);
            }
        }
    );
}

#endif
