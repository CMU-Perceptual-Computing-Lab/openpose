#include <opencv2/opencv.hpp>
#include <openpose/net/nmsBase.hpp>

namespace op
{
    template <typename T>
    void nmsRegisterKernelCPU(int* kernelPtr, const T* const sourcePtr, const int w, const int h,
                              const T& threshold, const int x, const int y)
    {
        // We have three scenarios for NMS, one for the border, 1 for the 1st inner border, and
        // 1 for the rest. cv::resize adds artifacts around the 1st inner border, causing two
        // maximas to occur side by side. Eg. [1 1 0.8 0.8 0.5 ..]. The CUDA kernel gives
        // [0.8 1 0.8 0.8 0.5 ..] Hence for this special case in the 1st inner border, we look at the
        // visible regions.

        const auto index = y*w + x;
        if (1 < x && x < (w-2) && 1 < y && y < (h-2))
        {
            const auto value = sourcePtr[index];
            if (value > threshold)
            {
                const auto topLeft     = sourcePtr[(y-1)*w + x-1];
                const auto top         = sourcePtr[(y-1)*w + x];
                const auto topRight    = sourcePtr[(y-1)*w + x+1];
                const auto left        = sourcePtr[    y*w + x-1];
                const auto right       = sourcePtr[    y*w + x+1];
                const auto bottomLeft  = sourcePtr[(y+1)*w + x-1];
                const auto bottom      = sourcePtr[(y+1)*w + x];
                const auto bottomRight = sourcePtr[(y+1)*w + x+1];

                if (value > topLeft && value > top && value > topRight
                    && value > left && value > right
                        && value > bottomLeft && value > bottom && value > bottomRight)
                    kernelPtr[index] = 1;
                else
                    kernelPtr[index] = 0;
            }
            else
                kernelPtr[index] = 0;
        }
        else if (x == 1 || x == (w-2) || y == 1 || y == (h-2))
        {
            //kernelPtr[index] = 0;
            const auto value = sourcePtr[index];
            if (value > threshold)
            {
                const auto topLeft      = ((0 < x && 0 < y)         ? sourcePtr[(y-1)*w + x-1]  : threshold);
                const auto top          = (0 < y                    ? sourcePtr[(y-1)*w + x]    : threshold);
                const auto topRight     = ((0 < y && x < (w-1))     ? sourcePtr[(y-1)*w + x+1]  : threshold);
                const auto left         = (0 < x                    ? sourcePtr[    y*w + x-1]  : threshold);
                const auto right        = (x < (w-1)                ? sourcePtr[y*w + x+1]      : threshold);
                const auto bottomLeft   = ((y < (h-1) && 0 < x)     ? sourcePtr[(y+1)*w + x-1]  : threshold);
                const auto bottom       = (y < (h-1)                ? sourcePtr[(y+1)*w + x]    : threshold);
                const auto bottomRight  = ((x < (w-1) && y < (h-1)) ? sourcePtr[(y+1)*w + x+1]  : threshold);

                if (value >= topLeft && value >= top && value >= topRight
                    && value >= left && value >= right
                        && value >= bottomLeft && value >= bottom && value >= bottomRight)
                    kernelPtr[index] = 1;
                else
                    kernelPtr[index] = 0;
            }
            else
                kernelPtr[index] = 0;
        }
        else
            kernelPtr[index] = 0;
    }

    template <typename T>
    void nmsAccuratePeakPosition(T* output, const T* const sourcePtr, const int& peakLocX, const int& peakLocY,
                                 const int& width, const int& height, const Point<T>& offset)
    {
        T xAcc = 0.f;
        T yAcc = 0.f;
        T scoreAcc = 0.f;
        const auto dWidth = 3;
        const auto dHeight = 3;
        for (auto dy = -dHeight ; dy <= dHeight ; dy++)
        {
            const auto y = peakLocY + dy;
            if (0 <= y && y < height) // Default height = 368
            {
                for (auto dx = -dWidth ; dx <= dWidth ; dx++)
                {
                    const auto x = peakLocX + dx;
                    if (0 <= x && x < width) // Default width = 656
                    {
                        const auto score = sourcePtr[y * width + x];
                        if (score > 0)
                        {
                            xAcc += x*score;
                            yAcc += y*score;
                            scoreAcc += score;
                        }
                    }
                }
            }
        }

        // Offset to keep Matlab format (empirically higher acc)
        // Best results for 1 scale: x + 0, y + 0.5
        // +0.5 to both to keep Matlab format
        output[0] = xAcc / scoreAcc + offset.x;
        output[1] = yAcc / scoreAcc + offset.y;
        output[2] = sourcePtr[peakLocY*width + peakLocX];
    }

    template <typename T>
    void nmsCpu(T* targetPtr, int* kernelPtr, const T* const sourcePtr, const T threshold,
                const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize,
                const Point<T>& offset)
    {
        try
        {
            // Security checks
            if (sourceSize.empty())
                error("sourceSize cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
            if (targetSize.empty())
                error("targetSize cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
            if (threshold < 0 || threshold > 1.0)
                error("threshold value invalid.", __LINE__, __FUNCTION__, __FILE__);

            // Params
            const auto channels = targetSize[1]; // 57
            const auto sourceHeight = sourceSize[2]; // 368
            const auto sourceWidth = sourceSize[3]; // 496
            const auto targetPeaks = targetSize[2]; // 97
            const auto targetPeakVec = targetSize[3]; // 3
            const auto sourceChannelOffset = sourceWidth * sourceHeight;
            const auto targetChannelOffset = targetPeaks * targetPeakVec;

            // Per channel operation
            for (auto c = 0 ; c < channels ; c++)
            {
                auto* currKernelPtr = &kernelPtr[c*sourceChannelOffset];
                const T* currSourcePtr = &sourcePtr[c*sourceChannelOffset];

                for (auto y = 0; y < sourceHeight; y++)
                    for (auto x = 0; x < sourceWidth; x++)
                        nmsRegisterKernelCPU(currKernelPtr, currSourcePtr, sourceWidth, sourceHeight, threshold, x, y);

                auto currentPeakCount = 1;
                auto* currTargetPtr = &targetPtr[c*targetChannelOffset];
                for (auto y = 0; y < sourceHeight; y++)
                {
                    for (auto x = 0; x < sourceWidth; x++)
                    {
                        const auto index = y*sourceWidth + x;
                        // Find high intensity points
                        if (currentPeakCount < targetPeaks)
                        {
                            if (currKernelPtr[index] == 1)
                            {
                                // Accurate Peak Position
                                nmsAccuratePeakPosition(&currTargetPtr[currentPeakCount*3], currSourcePtr, x, y,
                                                        sourceWidth, sourceHeight, offset);
                                currentPeakCount++;
                            }
                        }

                    }
                }
                currTargetPtr[0] = currentPeakCount-1;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void nmsCpu(float* targetPtr, int* kernelPtr, const float* const sourcePtr, const float threshold,
                         const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize,
                         const Point<float>& offset);
    template void nmsCpu(double* targetPtr, int* kernelPtr, const double* const sourcePtr, const double threshold,
                         const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize,
                         const Point<double>& offset);
}
