#include <openpose/core/nmsBase.hpp>

namespace op
{
    template <typename T>
    void nmsCpu(T* targetPtr, int* kernelPtr, const T* const sourcePtr, const T threshold, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize)
    {
        try
        {
            UNUSED(targetPtr);
            UNUSED(kernelPtr);
            UNUSED(sourcePtr);
            UNUSED(threshold);
            UNUSED(targetSize);
            UNUSED(sourceSize);
            error("CPU version not completely implemented.", __LINE__, __FUNCTION__, __FILE__);

            // TODO: THIS CODE IS WORKING, BUT IT DOES NOT CONSIDER THE MAX NUMBER OF PEAKS
            // const int num = bottom->shape(0);
            // //const int channel = bottom->shape(1);
            // const int oriSpatialHeight = bottom->shape(2);
            // const int oriSpatialWidth = bottom->shape(3);

            // T* dst_pointer = top->mutable_cpu_data();
            // const T* const src_pointer = bottom->cpu_data();
            // const int offset2 = oriSpatialHeight * oriSpatialWidth;
            // const int offset2_dst = (mMaxPeaks+1)*2;

            //stupid method
            // for (int n = 0; n < num; n++)
            // {
            //     //assume only one channel
            //     int peakCount = 0;
            //     for (int y = 0; y < oriSpatialHeight; y++)
            //     {
            //         for (int x = 0; x < oriSpatialWidth; x++)
            //         {
            //             const T value = src_pointer[n * offset2 + y*oriSpatialWidth + x];
            //             if (value >= mThreshold)
            //             {
            //                 const T top = (y == 0) ? 0 : src_pointer[n * offset2 + (y-1)*oriSpatialWidth + x];
            //                 const T bottom = (y == oriSpatialHeight - 1) ? 0 : src_pointer[n * offset2 + (y+1)*oriSpatialWidth + x];
            //                 const T left = (x == 0) ? 0 : src_pointer[n * offset2 + y*oriSpatialWidth + (x-1)];
            //                 const T right = (x == oriSpatialWidth - 1) ? 0 : src_pointer[n * offset2 + y*oriSpatialWidth + (x+1)];
            //                 if (value > top && value > bottom && value > left && value > right)
            //                 {
            //                     dst_pointer[n*offset2_dst + (peakCount + 1) * 2] = x;
            //                     dst_pointer[n*offset2_dst + (peakCount + 1) * 2 + 1] = y;
            //                     peakCount++;
            //                 }
            //             }
            //         }
            //     }
            //     dst_pointer[n*offset2_dst] = peakCount;
            // }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void nmsCpu(float* targetPtr, int* kernelPtr, const float* const sourcePtr, const float threshold, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);
    template void nmsCpu(double* targetPtr, int* kernelPtr, const double* const sourcePtr, const double threshold, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);
}
