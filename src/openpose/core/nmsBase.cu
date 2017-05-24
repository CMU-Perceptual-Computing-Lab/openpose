#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include "openpose/utilities/cuda.hpp"
#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/utilities/cuda.hpp"
#include "openpose/core/nmsBase.hpp"

namespace op
{
    const auto THREADS_PER_BLOCK_1D = 16u;
    const auto THREADS_PER_BLOCK = 512u;

    template <typename T>
    __global__ void nmsRegisterKernel(int* kernelPtr, const T* const sourcePtr, const int w, const int h, const T threshold)
    {
        // get pixel location (x,y)
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto index = y*w + x;

        if (0 < x && x < (w-1) && 0 < y && y < (h-1))
        {
            const auto value = sourcePtr[index];
            if (value > threshold)
            {
                const auto top_left        = sourcePtr[(y-1)*w + x-1];
                const auto top             = sourcePtr[(y-1)*w + x];
                const auto top_right       = sourcePtr[(y-1)*w + x+1];
                const auto left            = sourcePtr[y*w + (x-1)];
                const auto right           = sourcePtr[y*w + (x+1)];
                const auto bottom_left     = sourcePtr[(y+1)*w + x-1];
                const auto bottom          = sourcePtr[(y+1)*w + x];
                const auto bottom_right    = sourcePtr[(y+1)*w + x+1];

                if (value > top_left && value > top && value > top_right && value > left && value > right && value > bottom_left && value > bottom && value > bottom_right)
                    kernelPtr[index] = 1;
                else
                    kernelPtr[index] = 0;
            }
            else
                kernelPtr[index] = 0;
        }
        else if (x == 0 || x == (w-1) || y == 0 || y == (h-1))
            kernelPtr[index] = 0;
    }

    template <typename T>
    __global__ void writeResultKernel(T* output, const int length, const int* const input, const T* const sourcePtr, const int width, const int maxPeaks)
    {
        __shared__ int local[THREADS_PER_BLOCK+1]; // one more
        const auto globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

        if (globalIdx < length)
        {
            local[threadIdx.x] = input[globalIdx];
            //last thread in the block but not globally last, load one more
            if (threadIdx.x == THREADS_PER_BLOCK - 1 && globalIdx != length - 1)
                local[threadIdx.x+1] = input[globalIdx+1];

            __syncthreads();
            // see difference, except the globally last one
            if (globalIdx != length - 1)
            {
                if (local[threadIdx.x] != local[threadIdx.x + 1])
                {
                    //means A[globalIdx] == A[globalIdx + 1] as the input[globalIdx]-th repeat
                    const auto peakIndex = input[globalIdx]; //0-index
                    const auto peakLocX = globalIdx % width;
                    const auto peakLocY = globalIdx / width;

                    if (peakIndex < maxPeaks) // limitation
                    {
                        T xAcc = 0.f;
                        T yAcc = 0.f;
                        T scoreAcc = 0.f;
                        for (auto dy = -3 ; dy < 4 ; dy++)
                        {
                            if (0 < (peakLocY+dy) && (peakLocY+dy) < width)
                            {
                                for (auto dx = -3 ; dx < 4 ; dx++)
                                {
                                    if (0 < (peakLocX+dx) && (peakLocX+dx) < width)
                                    {
                                        const auto score = sourcePtr[(peakLocY+dy)*width + peakLocX+dx];
                                        const auto x = peakLocX+dx;
                                        const auto y = peakLocY+dy;
                                        if (score>0)
                                        {
                                            xAcc += x*score;
                                            yAcc += y*score;
                                            scoreAcc += score;
                                        }
                                    }
                                }
                            }
                        }

                        const auto output_index = (peakIndex + 1) * 3;
                        output[output_index] = xAcc / scoreAcc;
                        output[output_index + 1] = yAcc / scoreAcc;
                        output[output_index + 2] = sourcePtr[peakLocY*width + peakLocX];
                    }
                }
            }
            else
                output[0] = input[globalIdx]; //number of peaks
        }
    }

    template <typename T>
    void nmsGpu(T* targetPtr, int* kernelPtr, const T* const sourcePtr, const T threshold, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize)
    {
        try
        {
            //Forward_cpu(bottom, top);
            const auto num = sourceSize[0];
            const auto height = sourceSize[2];
            const auto width = sourceSize[3];
            const auto channels = targetSize[1];
            const auto maxPeaks = targetSize[2]-1;
            const auto imageOffset = height * width;
            const auto offsetTarget = (maxPeaks+1)*targetSize[3];

            const dim3 threadsPerBlock2D{THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D};
            const dim3 numBlocks2D{getNumberCudaBlocks(width, threadsPerBlock2D.x), getNumberCudaBlocks(height, threadsPerBlock2D.y)};
            const dim3 threadsPerBlock1D{THREADS_PER_BLOCK};
            const dim3 numBlocks1D{getNumberCudaBlocks(imageOffset, threadsPerBlock1D.x)};
            // log("num_b: " + std::to_string(bottom->shape(0)));       // = 1
            // log("channel_b: " + std::to_string(bottom->shape(1)));   // = 57 = 18 body parts + bkg + 19x2 PAFs
            // log("height_b: " + std::to_string(bottom->shape(2)));    // = 368 = height
            // log("width_b: " + std::to_string(bottom->shape(3)));     // = 656 = width
            // log("num_t: " + std::to_string(top->shape(0)));       // = 1
            // log("channel_t: " + std::to_string(top->shape(1)));   // = 18 = numberParts
            // log("height_t: " + std::to_string(top->shape(2)));    // = 97 = maxPeople + 1
            // log("width_t: " + std::to_string(top->shape(3)));     // = 3 = [x, y, score]
            // log("");

            for (auto n = 0; n < num; n++)
            {
                for (auto c = 0; c < channels; c++)
                {
                    // log("channel: " + std::to_string(c));
                    const auto offsetChannel = (n * channels + c);
                    auto* kernelPtrOffsetted = kernelPtr + offsetChannel * imageOffset;
                    const auto* const sourcePtrOffsetted = sourcePtr + offsetChannel * imageOffset;
                    auto* targetPtrOffsetted = targetPtr + offsetChannel * offsetTarget;

                    // This returns kernelPtrOffsetted, a binary array with 0s & 1s. 1s in the local maximum positions (size = size(sourcePtrOffsetted))
                    nmsRegisterKernel<<<numBlocks2D, threadsPerBlock2D>>>(kernelPtrOffsetted, sourcePtrOffsetted, width, height, threshold); //[0,0,0,0,1,0,0,0,0,1,0,0,0,0]
                    //debug
                    // if (c==3){
                    //     char filename[50];
                    //     sprintf(filename, "work%02d.txt", c);
                    //     std::ofstream fout(filename);
                    //     int* kernelPtrOffsetted_local = mKernelBlob.mutable_cpu_data() + n * parts_num * imageOffset + c * imageOffset;
                    //     for (int y = 0; y < height; y++){
                    //         for (int x = 0; x < width; x++)
                    //             fout << kernelPtrOffsetted_local[y*width + x] << "\t";
                    //         fout<< std::endl;
                    //     }
                    //     fout.close();
                    // }
                    auto kernelThrustPtr = thrust::device_pointer_cast(kernelPtrOffsetted);

                    // This modifies kernelPtrOffsetted, now it indicates the local maximum indexes. Format: 0,0,0,1,1,1,1,2,2,2,... First maximum at index 2, second at 6, etc...
                    thrust::exclusive_scan(kernelThrustPtr, kernelThrustPtr + imageOffset, kernelThrustPtr); //[0,0,0,0,0,1,1,1,1,1,2,2,2,2]

                    // This returns targetPtrOffsetted, with the NMS applied over it
                    writeResultKernel<<<numBlocks1D, threadsPerBlock1D>>>(targetPtrOffsetted, imageOffset, kernelPtrOffsetted, sourcePtrOffsetted, width, maxPeaks);
                }
            }
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void nmsGpu(float* targetPtr, int* kernelPtr, const float* const sourcePtr, const float threshold, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);
    template void nmsGpu(double* targetPtr, int* kernelPtr, const double* const sourcePtr, const double threshold, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);
}
