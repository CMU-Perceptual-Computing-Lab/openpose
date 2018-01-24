#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/core/resizeAndMergeBase.hpp>
#ifdef USE_OPENCL
    #include <openpose/core/clManager.hpp>
    #include <openpose/core/resizeAndMergeCL.hpp>
#endif

namespace op
{
    template <typename T>
    void resizeAndMergeCpu(T* targetPtr, const std::vector<const T*>& sourcePtrs,
                           const std::array<int, 4>& targetSize,
                           const std::vector<std::array<int, 4>>& sourceSizes,
                           const std::vector<T>& scaleInputToNetInputs)
    {
        try
        {
            // Security checks
            if (sourceSizes.empty())
                error("sourceSizes cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
            if (sourcePtrs.size() != sourceSizes.size() || sourceSizes.size() != scaleInputToNetInputs.size())
                error("Size(sourcePtrs) must match size(sourceSizes) and size(scaleInputToNetInputs). Currently: "
                      + std::to_string(sourcePtrs.size()) + " vs. " + std::to_string(sourceSizes.size()) + " vs. "
                      + std::to_string(scaleInputToNetInputs.size()) + ".", __LINE__, __FUNCTION__, __FILE__);

            // Params
            const auto nums = (signed)sourceSizes.size();
            const auto channels = targetSize[1]; // 57
            const auto targetHeight = targetSize[2]; // 368
            const auto targetWidth = targetSize[3]; // 496
            const auto targetChannelOffset = targetWidth * targetHeight;

            // No multi-scale merging or no merging required
            if (sourceSizes.size() == 1)
            {
                // Params
                const auto& sourceSize = sourceSizes[0];
                const auto sourceHeight = sourceSize[2]; // 368/8 ..
                const auto sourceWidth = sourceSize[3]; // 496/8 ..
                const auto sourceChannelOffset = sourceHeight * sourceWidth;
                if (sourceSize[0] != 1)
                    error("It should never reache this point. Notify us otherwise.",
                          __LINE__, __FUNCTION__, __FILE__);

                // Per channel resize
                const T* sourcePtr = sourcePtrs[0];
                for (auto c = 0 ; c < channels ; c++)
                {
                    cv::Mat source(cv::Size(sourceWidth, sourceHeight), CV_32FC1,
                                   const_cast<T*>(&sourcePtr[c*sourceChannelOffset]));
                    cv::Mat target(cv::Size(targetWidth, targetHeight), CV_32FC1,
                                   (&targetPtr[c*targetChannelOffset]));
                    cv::resize(source, target, {targetWidth, targetHeight}, 0, 0, CV_INTER_CUBIC);
                }
            }
            // Multi-scale merging
            else
            {
                // Construct temp targets. We resuse targetPtr to store first scale
                std::vector<std::unique_ptr<T>> tempTargetPtrs;
                for (auto n = 1; n < nums; n++){
                    tempTargetPtrs.emplace_back(std::unique_ptr<T>(new T[targetChannelOffset * channels]()));
                }

                // Resize and sum
                for (auto n = 0; n < nums; n++){

                    // Params
                    const auto& sourceSize = sourceSizes[n];
                    const auto sourceHeight = sourceSize[2]; // 368/6 ..
                    const auto sourceWidth = sourceSize[3]; // 496/8 ..
                    const auto sourceChannelOffset = sourceHeight * sourceWidth;

                    // Access pointers
                    const T* sourcePtr = sourcePtrs[n];
                    T* tempTargetPtr;
                    if(n != 0)
                        tempTargetPtr = tempTargetPtrs[n-1].get();
                    else
                        tempTargetPtr = targetPtr;

                    T* firstTempTargetPtr = targetPtr;
                    for (auto c = 0 ; c < channels ; c++)
                    {
                        // Resize
                        cv::Mat source(cv::Size(sourceWidth, sourceHeight), CV_32FC1,
                                       const_cast<T*>(&sourcePtr[c*sourceChannelOffset]));
                        cv::Mat target(cv::Size(targetWidth, targetHeight), CV_32FC1,
                                       (&tempTargetPtr[c*targetChannelOffset]));
                        cv::resize(source, target, {targetWidth, targetHeight}, 0, 0, CV_INTER_CUBIC);

                        // Add
                        if (n != 0)
                        {
                            cv::Mat addTarget(cv::Size(targetWidth, targetHeight), CV_32FC1,
                                              (&firstTempTargetPtr[c*targetChannelOffset]));
                            cv::add(target, addTarget, addTarget);
                        }
                    }
                }

                // Average
                for (auto c = 0 ; c < channels ; c++)
                {
                    cv::Mat target(cv::Size(targetWidth, targetHeight), CV_32FC1, (&targetPtr[c*targetChannelOffset]));
                    target /= (float)nums;
                }

            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void resizeAndMergeOcl(T* targetPtr, const std::vector<const T*>& sourcePtrs,
                           const std::array<int, 4>& targetSize,
                           const std::vector<std::array<int, 4>>& sourceSizes,
                           const std::vector<T>& scaleInputToNetInputs,
                           int gpuID)
    {
        try
        {
            // Security checks
            if (sourceSizes.empty())
                error("sourceSizes cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
            if (sourcePtrs.size() != sourceSizes.size() || sourceSizes.size() != scaleInputToNetInputs.size())
                error("Size(sourcePtrs) must match size(sourceSizes) and size(scaleInputToNetInputs). Currently: "
                      + std::to_string(sourcePtrs.size()) + " vs. " + std::to_string(sourceSizes.size()) + " vs. "
                      + std::to_string(scaleInputToNetInputs.size()) + ".", __LINE__, __FUNCTION__, __FILE__);

            // Get Kernels
            cl::Kernel& resizeAndMergeKernel = op::CLManager::getInstance(gpuID)->getKernelFromManager<T>("resizeAndMergeKernel",op::commonKernels+op::resizeAndMergeKernel);
            cl::Buffer targetPtrBuffer = cl::Buffer((cl_mem)(targetPtr), true);

            // Parameters
            const auto channels = targetSize[1];
            const auto targetHeight = targetSize[2];
            const auto targetWidth = targetSize[3];
            const auto& sourceSize = sourceSizes[0];
            const auto sourceHeight = sourceSize[2];
            const auto sourceWidth = sourceSize[3];

            // No multi-scale merging or no merging required
            if (sourceSizes.size() == 1)
            {
                const auto num = sourceSize[0];
                if (targetSize[0] > 1 || num == 1)
                {
                    cl::Buffer sourcePtrBuffer = cl::Buffer((cl_mem)(sourcePtrs.at(0)), true);
                    const auto sourceChannelOffshet = sourceHeight * sourceWidth;
                    const auto targetChannelOffset = targetWidth * targetHeight;
                    for (auto n = 0; n < num; n++)
                    {
                        const auto offsetBase = n*channels;
                        for (auto c = 0 ; c < channels ; c++)
                        {
                            const auto offset = offsetBase + c;
                            cl_buffer_region targerRegion = op::CLManager::getBufferRegion<T>(offset * targetChannelOffset, targetChannelOffset);
                            cl::Buffer targetBuffer = targetPtrBuffer.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &targerRegion);
                            cl_buffer_region sourceRegion = op::CLManager::getBufferRegion<T>(offset * sourceChannelOffset, sourceChannelOffset);
                            cl::Buffer sourceBuffer = sourcePtrBuffer.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &sourceRegion);
                            resizeAndMergeKernel.setArg(0, targetBuffer);
                            resizeAndMergeKernel.setArg(1, sourceBuffer);
                            resizeAndMergeKernel.setArg(2, sourceWidth);
                            resizeAndMergeKernel.setArg(3, sourceHeight);
                            resizeAndMergeKernel.setArg(4, targetWidth);
                            resizeAndMergeKernel.setArg(5, targetHeight);
                            op::CLManager::getInstance(gpuID)->getQueue().enqueueNDRangeKernel(resizeAndMergeKernel, cl::NDRange(), cl::NDRange(targetWidth,targetHeight), cl::NDRange(), NULL, NULL);
                        }
                    }
                    op::CLManager::getInstance(gpuID)->getQueue().finish();
                }
                // Old inefficient multi-scale merging
                else
                    error("It should never reache this point. Notify us otherwise.", __LINE__, __FUNCTION__, __FILE__);
            }
            // Multi-scaling merging
            else
            {
                error("Not Implemented", __LINE__, __FUNCTION__, __FILE__);
//                const auto targetChannelOffset = targetWidth * targetHeight;
//                cudaMemset(targetPtr, 0.f, channels*targetChannelOffset * sizeof(T));
//                const auto scaleToMainScaleWidth = targetWidth / T(sourceWidth);
//                const auto scaleToMainScaleHeight = targetHeight / T(sourceHeight);

//                for (auto i = 0u ; i < sourceSizes.size(); i++)
//                {
//                    const auto& currentSize = sourceSizes.at(i);
//                    const auto currentHeight = currentSize[2];
//                    const auto currentWidth = currentSize[3];
//                    const auto sourceChannelOffset = currentHeight * currentWidth;
//                    const auto scaleInputToNet = scaleInputToNetInputs[i] / scaleInputToNetInputs[0];
//                    const auto scaleWidth = scaleToMainScaleWidth / scaleInputToNet;
//                    const auto scaleHeight = scaleToMainScaleHeight / scaleInputToNet;
//                    // All but last image --> add
//                    if (i < sourceSizes.size() - 1)
//                    {
//                        for (auto c = 0 ; c < channels ; c++)
//                        {
//                            resizeKernelAndAdd<<<numBlocks, threadsPerBlock>>>(
//                                targetPtr + c * targetChannelOffset, sourcePtrs[i] + c * sourceChannelOffset,
//                                scaleWidth, scaleHeight, currentWidth, currentHeight, targetWidth,
//                                targetHeight
//                            );
//                        }
//                    }
//                    // Last image --> average all
//                    else
//                    {
//                        for (auto c = 0 ; c < channels ; c++)
//                        {
//                            resizeKernelAndAverage<<<numBlocks, threadsPerBlock>>>(
//                                targetPtr + c * targetChannelOffset, sourcePtrs[i] + c * sourceChannelOffset,
//                                scaleWidth, scaleHeight, currentWidth, currentHeight, targetWidth,
//                                targetHeight, sourceSizes.size()
//                            );
//                        }
//                    }
//                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void resizeAndMergeCpu(float* targetPtr, const std::vector<const float*>& sourcePtrs,
                                    const std::array<int, 4>& targetSize,
                                    const std::vector<std::array<int, 4>>& sourceSizes,
                                    const std::vector<float>& scaleInputToNetInputs);
    template void resizeAndMergeCpu(double* targetPtr, const std::vector<const double*>& sourcePtrs,
                                    const std::array<int, 4>& targetSize,
                                    const std::vector<std::array<int, 4>>& sourceSizes,
                                    const std::vector<double>& scaleInputToNetInputs);
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
