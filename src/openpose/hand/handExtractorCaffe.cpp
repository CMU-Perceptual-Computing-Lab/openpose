#ifdef USE_CAFFE
    #include <caffe/blob.hpp>
#endif
#include <opencv2/opencv.hpp> // CV_WARP_INVERSE_MAP, CV_INTER_LINEAR
#include <openpose/gpu/cuda.hpp>
#include <openpose/hand/handParameters.hpp>
#include <openpose/net/maximumCaffe.hpp>
#include <openpose/net/netCaffe.hpp>
#include <openpose/net/resizeAndMergeCaffe.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/hand/handExtractorCaffe.hpp>

namespace op
{
    struct HandExtractorCaffe::ImplHandExtractorCaffe
    {
        #ifdef USE_CAFFE
            bool netInitialized;
            const int mGpuId;
            std::shared_ptr<NetCaffe> spNetCaffe;
            std::shared_ptr<ResizeAndMergeCaffe<float>> spResizeAndMergeCaffe;
            std::shared_ptr<MaximumCaffe<float>> spMaximumCaffe;
            // Init with thread
            std::shared_ptr<ArrayCpuGpu<float>> spCaffeNetOutputBlob;
            std::shared_ptr<ArrayCpuGpu<float>> spHeatMapsBlob;
            std::shared_ptr<ArrayCpuGpu<float>> spPeaksBlob;

            ImplHandExtractorCaffe(const std::string& modelFolder, const int gpuId,
                                   const bool enableGoogleLogging) :
                netInitialized{false},
                mGpuId{gpuId},
                spNetCaffe{std::make_shared<NetCaffe>(modelFolder + HAND_PROTOTXT, modelFolder + HAND_TRAINED_MODEL,
                                                      gpuId, enableGoogleLogging)},
                spResizeAndMergeCaffe{std::make_shared<ResizeAndMergeCaffe<float>>()},
                spMaximumCaffe{std::make_shared<MaximumCaffe<float>>()}
            {
            }
        #endif
    };

    #ifdef USE_CAFFE
        void cropFrame(Array<float>& handImageCrop, cv::Mat& affineMatrix, const cv::Mat& cvInputData,
                       const Rectangle<float>& handRectangle, const int netInputSide,
                       const Point<int>& netOutputSize, const bool mirrorImage)
        {
            try
            {
                // Resize image to hands positions
                const auto scaleLeftHand = handRectangle.width / (float)netInputSide;
                affineMatrix = cv::Mat::eye(2,3,CV_64F);
                if (mirrorImage)
                    affineMatrix.at<double>(0,0) = -scaleLeftHand;
                else
                    affineMatrix.at<double>(0,0) = scaleLeftHand;
                affineMatrix.at<double>(1,1) = scaleLeftHand;
                if (mirrorImage)
                    affineMatrix.at<double>(0,2) = handRectangle.x + handRectangle.width;
                else
                    affineMatrix.at<double>(0,2) = handRectangle.x;
                affineMatrix.at<double>(1,2) = handRectangle.y;
                cv::Mat handImage;
                cv::warpAffine(cvInputData, handImage, affineMatrix, cv::Size{netOutputSize.x, netOutputSize.y},
                               CV_INTER_LINEAR | CV_WARP_INVERSE_MAP, cv::BORDER_CONSTANT, cv::Scalar{0,0,0});
                               // CV_INTER_CUBIC | CV_WARP_INVERSE_MAP, cv::BORDER_CONSTANT, cv::Scalar{0,0,0});
                // cv::Mat -> float*
                uCharCvMatToFloatPtr(handImageCrop.getPtr(), handImage, true);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void connectKeypoints(Array<float>& handCurrent, const int person,
                              const cv::Mat& affineMatrix, const float* handPeaks)
        {
            try
            {
                // Estimate keypoint locations
                for (auto part = 0 ; part < handCurrent.getSize(1) ; part++)
                {
                    const auto xyIndex = part * handCurrent.getSize(2);
                    const auto x = handPeaks[xyIndex];
                    const auto y = handPeaks[xyIndex + 1];
                    const auto score = handPeaks[xyIndex + 2];
                    const auto baseIndex = handCurrent.getSize(2) * (part + person * handCurrent.getSize(1));
                    handCurrent[baseIndex] = (float)(affineMatrix.at<double>(0,0)*x + affineMatrix.at<double>(0,1)*y
                                                     + affineMatrix.at<double>(0,2));
                    handCurrent[baseIndex+1] = (float)(affineMatrix.at<double>(1,0)*x + affineMatrix.at<double>(1,1)*y
                                                       + affineMatrix.at<double>(1,2));
                    handCurrent[baseIndex+2] = score;
                }
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        Rectangle<float> getHandRectangle(Array<float>& handCurrent, const int person, const float increaseRatio,
                                          const float thresholdRectangle,
                                          const Rectangle<float>& previousHandRectangle = Rectangle<float>{})
        {
            try
            {
                // Initial Rectangle
                auto handRectangle = getKeypointsRectangle(handCurrent, person, thresholdRectangle);
                // Get final width
                auto finalWidth = fastMax(handRectangle.width, handRectangle.height) * increaseRatio;
                if (previousHandRectangle.width > 0 && previousHandRectangle.height > 0)
                    finalWidth = fastMax(handRectangle.width, 0.85f
                                         * fastMax(previousHandRectangle.width, previousHandRectangle.height));
                // Update Rectangle
                handRectangle.recenter(finalWidth, finalWidth);
                return handRectangle;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return Rectangle<float>{};
            }
        }

        void updateHandHeatMapsForPerson(Array<float>& heatMaps, const int person, const ScaleMode heatMapScaleMode,
                                         const float* heatMapsGpuPtr)
        {
            try
            {
                // Copy memory
                const auto channelOffset = heatMaps.getVolume(2, 3);
                const auto volumeBodyParts = HAND_NUMBER_PARTS * channelOffset;
                auto totalOffset = 0u;
                auto* heatMapsPtr = &heatMaps.getPtr()[person*volumeBodyParts];
                // Copy hand parts
                #ifdef USE_CUDA
                    cudaMemcpy(heatMapsPtr, heatMapsGpuPtr, volumeBodyParts * sizeof(float), cudaMemcpyDeviceToHost);
                #else
                    //std::memcpy(heatMapsPtr, heatMapsGpuPtr, volumeBodyParts * sizeof(float));
                    std::copy(heatMapsGpuPtr, heatMapsGpuPtr + volumeBodyParts, heatMapsPtr);
                #endif
                // Change from [0,1] to [-1,1]
                if (heatMapScaleMode == ScaleMode::PlusMinusOne)
                    for (auto i = 0u ; i < volumeBodyParts ; i++)
                        heatMapsPtr[i] = fastTruncate(heatMapsPtr[i]) * 2.f - 1.f;
                // [0, 255]
                else if (heatMapScaleMode == ScaleMode::UnsignedChar)
                    for (auto i = 0u ; i < volumeBodyParts ; i++)
                        heatMapsPtr[i] = (float)positiveIntRound(fastTruncate(heatMapsPtr[i]) * 255.f);
                // Avoid values outside original range
                else
                    for (auto i = 0u ; i < volumeBodyParts ; i++)
                        heatMapsPtr[i] = fastTruncate(heatMapsPtr[i]);
                totalOffset += (unsigned int)volumeBodyParts;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        inline void reshapeHandExtractorCaffe(std::shared_ptr<ResizeAndMergeCaffe<float>>& resizeAndMergeCaffe,
                                              std::shared_ptr<MaximumCaffe<float>>& maximumCaffe,
                                              std::shared_ptr<ArrayCpuGpu<float>>& caffeNetOutputBlob,
                                              std::shared_ptr<ArrayCpuGpu<float>>& heatMapsBlob,
                                              std::shared_ptr<ArrayCpuGpu<float>>& peaksBlob,
                                              const int gpuID)
        {
            try
            {
                // HeatMaps extractor blob and layer
                const bool mergeFirstDimension = true;
                resizeAndMergeCaffe->Reshape(
                    std::vector<ArrayCpuGpu<float>*>{caffeNetOutputBlob.get()},
                    std::vector<ArrayCpuGpu<float>*>{heatMapsBlob.get()},
                    HAND_CCN_DECREASE_FACTOR, 1.f, mergeFirstDimension, gpuID);
                // Pose extractor blob and layer
                maximumCaffe->Reshape({heatMapsBlob.get()}, {peaksBlob.get()});
                // Cuda check
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }
    #endif

    HandExtractorCaffe::HandExtractorCaffe(const Point<int>& netInputSize, const Point<int>& netOutputSize,
                                           const std::string& modelFolder, const int gpuId,
                                           const unsigned short numberScales,
                                           const float rangeScales, const std::vector<HeatMapType>& heatMapTypes,
                                           const ScaleMode heatMapScaleMode,
                                           const bool enableGoogleLogging) :
        HandExtractorNet{netInputSize, netOutputSize, numberScales, rangeScales, heatMapTypes, heatMapScaleMode}
        #ifdef USE_CAFFE
        , upImpl{new ImplHandExtractorCaffe{modelFolder, gpuId, enableGoogleLogging}}
        #endif
    {
        try
        {
            #if !defined USE_CAFFE
                UNUSED(netInputSize);
                UNUSED(netOutputSize);
                UNUSED(modelFolder);
                UNUSED(gpuId);
                UNUSED(numberScales);
                UNUSED(rangeScales);
                UNUSED(heatMapTypes);
                UNUSED(heatMapScaleMode);
                UNUSED(enableGoogleLogging);
                error("OpenPose must be compiled with the `USE_CAFFE` & `USE_CUDA` macro definitions in order to run"
                      " this functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    HandExtractorCaffe::~HandExtractorCaffe()
    {
    }

    void HandExtractorCaffe::netInitializationOnThread()
    {
        try
        {
            #ifdef USE_CAFFE
                // Logging
                log("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Initialize Caffe net
                upImpl->spNetCaffe->initializationOnThread();
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
                // Initialize blobs
                upImpl->spCaffeNetOutputBlob = upImpl->spNetCaffe->getOutputBlobArray();
                upImpl->spHeatMapsBlob = {std::make_shared<ArrayCpuGpu<float>>(1,1,1,1)};
                upImpl->spPeaksBlob = {std::make_shared<ArrayCpuGpu<float>>(1,1,1,1)};
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
                // Logging
                log("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void HandExtractorCaffe::forwardPass(
        const std::vector<std::array<Rectangle<float>, 2>> handRectangles, const cv::Mat& cvInputData)
    {
        try
        {
            #ifdef USE_CAFFE
                if (mEnabled && !handRectangles.empty())
                {
                    // Sanity check
                    if (cvInputData.empty())
                        error("Empty cvInputData.", __LINE__, __FUNCTION__, __FILE__);

                    // Fix parameters
                    const auto netInputSide = fastMin(mNetOutputSize.x, mNetOutputSize.y);

                    // Set hand size
                    const auto numberPeople = (int)handRectangles.size();
                    mHandKeypoints[0].reset({numberPeople, (int)HAND_NUMBER_PARTS, 3}, 0.f);
                    mHandKeypoints[1].reset(mHandKeypoints[0].getSize(), 0.f);

                    // HeatMaps: define size
                    if (!mHeatMapTypes.empty())
                    {
                        mHeatMaps[0].reset({numberPeople, (int)HAND_NUMBER_PARTS, mNetOutputSize.y, mNetOutputSize.x});
                        mHeatMaps[1].reset({numberPeople, (int)HAND_NUMBER_PARTS, mNetOutputSize.y, mNetOutputSize.x});
                    }

                    // // Debugging
                    // cv::Mat cvInputDataCopied = cvInputData.clone();
                    // Extract hand keypoints for each person
                    for (auto hand = 0 ; hand < 2 ; hand++)
                    {
                        // Parameters
                        auto& handCurrent = mHandKeypoints[hand];
                        const bool mirrorImage = (hand == 0);
                        for (auto person = 0 ; person < numberPeople ; person++)
                        {
                            const auto& handRectangle = handRectangles.at(person).at(hand);
                            // Sanity check
                            if (handRectangle.width != handRectangle.height)
                                error("Hand rectangle for hand keypoint estimation must be squared, i.e.,"
                                      " width = height (" + std::to_string(handRectangle.width) + " vs. "
                                      + std::to_string(handRectangle.height) + ").", __LINE__, __FUNCTION__, __FILE__);
                            // Only consider faces with a minimum pixel area
                            const auto minHandSize = fastMin(handRectangle.width, handRectangle.height);
                            // // Debugging -> red rectangle
                            // if (handRectangle.width > 0)
                            //     cv::rectangle(
                            //         cvInputDataCopied,
                            //         cv::Point{positiveIntRound(handRectangle.x),
                            //                   positiveIntRound(handRectangle.y)},
                            //         cv::Point{positiveIntRound(handRectangle.x + handRectangle.width),
                            //                   positiveIntRound(handRectangle.y + handRectangle.height)},
                            //         cv::Scalar{(hand * 255.f),0.f,255.f}, 2);
                            // Get parts
                            if (minHandSize > 1 && handRectangle.area() > 10)
                            {
                                // Single-scale detection
                                if (mMultiScaleNumberAndRange.first == 1)
                                {
                                    // // Debugging -> green rectangle overwriting red one
                                    // if (handRectangle.width > 0)
                                    //     cv::rectangle(
                                    //         cvInputDataCopied,
                                    //         cv::Point{positiveIntRound(handRectangle.x),
                                    //                   positiveIntRound(handRectangle.y)},
                                    //         cv::Point{positiveIntRound(handRectangle.x + handRectangle.width),
                                    //                   positiveIntRound(handRectangle.y + handRectangle.height)},
                                    //         cv::Scalar{(hand * 255.f),255.f,0.f}, 2);
                                    // Parameters
                                    cv::Mat affineMatrix;
                                    // Resize image to hands positions + cv::Mat -> float*
                                    cropFrame(mHandImageCrop, affineMatrix, cvInputData, handRectangle, netInputSide,
                                              mNetOutputSize, mirrorImage);
                                    // Deep net + Estimate keypoint locations
                                    detectHandKeypoints(handCurrent, person, affineMatrix);
                                }
                                // Multi-scale detection
                                else
                                {
                                    const auto handPtrArea = handCurrent.getSize(1) * handCurrent.getSize(2);
                                    auto* handCurrentPtr = handCurrent.getPtr() + person * handPtrArea;
                                    const auto numberScales = mMultiScaleNumberAndRange.first;
                                    const auto initScale = 1.f - mMultiScaleNumberAndRange.second / 2.f;
                                    for (auto i = 0 ; i < numberScales ; i++)
                                    {
                                        // Get current scale
                                        const auto scale = initScale
                                                         + mMultiScaleNumberAndRange.second * i / (numberScales-1.f);
                                        // Process hand
                                        Array<float> handEstimated(
                                            {1, handCurrent.getSize(1), handCurrent.getSize(2)}, 0.f);
                                        const auto handRectangleScale = recenter(
                                            handRectangle,
                                            (float)(positiveIntRound(handRectangle.width * scale) / 2 * 2),
                                            (float)(positiveIntRound(handRectangle.height * scale) / 2 * 2)
                                        );
                                        // // Debugging -> blue rectangle
                                        // cv::rectangle(cvInputDataCopied,
                                        //               cv::Point{positiveIntRound(handRectangleScale.x),
                                        //                         positiveIntRound(handRectangleScale.y)},
                                        //               cv::Point{positiveIntRound(handRectangleScale.x
                                        //                                  + handRectangleScale.width),
                                        //                         positiveIntRound(handRectangleScale.y
                                        //                                  + handRectangleScale.height)},
                                        //               cv::Scalar{255,0,0}, 2);
                                        // Parameters
                                        cv::Mat affineMatrix;
                                        // Resize image to hands positions + cv::Mat -> float*
                                        cropFrame(mHandImageCrop, affineMatrix, cvInputData, handRectangleScale,
                                                  netInputSide, mNetOutputSize, mirrorImage);
                                        // Deep net + Estimate keypoint locations
                                        detectHandKeypoints(handEstimated, 0, affineMatrix);
                                        if (i == 0
                                            || getAverageScore(handEstimated,0) > getAverageScore(handCurrent,person))
                                            std::copy(handEstimated.getConstPtr(),
                                                      handEstimated.getConstPtr() + handPtrArea, handCurrentPtr);
                                    }
                                }
                                // HeatMaps: storing
                                if (!mHeatMapTypes.empty()){
                                    #ifdef USE_CUDA
                                        updateHandHeatMapsForPerson(mHeatMaps[hand], person, mHeatMapScaleMode,
                                                                    upImpl->spHeatMapsBlob->gpu_data());
                                    #else
                                        updateHandHeatMapsForPerson(mHeatMaps[hand], person, mHeatMapScaleMode,
                                                                    upImpl->spHeatMapsBlob->cpu_data());
                                    #endif
                                }
                            }
                        }
                    }
                    // // Debugging
                    // cv::imshow("cvInputDataCopied", cvInputDataCopied);
                }
                else
                {
                    mHandKeypoints[0].reset();
                    mHandKeypoints[1].reset();
                }
            #else
                UNUSED(handRectangles);
                UNUSED(cvInputData);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void HandExtractorCaffe::detectHandKeypoints(Array<float>& handCurrent, const int person,
                                                 const cv::Mat& affineMatrix)
    {
        try
        {
            #ifdef USE_CAFFE
                // 1. Deep net
                upImpl->spNetCaffe->forwardPass(mHandImageCrop);

                // Reshape blobs
                if (!upImpl->netInitialized)
                {
                    upImpl->netInitialized = true;
                    reshapeHandExtractorCaffe(upImpl->spResizeAndMergeCaffe, upImpl->spMaximumCaffe,
                                              upImpl->spCaffeNetOutputBlob, upImpl->spHeatMapsBlob,
                                              upImpl->spPeaksBlob, upImpl->mGpuId);
                }

                // 2. Resize heat maps + merge different scales
                upImpl->spResizeAndMergeCaffe->Forward(
                    {upImpl->spCaffeNetOutputBlob.get()}, {upImpl->spHeatMapsBlob.get()});

                // 3. Get peaks by Non-Maximum Suppression
                upImpl->spMaximumCaffe->Forward({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()});

                // Estimate keypoint locations
                connectKeypoints(handCurrent, person, affineMatrix,
                                 upImpl->spPeaksBlob->mutable_cpu_data());

                // 5. CUDA sanity check
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
            #else
                UNUSED(handCurrent);
                UNUSED(person);
                UNUSED(affineMatrix);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
