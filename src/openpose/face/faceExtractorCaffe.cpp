#ifdef USE_CAFFE
    #include <caffe/blob.hpp>
#endif
#include <opencv2/opencv.hpp> // CV_WARP_INVERSE_MAP, CV_INTER_LINEAR
#include <openpose/face/faceParameters.hpp>
#include <openpose/gpu/cuda.hpp>
#include <openpose/net/maximumCaffe.hpp>
#include <openpose/net/netCaffe.hpp>
#include <openpose/net/resizeAndMergeCaffe.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/face/faceExtractorCaffe.hpp>

namespace op
{
    struct FaceExtractorCaffe::ImplFaceExtractorCaffe
    {
        #ifdef USE_CAFFE
            bool netInitialized;
            const int mGpuId;
            std::shared_ptr<NetCaffe> spNetCaffe;
            std::shared_ptr<ResizeAndMergeCaffe<float>> spResizeAndMergeCaffe;
            std::shared_ptr<MaximumCaffe<float>> spMaximumCaffe;
            // Init with thread
            boost::shared_ptr<caffe::Blob<float>> spCaffeNetOutputBlob;
            std::shared_ptr<caffe::Blob<float>> spHeatMapsBlob;
            std::shared_ptr<caffe::Blob<float>> spPeaksBlob;

            ImplFaceExtractorCaffe(const std::string& modelFolder, const int gpuId, const bool enableGoogleLogging) :
                netInitialized{false},
                mGpuId{gpuId},
                spNetCaffe{std::make_shared<NetCaffe>(modelFolder + FACE_PROTOTXT, modelFolder + FACE_TRAINED_MODEL,
                                                      gpuId, enableGoogleLogging)},
                spResizeAndMergeCaffe{std::make_shared<ResizeAndMergeCaffe<float>>()},
                spMaximumCaffe{std::make_shared<MaximumCaffe<float>>()}
            {
            }
        #endif
    };

    #ifdef USE_CAFFE
        void updateFaceHeatMapsForPerson(Array<float>& heatMaps, const int person, const ScaleMode heatMapScaleMode,
                                         const float* heatMapsGpuPtr)
        {
            try
            {
                // Copy memory
                const auto channelOffset = heatMaps.getVolume(2, 3);
                const auto volumeBodyParts = FACE_NUMBER_PARTS * channelOffset;
                auto totalOffset = 0u;
                auto* heatMapsPtr = &heatMaps.getPtr()[person*volumeBodyParts];
                // Copy face parts                                      
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
                        heatMapsPtr[i] = (float)intRound(fastTruncate(heatMapsPtr[i]) * 255.f);
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

        inline void reshapeFaceExtractorCaffe(std::shared_ptr<ResizeAndMergeCaffe<float>>& resizeAndMergeCaffe,
                                              std::shared_ptr<MaximumCaffe<float>>& maximumCaffe,
                                              boost::shared_ptr<caffe::Blob<float>>& caffeNetOutputBlob,
                                              std::shared_ptr<caffe::Blob<float>>& heatMapsBlob,
                                              std::shared_ptr<caffe::Blob<float>>& peaksBlob,
                                              const int gpuID)
        {
            try
            {
                // HeatMaps extractor blob and layer
                const bool mergeFirstDimension = true;
                resizeAndMergeCaffe->Reshape({caffeNetOutputBlob.get()}, {heatMapsBlob.get()},
                                             FACE_CCN_DECREASE_FACTOR, 1.f, mergeFirstDimension, gpuID);
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

    FaceExtractorCaffe::FaceExtractorCaffe(const Point<int>& netInputSize, const Point<int>& netOutputSize,
                                           const std::string& modelFolder, const int gpuId,
                                           const std::vector<HeatMapType>& heatMapTypes,
                                           const ScaleMode heatMapScale, const bool enableGoogleLogging) :
        FaceExtractorNet{netInputSize, netOutputSize, heatMapTypes, heatMapScale}
        #ifdef USE_CAFFE
        , upImpl{new ImplFaceExtractorCaffe{modelFolder, gpuId, enableGoogleLogging}}
        #endif
    {
        try
        {
            #ifndef USE_CAFFE
                UNUSED(netInputSize);
                UNUSED(netOutputSize);
                UNUSED(modelFolder);
                UNUSED(gpuId);
                UNUSED(heatMapTypes);
                UNUSED(heatMapScale);
                error("OpenPose must be compiled with the `USE_CAFFE` & `USE_CUDA` macro definitions in order to run"
                      " this functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
            #ifdef COMMERCIAL_LICENSE
                error("Face is not included in the commercial version of OpenPose yet. We might include it in the future after some"
                      " commercial issues have been solved. Thanks!", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    FaceExtractorCaffe::~FaceExtractorCaffe()
    {
    }

    void FaceExtractorCaffe::netInitializationOnThread()
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
                upImpl->spCaffeNetOutputBlob = upImpl->spNetCaffe->getOutputBlob();
                upImpl->spHeatMapsBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
                upImpl->spPeaksBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
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

    void FaceExtractorCaffe::forwardPass(const std::vector<Rectangle<float>>& faceRectangles,
                                         const cv::Mat& cvInputData)
    {
        try
        {
            #ifdef USE_CAFFE
                if (mEnabled && !faceRectangles.empty())
                {
                    // Sanity check
                    if (cvInputData.empty())
                        error("Empty cvInputData.", __LINE__, __FUNCTION__, __FILE__);

                    // Fix parameters
                    const auto netInputSide = fastMin(mNetOutputSize.x, mNetOutputSize.y);

                    // Set face size
                    const auto numberPeople = (int)faceRectangles.size();
                    mFaceKeypoints.reset({numberPeople, (int)FACE_NUMBER_PARTS, 3}, 0);

                    // HeatMaps: define size
                    if (!mHeatMapTypes.empty())
                        mHeatMaps.reset({numberPeople, (int)FACE_NUMBER_PARTS, mNetOutputSize.y, mNetOutputSize.x});

                    // // Debugging
                    // cv::Mat cvInputDataCopy = cvInputData.clone();
                    // Extract face keypoints for each person
                    for (auto person = 0 ; person < numberPeople ; person++)
                    {
                        const auto& faceRectangle = faceRectangles.at(person);
                        // Only consider faces with a minimum pixel area
                        const auto minFaceSize = fastMin(faceRectangle.width, faceRectangle.height);
                        // // Debugging -> red rectangle
                        // log(std::to_string(cvInputData.cols) + " " + std::to_string(cvInputData.rows));
                        // cv::rectangle(cvInputDataCopy,
                        //               cv::Point{(int)faceRectangle.x, (int)faceRectangle.y},
                        //               cv::Point{(int)faceRectangle.bottomRight().x,
                        //                         (int)faceRectangle.bottomRight().y},
                        //               cv::Scalar{0,0,255}, 2);
                        // Get parts
                        if (minFaceSize > 40)
                        {
                            // // Debugging -> green rectangle overwriting red one
                            // log(std::to_string(cvInputData.cols) + " " + std::to_string(cvInputData.rows));
                            // cv::rectangle(cvInputDataCopy,
                            //               cv::Point{(int)faceRectangle.x, (int)faceRectangle.y},
                            //               cv::Point{(int)faceRectangle.bottomRight().x,
                            //                         (int)faceRectangle.bottomRight().y},
                            //               cv::Scalar{0,255,0}, 2);
                            // Resize and shift image to face rectangle positions
                            const auto faceSize = fastMax(faceRectangle.width, faceRectangle.height);
                            const double scaleFace = faceSize / (double)netInputSide;
                            cv::Mat Mscaling = cv::Mat::eye(2, 3, CV_64F);
                            Mscaling.at<double>(0,0) = scaleFace;
                            Mscaling.at<double>(1,1) = scaleFace;
                            Mscaling.at<double>(0,2) = faceRectangle.x;
                            Mscaling.at<double>(1,2) = faceRectangle.y;

                            cv::Mat faceImage;
                            cv::warpAffine(cvInputData, faceImage, Mscaling,
                                           cv::Size{mNetOutputSize.x, mNetOutputSize.y},
                                           CV_INTER_LINEAR | CV_WARP_INVERSE_MAP,
                                           cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

                            // cv::Mat -> float*
                            uCharCvMatToFloatPtr(mFaceImageCrop.getPtr(), faceImage, true);

                            // // Debugging
                            // if (person < 5)
                            // cv::imshow("faceImage" + std::to_string(person), faceImage);

                            // 1. Caffe deep network
                            upImpl->spNetCaffe->forwardPass(mFaceImageCrop);

                            // Reshape blobs
                            if (!upImpl->netInitialized)
                            {
                                upImpl->netInitialized = true;
                                reshapeFaceExtractorCaffe(upImpl->spResizeAndMergeCaffe, upImpl->spMaximumCaffe,
                                                          upImpl->spCaffeNetOutputBlob, upImpl->spHeatMapsBlob,
                                                          upImpl->spPeaksBlob, upImpl->mGpuId);
                            }

                            // 2. Resize heat maps + merge different scales
                            upImpl->spResizeAndMergeCaffe->Forward(
                                {upImpl->spCaffeNetOutputBlob.get()}, {upImpl->spHeatMapsBlob.get()});

                            // 3. Get peaks by Non-Maximum Suppression
                            upImpl->spMaximumCaffe->Forward(
                                {upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()});

                            const auto* facePeaksPtr = upImpl->spPeaksBlob->mutable_cpu_data();
                            for (auto part = 0 ; part < mFaceKeypoints.getSize(1) ; part++)
                            {
                                const auto xyIndex = part * mFaceKeypoints.getSize(2);
                                const auto x = facePeaksPtr[xyIndex];
                                const auto y = facePeaksPtr[xyIndex + 1];
                                const auto score = facePeaksPtr[xyIndex + 2];
                                const auto baseIndex = mFaceKeypoints.getSize(2)
                                                     * (part + person * mFaceKeypoints.getSize(1));
                                mFaceKeypoints[baseIndex] = (float)(Mscaling.at<double>(0,0) * x
                                                                    + Mscaling.at<double>(0,1) * y
                                                                    + Mscaling.at<double>(0,2));
                                mFaceKeypoints[baseIndex+1] = (float)(Mscaling.at<double>(1,0) * x
                                                                      + Mscaling.at<double>(1,1) * y
                                                                      + Mscaling.at<double>(1,2));
                                mFaceKeypoints[baseIndex+2] = score;
                            }
                            // HeatMaps: storing
                            if (!mHeatMapTypes.empty())
                            {
                                updateFaceHeatMapsForPerson(
                                    mHeatMaps, person, mHeatMapScaleMode,
                                    #ifdef USE_CUDA
                                        upImpl->spHeatMapsBlob->gpu_data()
                                    #else
                                        upImpl->spHeatMapsBlob->cpu_data()
                                    #endif
                                );
                            }
                        }
                    }
                    // // Debugging
                    // cv::imshow("AcvInputDataCopy", cvInputDataCopy);
                }
                else
                    mFaceKeypoints.reset();

                // 5. CUDA sanity check
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
            #else
                UNUSED(faceRectangles);
                UNUSED(cvInputData);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
