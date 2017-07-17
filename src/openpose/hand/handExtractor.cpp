#include <limits> // std::numeric_limits
#include <opencv2/opencv.hpp> // CV_WARP_INVERSE_MAP, CV_INTER_LINEAR
#include <openpose/core/netCaffe.hpp>
#include <openpose/hand/handParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/cuda.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/hand/handExtractor.hpp>
 
namespace op
{
    void cropFrame(Array<float>& handImageCrop, cv::Mat& affineMatrix, const cv::Mat& cvInputData, const Rectangle<float>& handRectangle,
                   const int netInputSide, const Point<int>& netOutputSize, const bool mirrorImage)
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

    void connectKeypoints(Array<float>& handCurrent, const float scaleInputToOutput, const int person, const cv::Mat& affineMatrix,
                          const float* handPeaks)
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
                handCurrent[baseIndex] = (float)(scaleInputToOutput * (affineMatrix.at<double>(0,0)*x + affineMatrix.at<double>(0,1)*y
                                                                       + affineMatrix.at<double>(0,2)));
                handCurrent[baseIndex+1] = (float)(scaleInputToOutput * (affineMatrix.at<double>(1,0)*x + affineMatrix.at<double>(1,1)*y
                                                                       + affineMatrix.at<double>(1,2)));
                handCurrent[baseIndex+2] = score;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Rectangle<float> getHandRectangle(Array<float>& handCurrent, const int person, const float increaseRatio,
                                      const int handNumberParts, const float thresholdRectangle,
                                      const Rectangle<float>& previousHandRectangle = Rectangle<float>{})
    {
        try
        {
            // Initial Rectangle
            auto handRectangle = getKeypointsRectangle(handCurrent, person, handNumberParts, thresholdRectangle);
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

    HandExtractor::HandExtractor(const Point<int>& netInputSize, const Point<int>& netOutputSize,
                                 const std::string& modelFolder, const int gpuId, const unsigned short numberScales,
                                 const float rangeScales) :
        mMultiScaleNumberAndRange{std::make_pair(numberScales, rangeScales)},
        mNetOutputSize{netOutputSize},
        spNet{std::make_shared<NetCaffe>(std::array<int,4>{1, 3, mNetOutputSize.y, mNetOutputSize.x}, modelFolder + HAND_PROTOTXT,
                                         modelFolder + HAND_TRAINED_MODEL, gpuId)},
        spResizeAndMergeCaffe{std::make_shared<ResizeAndMergeCaffe<float>>()},
        spMaximumCaffe{std::make_shared<MaximumCaffe<float>>()},
        mHandImageCrop{mNetOutputSize.area()*3}
    {
        try
        {
            checkE(netOutputSize.x, netInputSize.x, "Net input and output size must be equal.", __LINE__, __FUNCTION__, __FILE__);
            checkE(netOutputSize.y, netInputSize.y, "Net input and output size must be equal.", __LINE__, __FUNCTION__, __FILE__);
            checkE(netInputSize.x, netInputSize.y, "Net input size must be squared.", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void HandExtractor::initializationOnThread()
    {
        try
        {
            log("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

            // Get thread id
            mThreadId = {std::this_thread::get_id()};

            // Caffe net
            spNet->initializationOnThread();
            spCaffeNetOutputBlob = ((NetCaffe*)spNet.get())->getOutputBlob();
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);

            // HeatMaps extractor blob and layer
            spHeatMapsBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
            const bool mergeFirstDimension = true;
            spResizeAndMergeCaffe->Reshape({spCaffeNetOutputBlob.get()}, {spHeatMapsBlob.get()}, HAND_CCN_DECREASE_FACTOR, mergeFirstDimension);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);

            // Pose extractor blob and layer
            spPeaksBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
            spMaximumCaffe->Reshape({spHeatMapsBlob.get()}, {spPeaksBlob.get()});
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
 
            log("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void HandExtractor::forwardPass(const std::vector<std::array<Rectangle<float>, 2>> handRectangles, const cv::Mat& cvInputData,
                                    const float scaleInputToOutput)
    {
        try
        {
            if (!handRectangles.empty())
            {
                // Security checks
                if (cvInputData.empty())
                    error("Empty cvInputData.", __LINE__, __FUNCTION__, __FILE__);

                // Fix parameters
                const auto netInputSide = fastMin(mNetOutputSize.x, mNetOutputSize.y);

                // Set hand size
                const auto numberPeople = (int)handRectangles.size();
                mHandKeypoints[0].reset({numberPeople, (int)HAND_NUMBER_PARTS, 3}, 0);
                mHandKeypoints[1].reset(mHandKeypoints[0].getSize(), 0);

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
                        // Only consider faces with a minimum pixel area
                        const auto minHandSize = fastMin(handRectangle.width, handRectangle.height);
                        // // Debugging -> red rectangle
                        // if (handRectangle.width > 0)
                        //     cv::rectangle(cvInputDataCopied,
                        //                   cv::Point{intRound(handRectangle.x), intRound(handRectangle.y)},
                        //                   cv::Point{intRound(handRectangle.x + handRectangle.width),
                        //                             intRound(handRectangle.y + handRectangle.height)},
                        //                   cv::Scalar{(hand * 255.f),0.f,255.f}, 2);
                        // Get parts
                        if (minHandSize > 1 && handRectangle.area() > 10)
                        {
                            // Single-scale detection
                            if (mMultiScaleNumberAndRange.first == 1)
                            {
                                // // Debugging -> green rectangle overwriting red one
                                // if (handRectangle.width > 0)
                                //     cv::rectangle(cvInputDataCopied,
                                //                   cv::Point{intRound(handRectangle.x), intRound(handRectangle.y)},
                                //                   cv::Point{intRound(handRectangle.x + handRectangle.width),
                                //                             intRound(handRectangle.y + handRectangle.height)},
                                //                   cv::Scalar{(hand * 255.f),255.f,0.f}, 2);
                                // Parameters
                                cv::Mat affineMatrix;
                                // Resize image to hands positions + cv::Mat -> float*
                                cropFrame(mHandImageCrop, affineMatrix, cvInputData, handRectangle, netInputSide, mNetOutputSize, mirrorImage);
                                // Deep net + Estimate keypoint locations
                                detectHandKeypoints(handCurrent, scaleInputToOutput, person, affineMatrix);
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
                                    const auto scale = initScale + mMultiScaleNumberAndRange.second * i / (numberScales-1.f);
                                    // Process hand
                                    Array<float> handEstimated({1, handCurrent.getSize(1), handCurrent.getSize(2)}, 0);
                                    const auto handRectangleScale = recenter(handRectangle,
                                                                             (float)(intRound(handRectangle.width * scale) / 2 * 2),
                                                                             (float)(intRound(handRectangle.height * scale) / 2 * 2));
                                    // // Debugging -> blue rectangle
                                    // cv::rectangle(cvInputDataCopied,
                                    //               cv::Point{intRound(handRectangleScale.x), intRound(handRectangleScale.y)},
                                    //               cv::Point{intRound(handRectangleScale.x + handRectangleScale.width),
                                    //                         intRound(handRectangleScale.y + handRectangleScale.height)},
                                    //               cv::Scalar{255,0,0}, 2);
                                    // Parameters
                                    cv::Mat affineMatrix;
                                    // Resize image to hands positions + cv::Mat -> float*
                                    cropFrame(mHandImageCrop, affineMatrix, cvInputData, handRectangleScale, netInputSide, mNetOutputSize, mirrorImage);
                                    // Deep net + Estimate keypoint locations
                                    detectHandKeypoints(handEstimated, scaleInputToOutput, 0, affineMatrix);
                                    if (i == 0 || getAverageScore(handEstimated, 0) > getAverageScore(handCurrent, person))
                                        std::copy(handEstimated.getConstPtr(), handEstimated.getConstPtr() + handPtrArea, handCurrentPtr);
                                }
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
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::array<Array<float>, 2> HandExtractor::getHandKeypoints() const
    {
        try
        {
            checkThread();
            return mHandKeypoints;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::array<Array<float>, 2>(); // Parentheses instead of braces to avoid error in GCC 4.8
        }
    }

    void HandExtractor::checkThread() const
    {
        try
        {
            if(mThreadId != std::this_thread::get_id())
                error("The CPU/GPU pointer data cannot be accessed from a different thread.", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void HandExtractor::detectHandKeypoints(Array<float>& handCurrent, const float scaleInputToOutput, const int person,
                                            const cv::Mat& affineMatrix)
    {
        try
        {
            // Deep net
            // 1. Caffe deep network
            auto* inputDataGpuPtr = spNet->getInputDataGpuPtr();
            cudaMemcpy(inputDataGpuPtr, mHandImageCrop.getConstPtr(), mNetOutputSize.area() * 3 * sizeof(float), cudaMemcpyHostToDevice);
            spNet->forwardPass();

            // 2. Resize heat maps + merge different scales
            #ifndef CPU_ONLY
                spResizeAndMergeCaffe->Forward_gpu({spCaffeNetOutputBlob.get()}, {spHeatMapsBlob.get()});
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #else
                spResizeAndMergeCaffe->Forward_cpu({spCaffeNetOutputBlob.get()}, {spHeatMapsBlob.get()});
            #endif

            // 3. Get peaks by Non-Maximum Suppression
            #ifndef CPU_ONLY
                spMaximumCaffe->Forward_gpu({spHeatMapsBlob.get()}, {spPeaksBlob.get()});
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #else
                spMaximumCaffe->Forward_cpu({spHeatMapsBlob.get()}, {spPeaksBlob.get()});
            #endif

            // Estimate keypoint locations
            connectKeypoints(handCurrent, scaleInputToOutput, person, affineMatrix, spPeaksBlob->mutable_cpu_data());
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
