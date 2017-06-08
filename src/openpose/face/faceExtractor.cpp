#include <opencv2/opencv.hpp> // CV_WARP_INVERSE_MAP, CV_INTER_LINEAR
#include <openpose/core/netCaffe.hpp>
#include <openpose/face/faceParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/cuda.hpp>
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/face/faceExtractor.hpp>
 
namespace op
{
    FaceExtractor::FaceExtractor(const Point<int>& netInputSize, const Point<int>& netOutputSize, const std::string& modelFolder, const int gpuId) :
        mNetOutputSize{netOutputSize},
        spNet{std::make_shared<NetCaffe>(std::array<int,4>{1, 3, mNetOutputSize.y, mNetOutputSize.x}, modelFolder + FACE_PROTOTXT, modelFolder + FACE_TRAINED_MODEL, gpuId)},
        spResizeAndMergeCaffe{std::make_shared<ResizeAndMergeCaffe<float>>()},
        spNmsCaffe{std::make_shared<NmsCaffe<float>>()},
        mFaceImageCrop{mNetOutputSize.area()*3}
    {
        try
        {
            checkE(netOutputSize, netInputSize, "Net input and output size must be equal.", __LINE__, __FUNCTION__, __FILE__);
            // Properties
            for (auto& property : mProperties)
                property = 0.;
            mProperties[(int)FaceProperty::NMSThreshold] = FACE_DEFAULT_NMS_THRESHOLD;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void FaceExtractor::initializationOnThread()
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
            spResizeAndMergeCaffe->Reshape({spCaffeNetOutputBlob.get()}, {spHeatMapsBlob.get()}, FACE_CCN_DECREASE_FACTOR, mergeFirstDimension);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
 
            // Pose extractor blob and layer
            spPeaksBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
            spNmsCaffe->Reshape({spHeatMapsBlob.get()}, {spPeaksBlob.get()}, FACE_MAX_PEAKS, FACE_NUMBER_PARTS+1);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
 
            log("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void FaceExtractor::forwardPass(const std::vector<Rectangle<float>>& faceRectangles, const cv::Mat& cvInputData)
    {
        try
        {
            if (!faceRectangles.empty())
            {
                // Security checks
                if (cvInputData.empty())
                    error("Empty cvInputData.", __LINE__, __FUNCTION__, __FILE__);

                // Set face size
                const auto numberPeople = (int)faceRectangles.size();
                mFaceKeypoints.reset({numberPeople, FACE_NUMBER_PARTS, 3}, 0);

// log("\nAreas:");
// cv::Mat cvInputDataCopy = cvInputData.clone();
                for (auto person = 0 ; person < numberPeople ; person++)
                {
                    // Only consider faces with a minimum pixel area
                    const auto faceAreaSquared = std::sqrt(faceRectangles.at(person).area());
                    // Get parts
                    if (faceAreaSquared > 60)
                    {
                        const auto& faceRectangle = faceRectangles.at(person);
// log(faceAreaSquared);
// cv::rectangle(cvInputDataCopy,
//               cv::Point{(int)faceRectangle.x, (int)faceRectangle.y},
//               // cv::Point{(int)(faceRectangle.x + faceRectangle.width), (int)(faceRectangle.y + faceRectangle.height)},
//               cv::Point{(int)faceRectangle.bottomRight().x, (int)faceRectangle.bottomRight().y},
//               cv::Scalar{255,0,255}, 2);
                        // Get face position(s)
                        const Point<float> faceCenterPosition{faceRectangle.topLeft()};
                        const auto faceSize = fastMax(faceRectangle.width, faceRectangle.height);

                        // Resize and shift image to face rectangle positions
                        const double scaleFace = faceSize / (double)fastMin(mNetOutputSize.x, mNetOutputSize.y);
                        cv::Mat Mscaling = cv::Mat::eye(2, 3, CV_64F);
                        Mscaling.at<double>(0,0) = scaleFace;
                        Mscaling.at<double>(1,1) = scaleFace;
                        Mscaling.at<double>(0,2) = faceCenterPosition.x;
                        Mscaling.at<double>(1,2) = faceCenterPosition.y;

                        cv::Mat faceImage;
                        cv::warpAffine(cvInputData, faceImage, Mscaling, cv::Size{mNetOutputSize.x, mNetOutputSize.y}, CV_INTER_LINEAR | CV_WARP_INVERSE_MAP, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

                        // cv::Mat -> float*
                        uCharCvMatToFloatPtr(mFaceImageCrop.getPtr(), faceImage, true);
                        // 1. Caffe deep network
                        auto* inputDataGpuPtr = spNet->getInputDataGpuPtr();
                        cudaMemcpy(inputDataGpuPtr, mFaceImageCrop.getPtr(), mNetOutputSize.area() * 3 * sizeof(float), cudaMemcpyHostToDevice);
                        spNet->forwardPass();
     
                        // 2. Resize heat maps + merge different scales
                        #ifndef CPU_ONLY
                            spResizeAndMergeCaffe->Forward_gpu({spCaffeNetOutputBlob.get()}, {spHeatMapsBlob.get()});
                            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                        #else
                            spResizeAndMergeCaffe->Forward_cpu({spCaffeNetOutputBlob.get()}, {spHeatMapsBlob.get()});
                        #endif
     
                        // 3. Get peaks by Non-Maximum Suppression
                        spNmsCaffe->setThreshold(get(FaceProperty::NMSThreshold));
                        #ifndef CPU_ONLY
                            spNmsCaffe->Forward_gpu({spHeatMapsBlob.get()}, {spPeaksBlob.get()});
                            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                        #else
                            spNmsCaffe->Forward_cpu({spHeatMapsBlob.get()}, {spPeaksBlob.get()});
                        #endif
     
                        const auto* facePeaksPtr = spPeaksBlob->mutable_cpu_data();
                        const auto facePeaksOffset = (FACE_MAX_PEAKS+1) * 3;

                        for (auto part = 0 ; part < FACE_NUMBER_PARTS ; part++)
                        {
                            // Get max peak
                            const int numPeaks = intRound(facePeaksPtr[facePeaksOffset*part]);
                            auto maxScore = -1.f;
                            auto maxPeak = -1;
                            for (auto peak = 0 ; peak < numPeaks ; peak++)
                            {
                                const auto xyIndex = facePeaksOffset * part + (1 + peak) * 3;
                                const auto score = facePeaksPtr[xyIndex + 2];
                                if (score > maxScore)
                                {
                                    maxScore = score;
                                    maxPeak = peak;
                                }
                            }
                            // Fill face keypoints
                            if (maxPeak >= 0)
                            {
                                const auto xyIndex = facePeaksOffset * part + (1 + maxPeak) * 3;
                                const auto x = facePeaksPtr[xyIndex];
                                const auto y = facePeaksPtr[xyIndex + 1];
                                const auto score = facePeaksPtr[xyIndex + 2];
                                const auto baseIndex = (person * FACE_NUMBER_PARTS + part) * 3;
                                mFaceKeypoints[baseIndex] = Mscaling.at<double>(0,0) * x + Mscaling.at<double>(0,1) * y + Mscaling.at<double>(0,2);
                                mFaceKeypoints[baseIndex+1] = Mscaling.at<double>(1,0) * x + Mscaling.at<double>(1,1) * y + Mscaling.at<double>(1,2);
                                mFaceKeypoints[baseIndex+2] = score;
                            }
                        }
                    }
                }
// cv::imshow("AcvInputDataCopy", cvInputDataCopy);
            }
            else
                mFaceKeypoints.reset();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Array<float> FaceExtractor::getFaceKeypoints() const
    {
        try
        {
            checkThread();
            return mFaceKeypoints;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }

    double FaceExtractor::get(const FaceProperty property) const
    {
        try
        {
            return mProperties.at((int)property);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.;
        }
    }

    void FaceExtractor::set(const FaceProperty property, const double value)
    {
        try
        {
            mProperties.at((int)property) = {value};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void FaceExtractor::increase(const FaceProperty property, const double value)
    {
        try
        {
            mProperties[(int)property] = mProperties.at((int)property) + value;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void FaceExtractor::checkThread() const
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
}
