#include <opencv2/opencv.hpp> // CV_WARP_INVERSE_MAP, CV_INTER_LINEAR
#include <openpose/core/netCaffe.hpp>
#include <openpose/face/faceParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/cuda.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/face/faceExtractor.hpp>
 
namespace op
{
    FaceExtractor::FaceExtractor(const Point<int>& netInputSize, const Point<int>& netOutputSize, const std::string& modelFolder,
                                 const int gpuId) :
        mNetOutputSize{netOutputSize},
        spNet{std::make_shared<NetCaffe>(std::array<int,4>{1, 3, mNetOutputSize.y, mNetOutputSize.x}, modelFolder + FACE_PROTOTXT,
                                         modelFolder + FACE_TRAINED_MODEL, gpuId)},
        spResizeAndMergeCaffe{std::make_shared<ResizeAndMergeCaffe<float>>()},
        spMaximumCaffe{std::make_shared<MaximumCaffe<float>>()},
        mFaceImageCrop{mNetOutputSize.area()*3}
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
            spMaximumCaffe->Reshape({spHeatMapsBlob.get()}, {spPeaksBlob.get()});
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
 
            log("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void FaceExtractor::forwardPass(const std::vector<Rectangle<float>>& faceRectangles, const cv::Mat& cvInputData, const float scaleInputToOutput)
    {
        try
        {
            if (!faceRectangles.empty())
            {
                // Security checks
                if (cvInputData.empty())
                    error("Empty cvInputData.", __LINE__, __FUNCTION__, __FILE__);

                // Fix parameters
                const auto netInputSide = fastMin(mNetOutputSize.x, mNetOutputSize.y);

                // Set face size
                const auto numberPeople = (int)faceRectangles.size();
                mFaceKeypoints.reset({numberPeople, (int)FACE_NUMBER_PARTS, 3}, 0);

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
                    //               cv::Point{(int)faceRectangle.bottomRight().x, (int)faceRectangle.bottomRight().y},
                    //               cv::Scalar{0,0,255}, 2);
                    // Get parts
                    if (minFaceSize > 40)
                    {
                        // // Debugging -> green rectangle overwriting red one
                        // log(std::to_string(cvInputData.cols) + " " + std::to_string(cvInputData.rows));
                        // cv::rectangle(cvInputDataCopy,
                        //               cv::Point{(int)faceRectangle.x, (int)faceRectangle.y},
                        //               cv::Point{(int)faceRectangle.bottomRight().x, (int)faceRectangle.bottomRight().y},
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
                        cv::warpAffine(cvInputData, faceImage, Mscaling, cv::Size{mNetOutputSize.x, mNetOutputSize.y},
                                       CV_INTER_LINEAR | CV_WARP_INVERSE_MAP, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

                        // cv::Mat -> float*
                        uCharCvMatToFloatPtr(mFaceImageCrop.getPtr(), faceImage, true);

                        // // Debugging
                        // if (person < 5)
                        // cv::imshow("faceImage" + std::to_string(person), faceImage);

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
                        #ifndef CPU_ONLY
                            spMaximumCaffe->Forward_gpu({spHeatMapsBlob.get()}, {spPeaksBlob.get()});
                            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                        #else
                            spMaximumCaffe->Forward_cpu({spHeatMapsBlob.get()}, {spPeaksBlob.get()});
                        #endif
     
                        const auto* facePeaksPtr = spPeaksBlob->mutable_cpu_data();
                        for (auto part = 0 ; part < mFaceKeypoints.getSize(1) ; part++)
                        {
                            const auto xyIndex = part * mFaceKeypoints.getSize(2);
                            const auto x = facePeaksPtr[xyIndex];
                            const auto y = facePeaksPtr[xyIndex + 1];
                            const auto score = facePeaksPtr[xyIndex + 2];
                            const auto baseIndex = mFaceKeypoints.getSize(2) * (part + person * mFaceKeypoints.getSize(1));
                            mFaceKeypoints[baseIndex] = (float)(scaleInputToOutput * (Mscaling.at<double>(0,0) * x
                                                                                      + Mscaling.at<double>(0,1) * y
                                                                                      + Mscaling.at<double>(0,2)));
                            mFaceKeypoints[baseIndex+1] = (float)(scaleInputToOutput * (Mscaling.at<double>(1,0) * x
                                                                                      + Mscaling.at<double>(1,1) * y
                                                                                      + Mscaling.at<double>(1,2)));
                            mFaceKeypoints[baseIndex+2] = score;
                        }
                    }
                }
                // // Debugging
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
