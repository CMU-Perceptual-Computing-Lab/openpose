#include <limits> // std::numeric_limits
#include <opencv2/opencv.hpp> // CV_WARP_INVERSE_MAP, CV_INTER_LINEAR
#include <openpose/core/netCaffe.hpp>
#include <openpose/hand/handParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/cuda.hpp>
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/hand/handExtractor.hpp>
 
namespace op
{
    HandExtractor::HandExtractor(const Point<int>& netInputSize, const Point<int>& netOutputSize, const std::string& modelFolder, const int gpuId,
                                 const bool iterativeDetection) :
        mIterativeDetection{iterativeDetection},
        mNetOutputSize{netOutputSize},
        spNet{std::make_shared<NetCaffe>(std::array<int,4>{1, 3, mNetOutputSize.y, mNetOutputSize.x}, modelFolder + HAND_PROTOTXT,
                                         modelFolder + HAND_TRAINED_MODEL, gpuId)},
        spResizeAndMergeCaffe{std::make_shared<ResizeAndMergeCaffe<float>>()},
        spNmsCaffe{std::make_shared<NmsCaffe<float>>()},
        mHandImageCrop{mNetOutputSize.area()*3}
    {
        try
        {
error("Hands extraction is not implemented yet. COMING SOON!", __LINE__, __FUNCTION__, __FILE__);
            checkE(netOutputSize.x, netInputSize.x, "Net input and output size must be equal.", __LINE__, __FUNCTION__, __FILE__);
            checkE(netOutputSize.y, netInputSize.y, "Net input and output size must be equal.", __LINE__, __FUNCTION__, __FILE__);
            checkE(netInputSize.x, netInputSize.y, "Net input size must be squared.", __LINE__, __FUNCTION__, __FILE__);
            // Properties
            for (auto& property : mProperties)
                property = 0.;
            mProperties[(int)HandProperty::NMSThreshold] = HAND_DEFAULT_NMS_THRESHOLD;
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
            spNmsCaffe->Reshape({spHeatMapsBlob.get()}, {spPeaksBlob.get()}, HAND_MAX_PEAKS, HAND_NUMBER_PARTS+1);
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
error("Hands extraction is not implemented yet. COMING SOON!", __LINE__, __FUNCTION__, __FILE__);
            UNUSED(handRectangles);
            UNUSED(cvInputData);
            UNUSED(scaleInputToOutput);
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

    double HandExtractor::get(const HandProperty property) const
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

    void HandExtractor::set(const HandProperty property, const double value)
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

    void HandExtractor::increase(const HandProperty property, const double value)
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
                                            const cv::Mat& affineMatrix, const unsigned int handPeaksOffset)
    {
        try
        {
error("Hands extraction is not implemented yet. COMING SOON!", __LINE__, __FUNCTION__, __FILE__);
            UNUSED(handCurrent);
            UNUSED(scaleInputToOutput);
            UNUSED(person);
            UNUSED(affineMatrix);
            UNUSED(handPeaksOffset);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
