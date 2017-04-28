#include <opencv2/opencv.hpp> // CV_WARP_INVERSE_MAP, CV_INTER_LINEAR
#include "openpose/core/netCaffe.hpp"
#include "openpose/experimental/hand/handParameters.hpp"
#include "openpose/pose/poseParameters.hpp"
#include "openpose/utilities/cuda.hpp"
#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/utilities/fastMath.hpp"
#include "openpose/utilities/openCv.hpp"
#include "openpose/experimental/hand/handExtractor.hpp"
// #include "openpose/experimental/hand/handRenderGpu.hpp" // For commented debugging section
 
namespace op
{
    namespace experimental
    {
        HandExtractor::HandExtractor(const std::string& modelFolder, const int gpuId, const PoseModel poseModel) :
            mNetOutputSize{368, 368},
            mOutputSize{1280, 720},
            mRWrist{poseBodyPartMapStringToKey(poseModel, "RWrist")},
            mRElbow{poseBodyPartMapStringToKey(poseModel, "RElbow")},
            mLWrist{poseBodyPartMapStringToKey(poseModel, "LWrist")},
            mLElbow{poseBodyPartMapStringToKey(poseModel, "LElbow")},
            mNeck{poseBodyPartMapStringToKey(poseModel, "Neck")},
            mHeadNose{poseBodyPartMapStringToKey(poseModel, std::vector<std::string>{"Nose", "Head"})},
            spNet{std::make_shared<NetCaffe>(std::array<int,4>{2, 3, mNetOutputSize.height, mNetOutputSize.width}, modelFolder + HAND_PROTOTXT, modelFolder + HAND_TRAINED_MODEL, gpuId)},
            spResizeAndMergeCaffe{std::make_shared<ResizeAndMergeCaffe<float>>()},
            spNmsCaffe{std::make_shared<NmsCaffe<float>>()},
            mLeftHandCrop{mNetOutputSize.area()*3},
            mRightHandCrop{mLeftHandCrop.getSize()},
            mScaleLeftHand{100.f},
            mScaleRightHand{mScaleLeftHand}
        {
            try
            {
error("Hands and face extraction is not implemented yet. COMING SOON!", __LINE__, __FUNCTION__, __FILE__);
                // Properties
                for (auto& property : mProperties)
                    property = 0.;
                mProperties[(int)HandsProperty::NMSThreshold] = HAND_DEFAULT_NMS_THRESHOLD;
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
                const bool mergeFirstDimension = false;
                spResizeAndMergeCaffe->Reshape({spCaffeNetOutputBlob.get()}, {spHeatMapsBlob.get()}, HAND_CCN_DECREASE_FACTOR, mergeFirstDimension);
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);

                // Pose extractor blob and layer
                spPeaksBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
                spNmsCaffe->Reshape({spHeatMapsBlob.get()}, {spPeaksBlob.get()}, HAND_MAX_PEAKS, HAND_NUMBER_PARTS);
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
     
                log("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }
 
        void HandExtractor::forwardPass(const Array<float>& poseKeyPoints, const cv::Mat& cvInputData)
        {
            try
            {
UNUSED(poseKeyPoints);
UNUSED(cvInputData);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }
 
        Array<float> HandExtractor::getHandKeyPoints() const
        {
            try
            {
                checkThread();
                return mHands;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return Array<float>{};
            }
        }
 
        double HandExtractor::get(const HandsProperty property) const
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
 
        void HandExtractor::set(const HandsProperty property, const double value)
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
 
        void HandExtractor::increase(const HandsProperty property, const double value)
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
    }
}
