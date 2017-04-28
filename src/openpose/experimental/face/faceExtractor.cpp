#include <opencv2/opencv.hpp> // CV_WARP_INVERSE_MAP, CV_INTER_LINEAR
#include "openpose/core/netCaffe.hpp"
#include "openpose/experimental/face/faceParameters.hpp"
#include "openpose/pose/poseParameters.hpp"
#include "openpose/utilities/cuda.hpp"
#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/utilities/fastMath.hpp"
#include "openpose/utilities/openCv.hpp"
#include "openpose/experimental/face/faceExtractor.hpp"
#include "openpose/experimental/face/faceRenderGpu.hpp" // For commented debugging section
 
namespace op
{
    namespace experimental
    {
        FaceExtractor::FaceExtractor(const std::string& modelFolder, const int gpuId, const PoseModel poseModel) :
            mNetOutputSize{368, 368},
            mOutputSize{1280, 720},
            mNeck{poseBodyPartMapStringToKey(poseModel, "Neck")},
            mNose{poseBodyPartMapStringToKey(poseModel, std::vector<std::string>{"Nose", "Head"})},
            mLEar{poseBodyPartMapStringToKey(poseModel, std::vector<std::string>{"LEar", "Head"})},
            mREar{poseBodyPartMapStringToKey(poseModel, std::vector<std::string>{"REar", "Head"})},
            mLEye{poseBodyPartMapStringToKey(poseModel, std::vector<std::string>{"LEye", "Head"})},
            mREye{poseBodyPartMapStringToKey(poseModel, std::vector<std::string>{"REye", "Head"})},
            spNet{std::make_shared<NetCaffe>(std::array<int,4>{1, 3, mNetOutputSize.height, mNetOutputSize.width}, modelFolder + FACE_PROTOTXT, modelFolder + FACE_TRAINED_MODEL, gpuId)},
            spResizeAndMergeCaffe{std::make_shared<ResizeAndMergeCaffe<float>>()},
            spNmsCaffe{std::make_shared<NmsCaffe<float>>()},
            mFaceImageCrop{mNetOutputSize.area()*3}
        {
            try
            {
error("Hands and face extraction is not implemented yet. COMING SOON!", __LINE__, __FUNCTION__, __FILE__);
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
                spNmsCaffe->Reshape({spHeatMapsBlob.get()}, {spPeaksBlob.get()}, FACE_MAX_PEAKS, FACE_NUMBER_PARTS);
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
     
                log("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void FaceExtractor::forwardPass(const Array<float>& poseKeyPoints, const cv::Mat& cvInputData)
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
 
        Array<float> FaceExtractor::getFaceKeyPoints() const
        {
            try
            {
                checkThread();
                return mFaceKeyPoints;
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
}
