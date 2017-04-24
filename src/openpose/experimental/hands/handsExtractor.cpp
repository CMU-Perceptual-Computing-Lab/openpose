#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/experimental/hands/handsExtractor.hpp"

namespace op
{
    namespace experimental
    {
        HandsExtractor::HandsExtractor(const std::string& modelFolder, const int gpuId, const PoseModel poseModel) :
            mNetOutputSize{368, 368},
            mOutputSize{1280, 720},
            mRWrist{0},
            mRElbow{0},
            mLWrist{0},
            mLElbow{0},
            mNeck{0},
            mHeadNose{0},
            spNet{},
            spResizeAndMergeCaffe{},
            spNmsCaffe{},
            mLeftHandCrop{mNetOutputSize.area()*3},
            mRightHandCrop{mLeftHandCrop.getSize()},
            mScaleLeftHand{0.f},
            mScaleRightHand{0.f}
        {
            UNUSED(modelFolder);
            UNUSED(gpuId);
            UNUSED(poseModel);
            error("Hands code is not ready yet. A first beta version will be included in around 1-2 months. Please, set extractAndRenderHands = false in the OpenPose wrapper.",
                  __LINE__, __FUNCTION__, __FILE__);
        }

        void HandsExtractor::initializationOnThread()
        {
            error("Hands code is not ready yet. A first beta version will be included in around 1-2 months. Please, set extractAndRenderHands = false in the OpenPose wrapper.",
                  __LINE__, __FUNCTION__, __FILE__);
        }

        void HandsExtractor::forwardPass(const Array<float>& pose, const cv::Mat& cvInputData)
        {
            try
            {
                UNUSED(pose);
                UNUSED(cvInputData);
                error("Hands code is not ready yet. A first beta version will be included in around 1-2 months. Please, set extractAndRenderHands = false in the OpenPose wrapper.",
                      __LINE__, __FUNCTION__, __FILE__);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        Array<float> HandsExtractor::getHands() const
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

        double HandsExtractor::get(const HandsProperty property) const
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

        void HandsExtractor::set(const HandsProperty property, const double value)
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

        void HandsExtractor::increase(const HandsProperty property, const double value)
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

        void HandsExtractor::checkThread() const
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
