#ifndef OPENPOSE__HAND__HAND_EXTRACTOR_HPP
#define OPENPOSE__HAND__HAND_EXTRACTOR_HPP

#include <array>
#include <atomic>
#include <memory> // std::shared_ptr
#include <thread>
#include <opencv2/core/core.hpp>
#include "../../core/array.hpp"
#include "../../core/net.hpp"
#include "../../core/nmsCaffe.hpp"
#include "../../core/resizeAndMergeCaffe.hpp"
#include "../../pose/enumClasses.hpp"
#include "enumClasses.hpp"

namespace op
{
    namespace experimental
    {
        class HandExtractor
        {
        public:
            explicit HandExtractor(const std::string& modelFolder, const int gpuId, const PoseModel poseModel);

            void initializationOnThread();

            void forwardPass(const Array<float>& poseKeyPoints, const cv::Mat& cvInputData);

            Array<float> getHandKeyPoints() const;

            double get(const HandsProperty property) const;

            void set(const HandsProperty property, const double value);

            void increase(const HandsProperty property, const double value);

        private:
            const cv::Size mNetOutputSize;
            const cv::Size mOutputSize;
            const unsigned char mRWrist;
            const unsigned char mRElbow;
            const unsigned char mLWrist;
            const unsigned char mLElbow;
            const unsigned char mNeck;
            const unsigned char mHeadNose;
            std::array<std::atomic<double>, (int)HandsProperty::Size> mProperties;
            std::shared_ptr<Net> spNet;
            std::shared_ptr<ResizeAndMergeCaffe<float>> spResizeAndMergeCaffe;
            std::shared_ptr<NmsCaffe<float>> spNmsCaffe;
            Array<float> mLeftHandCrop;
            Array<float> mRightHandCrop;
            Array<float> mHands;
            float mScaleLeftHand;
            float mScaleRightHand;
            // Init with thread
            boost::shared_ptr<caffe::Blob<float>> spCaffeNetOutputBlob;
            std::shared_ptr<caffe::Blob<float>> spHeatMapsBlob;
            std::shared_ptr<caffe::Blob<float>> spPeaksBlob;
            std::thread::id mThreadId;

            void checkThread() const;

            DELETE_COPY(HandExtractor);
        };
    }
}

#endif // OPENPOSE__HAND__HAND_EXTRACTOR_HPP
