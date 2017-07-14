#ifndef OPENPOSE_FACE_FACE_EXTRACTOR_HPP
#define OPENPOSE_FACE_FACE_EXTRACTOR_HPP

#include <atomic>
#include <thread>
#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/common.hpp>
#include <openpose/core/maximumCaffe.hpp>
#include <openpose/core/net.hpp>
#include <openpose/core/resizeAndMergeCaffe.hpp>

namespace op
{
    class OP_API FaceExtractor
    {
    public:
        explicit FaceExtractor(const Point<int>& netInputSize, const Point<int>& netOutputSize, const std::string& modelFolder, const int gpuId);

        void initializationOnThread();

        void forwardPass(const std::vector<Rectangle<float>>& faceRectangles, const cv::Mat& cvInputData, const float scaleInputToOutput);

        Array<float> getFaceKeypoints() const;

    private:
        const Point<int> mNetOutputSize;
        const Point<int> mOutputSize;
        std::shared_ptr<Net> spNet;
        std::shared_ptr<ResizeAndMergeCaffe<float>> spResizeAndMergeCaffe;
        std::shared_ptr<MaximumCaffe<float>> spMaximumCaffe;
        Array<float> mFaceImageCrop;
        Array<float> mFaceKeypoints;
        // Init with thread
        boost::shared_ptr<caffe::Blob<float>> spCaffeNetOutputBlob;
        std::shared_ptr<caffe::Blob<float>> spHeatMapsBlob;
        std::shared_ptr<caffe::Blob<float>> spPeaksBlob;
        std::thread::id mThreadId;

        void checkThread() const;

        DELETE_COPY(FaceExtractor);
    };
}

#endif // OPENPOSE_FACE_FACE_EXTRACTOR_HPP
