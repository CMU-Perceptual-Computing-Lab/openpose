#ifndef OPENPOSE_HAND_HAND_EXTRACTOR_HPP
#define OPENPOSE_HAND_HAND_EXTRACTOR_HPP

#include <atomic>
#include <thread>
#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/common.hpp>
#include <openpose/core/maximumCaffe.hpp>
#include <openpose/core/net.hpp>
#include <openpose/core/resizeAndMergeCaffe.hpp>

namespace op
{
    class OP_API HandExtractor
    {
    public:
        explicit HandExtractor(const Point<int>& netInputSize, const Point<int>& netOutputSize, const std::string& modelFolder, const int gpuId,
                               const unsigned short numberScales = 1, const float rangeScales = 0.4f);

        void initializationOnThread();

        void forwardPass(const std::vector<std::array<Rectangle<float>, 2>> handRectangles, const cv::Mat& cvInputData,
                         const float scaleInputToOutput);

        std::array<Array<float>, 2> getHandKeypoints() const;

    private:
        // const bool mMultiScaleDetection;
        const std::pair<unsigned short, float> mMultiScaleNumberAndRange;
        const Point<int> mNetOutputSize;
        std::shared_ptr<Net> spNet;
        std::shared_ptr<ResizeAndMergeCaffe<float>> spResizeAndMergeCaffe;
        std::shared_ptr<MaximumCaffe<float>> spMaximumCaffe;
        Array<float> mHandImageCrop;
        std::array<Array<float>, 2> mHandKeypoints;
        // Init with thread
        boost::shared_ptr<caffe::Blob<float>> spCaffeNetOutputBlob;
        std::shared_ptr<caffe::Blob<float>> spHeatMapsBlob;
        std::shared_ptr<caffe::Blob<float>> spPeaksBlob;
        std::thread::id mThreadId;

        void checkThread() const;

        void detectHandKeypoints(Array<float>& handCurrent, const float scaleInputToOutput, const int person, const cv::Mat& affineMatrix);

        DELETE_COPY(HandExtractor);
    };
}

#endif // OPENPOSE_HAND_HAND_EXTRACTOR_HPP
