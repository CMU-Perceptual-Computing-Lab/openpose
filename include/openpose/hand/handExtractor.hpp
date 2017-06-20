#ifndef OPENPOSE_HAND_HAND_EXTRACTOR_HPP
#define OPENPOSE_HAND_HAND_EXTRACTOR_HPP

#include <array>
#include <atomic>
#include <memory> // std::shared_ptr
#include <thread>
#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/array.hpp>
#include <openpose/core/point.hpp>
#include <openpose/core/net.hpp>
#include <openpose/core/nmsCaffe.hpp>
#include <openpose/core/rectangle.hpp>
#include <openpose/core/resizeAndMergeCaffe.hpp>
#include <openpose/utilities/macros.hpp>
#include "enumClasses.hpp"

namespace op
{
    class HandExtractor
    {
    public:
        explicit HandExtractor(const Point<int>& netInputSize, const Point<int>& netOutputSize, const std::string& modelFolder, const int gpuId,
                               const bool iterativeDetection = false);

        void initializationOnThread();

        void forwardPass(const std::vector<std::array<Rectangle<float>, 2>> handRectangles, const cv::Mat& cvInputData,
                         const float scaleInputToOutput);

        std::array<Array<float>, 2> getHandKeypoints() const;

        double get(const HandProperty property) const;

        void set(const HandProperty property, const double value);

        void increase(const HandProperty property, const double value);

    private:
        const bool mIterativeDetection;
        const Point<int> mNetOutputSize;
        std::array<std::atomic<double>, (int)HandProperty::Size> mProperties;
        std::shared_ptr<Net> spNet;
        std::shared_ptr<ResizeAndMergeCaffe<float>> spResizeAndMergeCaffe;
        std::shared_ptr<NmsCaffe<float>> spNmsCaffe;
        Array<float> mHandImageCrop;
        std::array<Array<float>, 2> mHandKeypoints;
        // Init with thread
        boost::shared_ptr<caffe::Blob<float>> spCaffeNetOutputBlob;
        std::shared_ptr<caffe::Blob<float>> spHeatMapsBlob;
        std::shared_ptr<caffe::Blob<float>> spPeaksBlob;
        std::thread::id mThreadId;

        void checkThread() const;

        void detectHandKeypoints(Array<float>& handCurrent, const float scaleInputToOutput, const int person, const cv::Mat& affineMatrix,
                                 const unsigned int handPeaksOffset);

        DELETE_COPY(HandExtractor);
    };
}

#endif // OPENPOSE_HAND_HAND_EXTRACTOR_HPP
