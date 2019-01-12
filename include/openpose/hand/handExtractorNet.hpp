#ifndef OPENPOSE_HAND_HAND_EXTRACTOR_HPP
#define OPENPOSE_HAND_HAND_EXTRACTOR_HPP

#include <atomic>
#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/common.hpp>
#include <openpose/core/enumClasses.hpp>

namespace op
{
    /**
     * Hand keypoint extractor class.
     */
    class OP_API HandExtractorNet
    {
    public:
        /**
         * Constructor of the HandExtractorNet class.
         * @param netInputSize Size at which the cropped image (where the hand is located) is resized.
         * @param netOutputSize Size of the final results. At the moment, it must be equal than netOutputSize.
         * @param numberScales Number of scales to run. The more scales, the slower it will be but possibly also more
         * accurate.
         * @param rangeScales The range between the smaller and bigger scale.
         */
        explicit HandExtractorNet(const Point<int>& netInputSize, const Point<int>& netOutputSize,
                                  const unsigned short numberScales = 1, const float rangeScales = 0.4f,
                                  const std::vector<HeatMapType>& heatMapTypes = {},
                                  const ScaleMode heatMapScaleMode = ScaleMode::ZeroToOne);

        /**
         * Virtual destructor of the HandExtractorNet class.
         * Required to allow inheritance.
         */
        virtual ~HandExtractorNet();

        /**
         * This function must be call before using any other function. It must also be called inside the thread in
         * which the functions are going to be used.
         */
        void initializationOnThread();

        /**
         * This function extracts the hand keypoints for each detected hand in the image.
         * @param handRectangles location of the hands in the image. It is a length-variable std::vector, where
         * each index corresponds to a different person in the image. Internally the std::vector, a std::array of 2
         * elements: index 0 and 1 for left and right hand respectively. Inside each array element, a
         * op::Rectangle<float> (similar to cv::Rect for floating values) with the position of that hand (or 0,0,0,0 if
         * some hand is missing, e.g., if a specific person has only half of the body inside the image).
         * @param cvInputData Original image in cv::Mat format and BGR format.
         */
        virtual void forwardPass(const std::vector<std::array<Rectangle<float>, 2>> handRectangles,
                                 const cv::Mat& cvInputData) = 0;

        std::array<Array<float>, 2> getHeatMaps() const;

        /**
         * This function returns the hand keypoins. VERY IMPORTANT: use getHandKeypoints().clone() if the keypoints are
         * going to be edited in a different thread.
         * @return A std::array with all the left hand keypoints (index 0) and all the right ones (index 1). Each
         * Array<float> follows the pose structure, i.e., the first dimension corresponds to all the people in the
         * image, the second to each specific keypoint, and the third one to (x, y, score).
         */
        std::array<Array<float>, 2> getHandKeypoints() const;

        bool getEnabled() const;

        void setEnabled(const bool enabled);

    protected:
        const std::pair<unsigned short, float> mMultiScaleNumberAndRange;
        const Point<int> mNetOutputSize;
        Array<float> mHandImageCrop;
        std::array<Array<float>, 2> mHandKeypoints;
        // HeatMaps parameters
        const ScaleMode mHeatMapScaleMode;
        const std::vector<HeatMapType> mHeatMapTypes;
        std::array<Array<float>, 2> mHeatMaps;
        // Temporarily disable it
        std::atomic<bool> mEnabled;

        virtual void netInitializationOnThread() = 0;

    private:
        // Init with thread
        std::thread::id mThreadId;

        void checkThread() const;

        DELETE_COPY(HandExtractorNet);
    };
}

#endif // OPENPOSE_HAND_HAND_EXTRACTOR_HPP
