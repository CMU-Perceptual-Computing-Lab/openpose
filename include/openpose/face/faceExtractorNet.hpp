#ifndef OPENPOSE_FACE_FACE_EXTRACTOR_HPP
#define OPENPOSE_FACE_FACE_EXTRACTOR_HPP

#include <thread>
#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/common.hpp>
#include <openpose/core/enumClasses.hpp>

namespace op
{
    /**
     * Face keypoint extractor class.
     */
    class OP_API FaceExtractorNet
    {
    public:
        /**
         * Constructor of the FaceExtractorNet class.
         * @param netInputSize Size at which the cropped image (where the face is located) is resized.
         * @param netOutputSize Size of the final results. At the moment, it must be equal than netOutputSize.
         */
        explicit FaceExtractorNet(const Point<int>& netInputSize, const Point<int>& netOutputSize,
                                  const std::vector<HeatMapType>& heatMapTypes = {},
                                  const ScaleMode heatMapScale = ScaleMode::ZeroToOne);

        /**
         * Virtual destructor of the HandExtractor class.
         * Required to allow inheritance.
         */
        virtual ~FaceExtractorNet();

        /**
         * This function must be call before using any other function. It must also be called inside the thread in
         * which the functions are going to be used.
         */
        void initializationOnThread();

        /**
         * This function extracts the face keypoints for each detected face in the image.
         * @param faceRectangles location of the faces in the image. It is a length-variable std::vector, where
         * each index corresponds to a different person in the image. Internally, a op::Rectangle<float>
         * (similar to cv::Rect for floating values) with the position of that face (or 0,0,0,0 if
         * some face is missing, e.g. if a specific person has only half of the body inside the image).
         * @param cvInputData Original image in cv::Mat format and BGR format.
         */
        virtual void forwardPass(const std::vector<Rectangle<float>>& faceRectangles, const cv::Mat& cvInputData) = 0;

        Array<float> getHeatMaps() const;

        /**
         * This function returns the face keypoins. VERY IMPORTANT: use getFaceKeypoints().clone() if the keypoints are
         * going to be edited in a different thread.
         * @return A Array with all the face keypoints. It follows the pose structure, i.e. the first dimension
         * corresponds to all the people in the image, the second to each specific keypoint, and the third one to
         * (x, y, score).
         */
        Array<float> getFaceKeypoints() const;

    protected:
        const Point<int> mNetOutputSize;
        Array<float> mFaceImageCrop;
        Array<float> mFaceKeypoints;
        // HeatMaps parameters
        Array<float> mHeatMaps;
        const ScaleMode mHeatMapScaleMode;
        const std::vector<HeatMapType> mHeatMapTypes;

        virtual void netInitializationOnThread() = 0;

    private:
        // Init with thread
        std::thread::id mThreadId;

        void checkThread() const;

        DELETE_COPY(FaceExtractorNet);
    };
}

#endif // OPENPOSE_FACE_FACE_EXTRACTOR_HPP
