#ifndef OPENPOSE_FACE_FACE_DETECTOR_OPENCV_HPP
#define OPENPOSE_FACE_FACE_DETECTOR_OPENCV_HPP

#include <openpose/core/common.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>

namespace op
{
    class OP_API FaceDetectorOpenCV
    {
    public:
        explicit FaceDetectorOpenCV();

        std::vector<Rectangle<float>> detectFacesOpenCV(const cv::Mat& cvInputData, const float scaleInputToOutput);

    private:
        cv::CascadeClassifier face_cascade;
        DELETE_COPY(FaceDetectorOpenCV);
    };
}

#endif // OPENPOSE_FACE_FACE_DETECTOR_OPENCV_HPP
