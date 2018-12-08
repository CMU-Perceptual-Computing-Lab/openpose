#ifndef OPENPOSE_FACE_FACE_DETECTOR_OPENCV_HPP
#define OPENPOSE_FACE_FACE_DETECTOR_OPENCV_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <openpose/core/common.hpp>

namespace op
{
    class OP_API FaceDetectorOpenCV
    {
    public:
        explicit FaceDetectorOpenCV(const std::string& modelFolder);

        virtual ~FaceDetectorOpenCV();

        // No thread-save
        std::vector<Rectangle<float>> detectFaces(const cv::Mat& cvInputData);

    private:
        cv::CascadeClassifier mFaceCascade;

        DELETE_COPY(FaceDetectorOpenCV);
    };
}

#endif // OPENPOSE_FACE_FACE_DETECTOR_OPENCV_HPP
