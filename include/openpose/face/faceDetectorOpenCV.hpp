#ifndef OPENPOSE_FACE_FACE_DETECTOR_OPENCV_HPP
#define OPENPOSE_FACE_FACE_DETECTOR_OPENCV_HPP

#include <openpose/core/common.hpp>

namespace op
{
    class OP_API FaceDetectorOpenCV
    {
    public:
        explicit FaceDetectorOpenCV(const std::string& modelFolder);

        virtual ~FaceDetectorOpenCV();

        // No thread-save
        std::vector<Rectangle<float>> detectFaces(const Matrix& inputData);

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplFaceDetectorOpenCV;
        std::unique_ptr<ImplFaceDetectorOpenCV> upImpl;

        DELETE_COPY(FaceDetectorOpenCV);
    };
}

#endif // OPENPOSE_FACE_FACE_DETECTOR_OPENCV_HPP
