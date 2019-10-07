#include <openpose/face/faceDetectorOpenCV.hpp>
#include <opencv2/objdetect/objdetect.hpp> // cv::CascadeClassifier
#include <openpose/pose/poseParameters.hpp>
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp>

namespace op
{
    struct FaceDetectorOpenCV::ImplFaceDetectorOpenCV
    {
        cv::CascadeClassifier mFaceCascade;
    };

    FaceDetectorOpenCV::FaceDetectorOpenCV(const std::string& modelFolder) :
        upImpl{new ImplFaceDetectorOpenCV{}}
    {
        try
        {
            const std::string faceDetectorModelPath{modelFolder + "face/haarcascade_frontalface_alt.xml"};
            if (!upImpl->mFaceCascade.load(faceDetectorModelPath))
                error("Face detector model not found at: " + faceDetectorModelPath, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    FaceDetectorOpenCV::~FaceDetectorOpenCV()
    {
    }

    std::vector<Rectangle<float>> FaceDetectorOpenCV::detectFaces(const Matrix& inputData)
    {
        try
        {
            cv::Mat cvInputData = OP_OP2CVCONSTMAT(inputData);
            // Image to grey and pyrDown
            cv::Mat frameGray;
            cv::cvtColor(cvInputData, frameGray, cv::COLOR_BGR2GRAY);
            auto multiplier = 1.f;
            while (frameGray.cols * frameGray.rows > 640*360)
            {
                cv::pyrDown(frameGray, frameGray);
                multiplier *= 2.f;
            }
            // Face detection - Example from:
            // http://docs.opencv.org/2.4/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html
            std::vector<cv::Rect> detectedFaces;
            upImpl->mFaceCascade.detectMultiScale(frameGray, detectedFaces, 1.2, 3, 0|CV_HAAR_SCALE_IMAGE);
            // Rescale rectangles
            std::vector<Rectangle<float>> faceRectangles(detectedFaces.size());
            for(auto i = 0u; i < detectedFaces.size(); i++)
            {
                // Enlarge detected rectangle by 1.5x, so that it covers the whole face
                faceRectangles.at(i).x = detectedFaces.at(i).x - 0.25f*detectedFaces.at(i).width;
                faceRectangles.at(i).y = detectedFaces.at(i).y - 0.25f*detectedFaces.at(i).height;
                faceRectangles.at(i).width = 1.5f*detectedFaces.at(i).width;
                faceRectangles.at(i).height = 1.5f*detectedFaces.at(i).height;
                faceRectangles.at(i) *= multiplier;
            }
            return faceRectangles;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }
}
