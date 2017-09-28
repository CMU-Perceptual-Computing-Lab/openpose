#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/face/faceDetectorOpenCV.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>

 
namespace op
{
    FaceDetectorOpenCV::FaceDetectorOpenCV()
    {
        try
        {
            if ( !face_cascade.load( "./models/face/haarcascade_frontalface_alt.xml" ) )
            {
                error("Cannot load model for Haar Cascaded face detector!", __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::vector<Rectangle<float>> FaceDetectorOpenCV::detectFacesOpenCV(const cv::Mat& cvInputData, const float scaleInputToOutput)
    {
        try
        {
            std::vector<cv::Rect> detectedFaces;
            cv::Mat frameGray;
            cv::cvtColor( cvInputData, frameGray, cv::COLOR_BGR2GRAY );
            face_cascade.detectMultiScale( frameGray, detectedFaces, 1.1, 3, 0|cv::CASCADE_SCALE_IMAGE );

            std::vector<Rectangle<float>> faceRectangles(detectedFaces.size());
            for(auto i = 0u; i < detectedFaces.size(); i++)
            {
                faceRectangles.at(i).x = detectedFaces.at(i).x / scaleInputToOutput;
                faceRectangles.at(i).y = detectedFaces.at(i).y / scaleInputToOutput;
                faceRectangles.at(i).width = detectedFaces.at(i).width / scaleInputToOutput;
                faceRectangles.at(i).height = detectedFaces.at(i).height / scaleInputToOutput;
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
