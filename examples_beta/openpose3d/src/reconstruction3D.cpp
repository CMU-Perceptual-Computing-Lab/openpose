#include <opencv2/opencv.hpp>
#include <openpose3d/cameraParameters.hpp>
#include <openpose3d/pointGrey.hpp>
#include <openpose3d/reconstruction3D.hpp>

double calcReprojectionError(const cv::Mat& X, const std::vector<cv::Mat>& M, const std::vector<cv::Point2d>& pt2D)
{
    auto averageError = 0.;
    for(unsigned int i = 0 ; i < M.size() ; i++)
    {
        cv::Mat imageX = M[i] * X;
        imageX /= imageX.at<double>(2,0);
        const auto error = std::sqrt(std::pow(imageX.at<double>(0,0) -  pt2D[i].x,2) + std::pow(imageX.at<double>(1,0) - pt2D[i].y,2));
        //log("Error: " + std::to_string(error));
        averageError += error;
    }
    return averageError / M.size();
}

void triangulate(cv::Mat& X, const std::vector<cv::Mat>& matrixEachCamera, const std::vector<cv::Point2d>& pointOnEachCamera)
{
    // Security checks
    if (matrixEachCamera.empty() || matrixEachCamera.size() != pointOnEachCamera.size())
        op::error("numberCameras.empty() || numberCameras.size() != pointOnEachCamera.size()", __LINE__, __FUNCTION__, __FILE__);
    // Create and fill A
    const auto numberCameras = (int)matrixEachCamera.size();
    cv::Mat A = cv::Mat::zeros(numberCameras*2, 4, CV_64F);
    for (auto i = 0 ; i < numberCameras ; i++)
    {
        cv::Mat temp = pointOnEachCamera[i].x*matrixEachCamera[i].rowRange(2,3) - matrixEachCamera[i].rowRange(0,1);
        temp.copyTo(A.rowRange(i*2,i*2+1));
        temp = pointOnEachCamera[i].y*matrixEachCamera[i].rowRange(2,3) - matrixEachCamera[i].rowRange(1,2);
        temp.copyTo(A.rowRange(i*2+1,i*2+2));
    }
    // SVD on A
    cv::SVD svd{A};
    svd.solveZ(A,X);
    X /= X.at<double>(3);
}

// TODO: ask Hanbyul for the missing function: TriangulationOptimization
double triangulateWithOptimization(cv::Mat& X, const std::vector<cv::Mat>& matrixEachCamera, const std::vector<cv::Point2d>& pointOnEachCamera)
{
    triangulate(X, matrixEachCamera, pointOnEachCamera);

    // //if (matrixEachCamera.size() >= 3)
    // //double beforeError = calcReprojectionError(&matrixEachCamera, pointOnEachCamera, X);
    // double change = TriangulationOptimization(&matrixEachCamera, pointOnEachCamera, X);
    // //double afterError = calcReprojectionError(&matrixEachCamera,pointOnEachCamera,X);
    // //printfLog("!!Mine %.8f , inFunc %.8f \n",beforeError-afterError,change);
    // return change;
    return 0.;
}

void reconstructArray(op::Array<float>& keypoints3D, const std::vector<op::Array<float>>& keypointsVector)
{
    // Get number body parts
    auto detectionMissed = false;
    for (auto& keypoints : keypointsVector)
    {
        if (keypoints.empty())
        {
            detectionMissed = true;
            break;
        }
    }
    // If at least one keypoints element not empty
    if (!detectionMissed)
    {
        const auto numberBodyParts = keypointsVector.at(0).getSize(1);
        // Create x-y vector from high score results
        const auto threshold = 0.2f;
        std::vector<int> indexesUsed;
        std::vector<std::vector<cv::Point2d>> xyPoints;
        for (auto part = 0; part < numberBodyParts; part++)
        {
            // Create vector of points
            auto missedPoint = false;
            std::vector<cv::Point2d> xyPointsElement;
            for (auto& keypoints : keypointsVector)
            {
                if (keypoints[{0, part, 2}] > threshold)
                    xyPointsElement.emplace_back(cv::Point2d{ keypoints[{0, part, 0}], keypoints[{0, part, 1}]});
                else
                {
                    missedPoint = true;
                    break;
                }
            }
            if (!missedPoint)
            {
                indexesUsed.emplace_back(part);
                xyPoints.emplace_back(xyPointsElement);
            }
        }
        // 3D reconstruction
        if (!xyPoints.empty())
        {
            // Do 3D reconstruction
            std::vector<cv::Point3f> xyzPoints(xyPoints.size());
            for (auto i = 0u; i < xyPoints.size(); i++)
            {
                cv::Mat X;
                triangulateWithOptimization(X, M_EACH_CAMERA, xyPoints[i]);
                xyzPoints[i] = cv::Point3d{ X.at<double>(0), X.at<double>(1), X.at<double>(2) };
            }

            // 3D points to pose
            // OpenCV alternative:
            // // http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#triangulatepoints
            // cv::Mat reconstructedPoints{4, firstcv::Points.size(), CV_64F};
            // cv::triangulatecv::Points(cv::Mat::eye(3,4, CV_64F), M_3_1, firstcv::Points, secondcv::Points, reconstructedcv::Points);
            keypoints3D = op::Array<float>{ { 1, numberBodyParts, 4 }, 0 };
            for (auto index = 0u; index < indexesUsed.size(); index++)
            {
                auto& xValue = keypoints3D[{0, indexesUsed[index], 0}];
                auto& yValue = keypoints3D[{0, indexesUsed[index], 1}];
                auto& zValue = keypoints3D[{0, indexesUsed[index], 2}];
                auto& scoreValue = keypoints3D[{0, indexesUsed[index], 3}];
                if (std::isfinite(xyzPoints[index].x) && std::isfinite(xyzPoints[index].y) && std::isfinite(xyzPoints[index].z))
                {
                    xValue = xyzPoints[index].x;
                    yValue = xyzPoints[index].y;
                    zValue = xyzPoints[index].z;
                    scoreValue = 1.f;
                }
            }
        }
    }
}

void WReconstruction3D::work(std::shared_ptr<std::vector<Datum3D>>& datumsPtr)
{
    // User's post-processing (after OpenPose processing & before OpenPose outputs) here
        // datum.cvOutputData: rendered frame with pose or heatmaps
        // datum.poseKeypoints: Array<float> with the estimated pose
    try
    {
        // Profiling speed
        const auto profilerKey = op::Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
        if (datumsPtr != nullptr && /*!datumsPtr->empty() &&*/ datumsPtr->size() == 3)
        {
            std::vector<op::Array<float>> poseKeypointVector;
            std::vector<op::Array<float>> faceKeypointVector;
            std::vector<op::Array<float>> leftHandKeypointVector;
            std::vector<op::Array<float>> rightHandKeypointVector;
            for (auto& datumsElement : *datumsPtr)
            {
                poseKeypointVector.emplace_back(datumsElement.poseKeypoints);
                faceKeypointVector.emplace_back(datumsElement.faceKeypoints);
                leftHandKeypointVector.emplace_back(datumsElement.handKeypoints[0]);
                rightHandKeypointVector.emplace_back(datumsElement.handKeypoints[1]);
            }
			// Pose 3-D reconstruction
			reconstructArray(datumsPtr->at(0).poseKeypoints3D, poseKeypointVector);
			// Face 3-D reconstruction
			reconstructArray(datumsPtr->at(0).faceKeypoints3D, faceKeypointVector);
			// Left hand 3-D reconstruction
			reconstructArray(datumsPtr->at(0).leftHandKeypoints3D, leftHandKeypointVector);
			// Right hand 3-D reconstruction
			reconstructArray(datumsPtr->at(0).rightHandKeypoints3D, rightHandKeypointVector);
            // Profiling speed
            op::Profiler::timerEnd(profilerKey);
            op::Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__, 100);
        }
    }
    catch (const std::exception& e)
    {
        op::log("Some kind of unexpected error happened.");
        this->stop();
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}
