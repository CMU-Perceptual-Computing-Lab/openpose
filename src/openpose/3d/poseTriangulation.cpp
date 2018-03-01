#include <opencv2/calib3d/calib3d.hpp>
#include <openpose/3d/poseTriangulation.hpp>

namespace op
{
    double calcReprojectionError(const cv::Mat& X, const std::vector<cv::Mat>& M,
                                 const std::vector<cv::Point2d>& points2d)
    {
        try
        {
            auto averageError = 0.;
            for (auto i = 0u ; i < M.size() ; i++)
            {
                cv::Mat imageX = M[i] * X;
                imageX /= imageX.at<double>(2,0);
                const auto error = std::sqrt(std::pow(imageX.at<double>(0,0) -  points2d[i].x,2)
                                             + std::pow(imageX.at<double>(1,0) - points2d[i].y,2));
                // log("Error: " + std::to_string(error));
                averageError += error;
            }
            return averageError / M.size();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1.;
        }
    }

    void triangulate(cv::Mat& X, const std::vector<cv::Mat>& matrixEachCamera,
                     const std::vector<cv::Point2d>& pointsOnEachCamera)
    {
        try
        {
            // Security checks
            if (matrixEachCamera.empty() || matrixEachCamera.size() != pointsOnEachCamera.size())
                error("numberCameras.empty() || numberCameras.size() != pointsOnEachCamera.size()",
                          __LINE__, __FUNCTION__, __FILE__);
            // Create and fill A
            const auto numberCameras = (int)matrixEachCamera.size();
            cv::Mat A = cv::Mat::zeros(numberCameras*2, 4, CV_64F);
            for (auto i = 0 ; i < numberCameras ; i++)
            {
                cv::Mat temp = pointsOnEachCamera[i].x*matrixEachCamera[i].rowRange(2,3)
                             - matrixEachCamera[i].rowRange(0,1);
                temp.copyTo(A.rowRange(i*2, i*2+1));
                temp = pointsOnEachCamera[i].y*matrixEachCamera[i].rowRange(2,3) - matrixEachCamera[i].rowRange(1,2);
                temp.copyTo(A.rowRange(i*2+1, i*2+2));
            }
            // SVD on A
            cv::SVD svd{A};
            svd.solveZ(A,X);
            X /= X.at<double>(3);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    // TODO: ask for the missing function: TriangulationOptimization
    double triangulateWithOptimization(cv::Mat& X, const std::vector<cv::Mat>& matrixEachCamera,
                                       const std::vector<cv::Point2d>& pointsOnEachCamera)
    {
        try
        {
            triangulate(X, matrixEachCamera, pointsOnEachCamera);

            return 0.;
            // return calcReprojectionError(X, matrixEachCamera, pointsOnEachCamera);

            // //if (matrixEachCamera.size() >= 3)
            // //double beforeError = calcReprojectionError(&matrixEachCamera, pointsOnEachCamera, X);
            // double change = TriangulationOptimization(&matrixEachCamera, pointsOnEachCamera, X);
            // //double afterError = calcReprojectionError(&matrixEachCamera,pointsOnEachCamera,X);
            // //printfLog("!!Mine %.8f , inFunc %.8f \n",beforeError-afterError,change);
            // return change;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1.;
        }
    }

    Array<float> reconstructArray(const std::vector<Array<float>>& keypointsVector,
                                  const std::vector<cv::Mat>& matrixEachCamera)
    {
        try
        {
            Array<float> keypoints3D;
            // Security checks
            if (matrixEachCamera.size() < 2)
                error("Only 1 camera detected. The 3-D reconstruction module can only be used with > 1 cameras"
                      " simultaneously. E.g., using FLIR stereo cameras (`--flir_camera`).",
                      __LINE__, __FUNCTION__, __FILE__);
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
                const auto lastChannelLength = keypointsVector.at(0).getSize(2);
                // Create x-y vector from high score results
                const auto threshold = 0.2f;
                std::vector<int> indexesUsed;
                std::vector<std::vector<cv::Point2d>> xyPoints;
                for (auto part = 0; part < numberBodyParts; part++)
                {
                    // Create vector of points
                    auto missedPoint = false;
                    std::vector<cv::Point2d> xyPointsElement;
                    const auto baseIndex = part * lastChannelLength;
                    for (auto& keypoints : keypointsVector)
                    {
                        if (keypoints[baseIndex+2] > threshold)
                            xyPointsElement.emplace_back(cv::Point2d{ keypoints[baseIndex],
                                                                      keypoints[baseIndex+1]});
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
                keypoints3D.reset({ 1, numberBodyParts, 4 }, 0);
                if (!xyPoints.empty())
                {
                    // Do 3D reconstruction
                    std::vector<cv::Point3f> xyzPoints(xyPoints.size());
                    for (auto i = 0u; i < xyPoints.size(); i++)
                    {
                        cv::Mat X;
                        triangulateWithOptimization(X, matrixEachCamera, xyPoints[i]);
                        xyzPoints[i] = cv::Point3d{ X.at<double>(0), X.at<double>(1), X.at<double>(2) };
                    }

                    // 3D points to pose
                    // OpenCV alternative:
                    // http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#triangulatepoints
                    // cv::Mat reconstructedPoints{4, firstcv::Points.size(), CV_64F};
                    // cv::triangulatecv::Points(cv::Mat::eye(3,4, CV_64F), M_3_1, firstcv::Points, secondcv::Points,
                    //                           reconstructedcv::Points);
                    const auto lastChannelLength = keypoints3D.getSize(2);
                    for (auto index = 0u; index < indexesUsed.size(); index++)
                    {
                        if (std::isfinite(xyzPoints[index].x) && std::isfinite(xyzPoints[index].y)
                            && std::isfinite(xyzPoints[index].z))
                        {
                            const auto baseIndex = indexesUsed[index] * lastChannelLength;
                            keypoints3D[baseIndex] = xyzPoints[index].x;
                            keypoints3D[baseIndex + 1] = xyzPoints[index].y;
                            keypoints3D[baseIndex + 2] = xyzPoints[index].z;
                            keypoints3D[baseIndex + 3] = 1.f;
                        }
                    }
                }
            }
            return keypoints3D;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }
}
