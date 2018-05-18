#ifdef WITH_CERES
    #include <ceres/ceres.h>
    #include <ceres/rotation.h>
#endif
#include <opencv2/calib3d/calib3d.hpp>
#include <openpose/3d/poseTriangulation.hpp>

namespace op
{
    double calcReprojectionError(const cv::Mat& reconstructedPoint, const std::vector<cv::Mat>& cameraMatrices,
                                 const std::vector<cv::Point2d>& pointsOnEachCamera)
    {
        try
        {
            auto averageError = 0.;
            for (auto i = 0u ; i < cameraMatrices.size() ; i++)
            {
                cv::Mat imageX = cameraMatrices[i] * reconstructedPoint;
                imageX /= imageX.at<double>(2,0);
                const auto error = std::sqrt(std::pow(imageX.at<double>(0,0) -  pointsOnEachCamera[i].x,2)
                                             + std::pow(imageX.at<double>(1,0) - pointsOnEachCamera[i].y,2));
                // log("Error: " + std::to_string(error));
                averageError += error;
            }
            return averageError / cameraMatrices.size();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1.;
        }
    }

    #ifdef WITH_CERES
        // Nonlinear Optimization for 3D Triangulation
        struct ReprojectionErrorForTriangulation
        {
            ReprojectionErrorForTriangulation(const double x, const double y, const double* const param)
            {
                observed_x = x;
                observed_y = y;
                memcpy(camParam,param,sizeof(double)*12);
            }

            template <typename T>
            bool operator()(const T* const pt,
                            T* residuals) const ;

            inline virtual bool Evaluate(double const* const* pt,
                                         double* residuals,
                                         double** jacobians) const;

            double observed_x;
            double observed_y;
            double camParam[12];
        };

        template <typename T>
        bool ReprojectionErrorForTriangulation::operator()(const T* const pt,
                                                           T* residuals) const
        {
            try
            {
                T predicted[3] = {
                    T(camParam[0])*pt[0] + T(camParam[1])*pt[1] + T(camParam[2])*pt[2] + T(camParam[3]),
                    T(camParam[4])*pt[0] + T(camParam[5])*pt[1] + T(camParam[6])*pt[2] + T(camParam[7]),
                    T(camParam[8])*pt[0] + T(camParam[9])*pt[1] + T(camParam[10])*pt[2] + T(camParam[11])};

                predicted[0] /= predicted[2];
                predicted[1] /= predicted[2];

                residuals[0] = T(observed_x) - predicted[0];
                residuals[1] = T(observed_y) - predicted[1];

                // residuals[0] = T(pow(predicted[0] - observed_x,2) + pow(predicted[1] - observed_y,2));
                // residuals[0] = -pow(predicted[0] - T(observed_x),2);
                // residuals[1] = -pow(predicted[1] - T(observed_y),2);

                return  true;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return false;
            }
        }

        bool ReprojectionErrorForTriangulation::Evaluate(double const* const* pt,
            double* residuals,
            double** jacobians) const
        {
            try
            {
                UNUSED(jacobians);

                double predicted[3] = {
                    camParam[0]*pt[0][0] + camParam[1]*pt[0][1] + camParam[2]*pt[0][2] + camParam[3],
                    camParam[4]*pt[0][0] + camParam[5]*pt[0][1] + camParam[6]*pt[0][2] + camParam[7],
                    camParam[8]*pt[0][0] + camParam[9]*pt[0][1] + camParam[10]*pt[0][2] + camParam[11]};

                predicted[0] /= predicted[2];
                predicted[1] /= predicted[2];

                // residuals[0] = predicted[0] - observed_x;
                // residuals[1] = predicted[1] - observed_y;

                residuals[0] = std::sqrt(std::pow(predicted[0] - observed_x,2) + std::pow(predicted[1] - observed_y,2));
                // log("Residuals:");
                // residuals[0]= pow(predicted[0] - (observed_x),2);
                // residuals[1]= pow(predicted[1] - (observed_y),2);

                return true;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return false;
            }
        }
    #endif

    double triangulate(cv::Mat& reconstructedPoint, const std::vector<cv::Mat>& cameraMatrices,
                       const std::vector<cv::Point2d>& pointsOnEachCamera)
    {
        try
        {
            // Security checks
            if (cameraMatrices.size() != pointsOnEachCamera.size())
                error("numberCameras.size() != pointsOnEachCamera.size() (" + std::to_string(cameraMatrices.size())
                      + " vs. " + std::to_string(pointsOnEachCamera.size()) + ").",
                      __LINE__, __FUNCTION__, __FILE__);
            if (cameraMatrices.empty())
                error("numberCameras.empty()",
                      __LINE__, __FUNCTION__, __FILE__);
            // Create and fill A
            const auto numberCameras = (int)cameraMatrices.size();
            cv::Mat A = cv::Mat::zeros(numberCameras*2, 4, CV_64F);
            for (auto i = 0 ; i < numberCameras ; i++)
            {
                cv::Mat temp = pointsOnEachCamera[i].x*cameraMatrices[i].rowRange(2,3)
                             - cameraMatrices[i].rowRange(0,1);
                temp.copyTo(A.rowRange(i*2, i*2+1));
                temp = pointsOnEachCamera[i].y*cameraMatrices[i].rowRange(2,3) - cameraMatrices[i].rowRange(1,2);
                temp.copyTo(A.rowRange(i*2+1, i*2+2));
            }
            // SVD on A
            cv::SVD svd{A};
            svd.solveZ(A,reconstructedPoint);
            reconstructedPoint /= reconstructedPoint.at<double>(3);

            return calcReprojectionError(reconstructedPoint, cameraMatrices, pointsOnEachCamera);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1.;
        }
    }

    double triangulateWithOptimization(cv::Mat& reconstructedPoint, const std::vector<cv::Mat>& cameraMatrices,
                                       const std::vector<cv::Point2d>& pointsOnEachCamera)
    {
        try
        {
            // Information for 3 cameras:
            //     - Speed: triangulate ~0.01 ms vs. optimization ~0.2 ms
            //     - Accuracy: initial reprojection error ~14-21, reduced ~5% with non-linear optimization

            // Basic triangulation
            triangulate(reconstructedPoint, cameraMatrices, pointsOnEachCamera);

            #ifdef WITH_CERES
                // Slow equivalent: double paramX[3]; paramX[i] = reconstructedPoint.at<double>(i);
                double* paramX = (double*)reconstructedPoint.data;
                ceres::Problem problem;
                for (auto i = 0u; i < cameraMatrices.size(); ++i)
                {
                    // Slow equivalent: double camParam[12]; memcpy(camParam, cameraMatrices[i].data, sizeof(double)*12);
                    const double* const camParam = (double*)cameraMatrices[i].data;
                    // Each Residual block takes a point and a camera as input and outputs a 2
                    // dimensional residual. Internally, the cost function stores the observed
                    // image location and compares the reprojection against the observation.
                    ceres::CostFunction* cost_function =
                        new ceres::AutoDiffCostFunction<ReprojectionErrorForTriangulation, 2, 3>(
                            new ReprojectionErrorForTriangulation(
                                pointsOnEachCamera[i].x, pointsOnEachCamera[i].y, camParam));
                    // Add to problem
                    problem.AddResidualBlock(cost_function,
                        //NULL, //squared loss
                        new ceres::HuberLoss(2.0),
                        paramX); // paramX[0,1,2]
                }

                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
                // options.minimizer_progress_to_stdout = true;
                // options.parameter_tolerance = 1e-20;
                // options.function_tolerance = 1e-20;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                // if (summary.initial_cost > summary.final_cost)
                //     std::cout << summary.FullReport() << "\n";

                // const auto reprojectionErrorDecrease = std::sqrt((summary.initial_cost - summary.final_cost)/double(cameraMatrices.size()));
                // return reprojectionErrorDecrease;
            #endif
            // assert(reconstructedPoint.at<double>(3) == 1.);

            return calcReprojectionError(reconstructedPoint, cameraMatrices, pointsOnEachCamera);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1.;
        }
    }

    PoseTriangulation::PoseTriangulation(const int minViews3d) :
        mMinViews3d{minViews3d}
    {
        try
        {
            // Security checks
            if (0 <= mMinViews3d && mMinViews3d < 2)
                error("Minimum number of views must be at least 2 (e.g., `--3d_min_views 2`) or negative.",
                      __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Array<float> PoseTriangulation::reconstructArray(const std::vector<Array<float>>& keypointsVector,
                                                     const std::vector<cv::Mat>& cameraMatrices) const
    {
        try
        {
            Array<float> keypoints3D;
            // Security checks
            if (cameraMatrices.size() < 2)
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
                std::vector<std::vector<cv::Mat>> cameraMatricesPerPoint;
                for (auto part = 0; part < numberBodyParts; part++)
                {
                    // Create vector of points
                    // auto missedPoint = false;
                    std::vector<cv::Point2d> xyPointsElement;
                    std::vector<cv::Mat> cameraMatricesElement;
                    const auto baseIndex = part * lastChannelLength;
                    // for (auto& keypoints : keypointsVector)
                    for (auto i = 0u ; i < keypointsVector.size() ; i++)
                    {
                        const auto& keypoints = keypointsVector[i];
                        if (keypoints[baseIndex+2] > threshold)
                        {
                            xyPointsElement.emplace_back(cv::Point2d{keypoints[baseIndex],
                                                                     keypoints[baseIndex+1]});
                            cameraMatricesElement.emplace_back(cameraMatrices[i]);
                        }
                    }
                    // If visible from all views (mMinViews3d < 0)
                    // or if visible for at least mMinViews3d views
                    if ((mMinViews3d < 0 && cameraMatricesElement.size() == cameraMatrices.size())
                        || (mMinViews3d > 1 && mMinViews3d <= (int)xyPointsElement.size()))
                    {
                        indexesUsed.emplace_back(part);
                        xyPoints.emplace_back(xyPointsElement);
                        cameraMatricesPerPoint.emplace_back(cameraMatricesElement);
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
                        cv::Mat reconstructedPoint;
                        triangulateWithOptimization(reconstructedPoint, cameraMatricesPerPoint[i], xyPoints[i]);
                        xyzPoints[i] = cv::Point3d{reconstructedPoint.at<double>(0), reconstructedPoint.at<double>(1),
                            reconstructedPoint.at<double>(2)};
                    }

                    // 3D points to pose
                    // OpenCV alternative:
                    // http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#triangulatepoints
                    // cv::Mat reconstructedPoints{4, firstcv::Points.size(), CV_64F};
                    // cv::triangulatePoints(cv::Mat::eye(3,4, CV_64F), M_3_1, firstcv::Points, secondcv::Points,
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
