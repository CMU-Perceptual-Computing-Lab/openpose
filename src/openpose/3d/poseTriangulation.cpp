#include <numeric> // std::accumulate
#ifdef USE_CERES
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

    #ifdef USE_CERES
        // Nonlinear Optimization for 3D Triangulation
        struct ReprojectionErrorForTriangulation
        {
            ReprojectionErrorForTriangulation(const double x, const double y, const double* const param) :
                observed_x{x},
                observed_y{y}
            {
                memcpy(camParam, param, sizeof(double)*12);
            }

            template <typename T>
            bool operator()(const T* const pt,
                            T* residuals) const ;

            inline virtual bool Evaluate(double const* const* pt,
                                         double* residuals,
                                         double** jacobians) const;

            const double observed_x;
            const double observed_y;
            double camParam[12];
        };

        template <typename T>
        bool ReprojectionErrorForTriangulation::operator()(const T* const pt,
                                                           T* residuals) const
        {
            try
            {
                const T predicted[3] = {
                    T(camParam[0])*pt[0] + T(camParam[1])*pt[1] + T(camParam[2])*pt[2] + T(camParam[3]),
                    T(camParam[4])*pt[0] + T(camParam[5])*pt[1] + T(camParam[6])*pt[2] + T(camParam[7]),
                    T(camParam[8])*pt[0] + T(camParam[9])*pt[1] + T(camParam[10])*pt[2] + T(camParam[11])};

                residuals[0] = T(observed_x) - predicted[0] / predicted[2];
                residuals[1] = T(observed_y) - predicted[1] / predicted[2];

                // residuals[0] = T(pow(predicted[0] - observed_x,2) + pow(predicted[1] - observed_y,2));
                // residuals[0] = -pow(predicted[0] - T(observed_x),2);
                // residuals[1] = -pow(predicted[1] - T(observed_y),2);

                return true;
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

                const double predicted[3] = {
                    camParam[0]*pt[0][0] + camParam[1]*pt[0][1] + camParam[2]*pt[0][2] + camParam[3],
                    camParam[4]*pt[0][0] + camParam[5]*pt[0][1] + camParam[6]*pt[0][2] + camParam[7],
                    camParam[8]*pt[0][0] + camParam[9]*pt[0][1] + camParam[10]*pt[0][2] + camParam[11]};

                // residuals[0] = predicted[0] / predicted[2] - observed_x;
                // residuals[1] = predicted[1] / predicted[2] - observed_y;

                residuals[0] = std::sqrt(std::pow(predicted[0] / predicted[2] - observed_x,2)
                                         + std::pow(predicted[1] / predicted[2] - observed_y,2));

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
            // Sanity checks
            if (cameraMatrices.size() != pointsOnEachCamera.size())
                error("numberCameras.size() != pointsOnEachCamera.size() (" + std::to_string(cameraMatrices.size())
                      + " vs. " + std::to_string(pointsOnEachCamera.size()) + ").",
                      __LINE__, __FUNCTION__, __FILE__);
            if (cameraMatrices.empty())
                error("numberCameras.empty()",
                      __LINE__, __FUNCTION__, __FILE__);
            // Create and fill A for homogenous equation system Ax = 0
            const auto numberCameras = (int)cameraMatrices.size();
            cv::Mat A = cv::Mat::zeros(numberCameras*2, 4, CV_64F);
            for (auto i = 0 ; i < numberCameras ; i++)
            {
                A.rowRange(i*2, i*2+1) = pointsOnEachCamera[i].x*cameraMatrices[i].rowRange(2,3)
                                       - cameraMatrices[i].rowRange(0,1);
                A.rowRange(i*2+1, i*2+2) = pointsOnEachCamera[i].y*cameraMatrices[i].rowRange(2,3)
                                         - cameraMatrices[i].rowRange(1,2);
            }
            // Solve x for Ax = 0 --> SVD on A
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
                                       const std::vector<cv::Point2d>& pointsOnEachCamera,
                                       const double reprojectionMaxAcceptable)
    {
        try
        {
            // Information for 3 cameras:
            //     - Speed: triangulate ~0.01 ms vs. optimization ~0.2 ms
            //     - Accuracy: initial reprojection error ~14-21, reduced ~5% with non-linear optimization

            // Basic triangulation
            auto projectionError = triangulate(reconstructedPoint, cameraMatrices, pointsOnEachCamera);

            // Basic RANSAC (for >= 4 cameras if the reprojection error is higher than usual)
            // 1. Run with all cameras (already done)
            // 2. Run with all but 1 camera for each camera.
            // 3. Use the one with minimum average reprojection error.
            // Note: Meant to be used for up to 7-8 views. With more than that, it might not improve much.
            // Set initial values
            auto cameraMatricesFinal = cameraMatrices;
            auto pointsOnEachCameraFinal = pointsOnEachCamera;
            if (cameraMatrices.size() >= 4
                && projectionError > 0.5 * reprojectionMaxAcceptable
                /*&& projectionError < 1.5 * reprojectionMaxAcceptable*/)
            {
                auto bestReprojection = projectionError;
                auto bestReprojectionIndex = -1; // -1 means with all camera views
                for (auto i = 0u; i < cameraMatrices.size(); ++i)
                {
                    // Set initial values
                    auto cameraMatricesSubset = cameraMatrices;
                    auto pointsOnEachCameraSubset = pointsOnEachCamera;
                    // Remove camera i
                    cameraMatricesSubset.erase(cameraMatricesSubset.begin() + i);
                    pointsOnEachCameraSubset.erase(pointsOnEachCameraSubset.begin() + i);
                    // Remove camera i
                    const auto projectionErrorSubset = triangulate(reconstructedPoint, cameraMatricesSubset,
                                                                   pointsOnEachCameraSubset);
                    // If projection doesn't change much, it usually means all points are bad.
                    if (projectionErrorSubset > 0.9 * projectionError
                        && projectionErrorSubset < 1.1 * projectionError)
                    {
                        bestReprojectionIndex = -1;
                        break;
                    }
                    // Save maximum
                    if (bestReprojection > projectionErrorSubset)
                    {
                        bestReprojection = projectionErrorSubset;
                        bestReprojectionIndex = i;
                    }
                }

                if (bestReprojectionIndex != -1 && bestReprojection < 0.5 * reprojectionMaxAcceptable)
                {
                    // Remove camera i
                    cameraMatricesFinal.erase(cameraMatricesFinal.begin() + bestReprojectionIndex);
                    pointsOnEachCameraFinal.erase(pointsOnEachCameraFinal.begin() + bestReprojectionIndex);
                }
            }

            #ifdef USE_CERES
                // Empirically detected that reprojection error (for 4 cameras) only minimizes the error if initial
                // project error > ~2.5, and that it improves more the higher that error actually is
                // Therefore, we disable it for already accurate samples in order to get both:
                //     - Speed
                //     - Accuracy for already accurate samples
                if (projectionError > 3.0
                    && projectionError < 1.5*reprojectionMaxAcceptable)
                {
                    // Slow equivalent: double paramX[3]; paramX[i] = reconstructedPoint.at<double>(i);
                    double* paramX = (double*)reconstructedPoint.data;
                    ceres::Problem problem;
                    for (auto i = 0u; i < cameraMatricesFinal.size(); ++i)
                    {
                        // Slow copy equivalent:
                        //     double camParam[12]; memcpy(camParam, cameraMatricesFinal[i].data, sizeof(double)*12);
                        const double* const camParam = (double*)cameraMatricesFinal[i].data;
                        // Each Residual block takes a point and a camera as input and outputs a 2
                        // dimensional residual. Internally, the cost function stores the observed
                        // image location and compares the reprojection against the observation.
                        ceres::CostFunction* cost_function =
                            new ceres::AutoDiffCostFunction<ReprojectionErrorForTriangulation, 2, 3>(
                                new ReprojectionErrorForTriangulation(
                                    pointsOnEachCameraFinal[i].x, pointsOnEachCameraFinal[i].y, camParam));
                        // Add to problem
                        problem.AddResidualBlock(cost_function,
                            //NULL, //squared loss
                            new ceres::HuberLoss(2.0),
                            paramX); // paramX[0,1,2]
                    }

                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
                    // options.num_threads = 2; // It does not affect speed
                    // if (fastVersion)
                    {
                        // ~22 ms
                        // options.function_tolerance = 1e-3; //1e-6
                        // options.gradient_tolerance = 1e-5; //1e-10
                        // options.parameter_tolerance = 1e-5; //1e-8
                        // options.inner_iteration_tolerance = 1e-3; //1e-6
                        // ~30 ms (~30 FPS)
                        options.function_tolerance = 1e-4; //1e-6
                        options.gradient_tolerance = 1e-7; //1e-10
                        options.parameter_tolerance = 1e-6; //1e-8
                        options.inner_iteration_tolerance = 1e-4; //1e-6
                        // Default (none of the above) ~42 ms
                    }
                    // options.minimizer_progress_to_stdout = true;
                    // options.parameter_tolerance = 1e-20;
                    // options.function_tolerance = 1e-20;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    // if (summary.initial_cost > summary.final_cost)
                    //     std::cout << summary.FullReport() << "\n";

                    projectionError = calcReprojectionError(reconstructedPoint, cameraMatricesFinal,
                                                            pointsOnEachCameraFinal);
                    // const auto reprojectionErrorDecrease = std::sqrt((summary.initial_cost - summary.final_cost)
                    //                                      / double(cameraMatricesFinal.size()));
                }
            #else
                UNUSED(reprojectionMaxAcceptable);
            #endif
            // assert(reconstructedPoint.at<double>(3) == 1.);

            // // Check that our implementation gives similar result than OpenCV
            // // But we apply bundle adjustment + >2 views, so it should be better (and slower) than OpenCV one
            // if (cameraMatricesFinal.size() == 4)
            // {
            //     cv::Mat triangCoords4D;
            //     cv::triangulatePoints(cameraMatricesFinal.at(0), cameraMatricesFinal.at(3),
            //                           std::vector<cv::Point2d>{pointsOnEachCameraFinal.at(0)},
            //                           std::vector<cv::Point2d>{pointsOnEachCameraFinal.at(3)}, triangCoords4D);
            //     triangCoords4D /= triangCoords4D.at<double>(3);
            //     std::cout << reconstructedPoint << "\n"
            //               << triangCoords4D << "\n"
            //               << cv::norm(reconstructedPoint-triangCoords4D) << "\n" << std::endl;
            // }

            return projectionError;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1.;
        }
    }

    inline bool isValidKeypoint(const float* const keypointPtr, const Point<int>& imageSize)
    {
        try
        {
            const auto threshold = 0.35f;
            return (keypointPtr[2] > threshold
                    // If keypoint in border --> most probably it is actually out of the image,
                    // so removed to reduce that noise
                    && keypointPtr[0] > 8
                    && keypointPtr[0] < imageSize.x - 8
                    && keypointPtr[1] > 8
                    && keypointPtr[1] < imageSize.y - 8);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void reconstructArrayThread(Array<float>* keypoints3DPtr,
                                const std::vector<Array<float>>& keypointsVector,
                                const std::vector<cv::Mat>& cameraMatrices,
                                const std::vector<Point<int>>& imageSizes,
                                const int minViews3d)
    {
        try
        {
            auto& keypoints3D = *keypoints3DPtr;

            // Sanity check
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
                const auto channel0Length = keypointsVector.at(0).getSize(2);
                // Create x-y vector from high score results
                std::vector<int> indexesUsed;
                std::vector<std::vector<cv::Point2d>> xyPoints;
                std::vector<std::vector<cv::Mat>> cameraMatricesPerPoint;
                for (auto part = 0; part < numberBodyParts; part++)
                {
                    // Create vector of points
                    // auto missedPoint = false;
                    std::vector<cv::Point2d> xyPointsElement;
                    std::vector<cv::Mat> cameraMatricesElement;
                    const auto baseIndex = part * channel0Length;
                    // for (auto& keypoints : keypointsVector)
                    for (auto i = 0u ; i < keypointsVector.size() ; i++)
                    {
                        const auto& keypoints = keypointsVector[i];
                        if (isValidKeypoint(&keypoints[baseIndex], imageSizes[i]))
                        {
                            xyPointsElement.emplace_back(cv::Point2d{keypoints[baseIndex],
                                                                     keypoints[baseIndex+1]});
                            cameraMatricesElement.emplace_back(cameraMatrices[i]);
                        }
                    }
                    // If visible from all views (minViews3d < 0)
                    // or if visible for at least minViews3d views
                    if ((minViews3d < 0 && cameraMatricesElement.size() == cameraMatrices.size())
                        || (minViews3d > 1 && minViews3d <= (int)xyPointsElement.size()))
                    {
                        indexesUsed.emplace_back(part);
                        xyPoints.emplace_back(xyPointsElement);
                        cameraMatricesPerPoint.emplace_back(cameraMatricesElement);
                    }
                }
                // 3D reconstruction
                const auto imageRatio = std::sqrt(imageSizes[0].x * imageSizes[0].y / 1310720);
                const auto reprojectionMaxAcceptable = 25 * imageRatio;
                std::vector<double> reprojectionErrors(xyPoints.size());
                keypoints3D.reset({ 1, numberBodyParts, 4 }, 0);
                if (!xyPoints.empty())
                {
                    // Do 3D reconstruction
                    std::vector<cv::Point3f> xyzPoints(xyPoints.size());
                    for (auto i = 0u; i < xyPoints.size(); i++)
                    {
                        cv::Mat reconstructedPoint;
                        reprojectionErrors[i] = triangulateWithOptimization(reconstructedPoint,
                                                                            cameraMatricesPerPoint[i],
                                                                            xyPoints[i],
                                                                            reprojectionMaxAcceptable);
                        xyzPoints[i] = cv::Point3d{
                            reconstructedPoint.at<double>(0),
                            reconstructedPoint.at<double>(1),
                            reconstructedPoint.at<double>(2)};
                    }
                    const auto reprojectionErrorTotal = std::accumulate(
                        reprojectionErrors.begin(), reprojectionErrors.end(), 0.0) / xyPoints.size();

                    // 3D points to pose
                    // OpenCV alternative:
                    // http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#triangulatepoints
                    // cv::Mat reconstructedPoints{4, firstcv::Points.size(), CV_64F};
                    // cv::triangulatePoints(cv::Mat::eye(3,4, CV_64F), M_3_1, firstcv::Points, secondcv::Points,
                    //                           reconstructedcv::Points);
                    // 20 pixels for 1280x1024 image
                    bool atLeastOnePointProjected = false;
                    const auto lastChannelLength = keypoints3D.getSize(2);
                    for (auto index = 0u; index < indexesUsed.size(); index++)
                    {
                        if (std::isfinite(xyzPoints[index].x) && std::isfinite(xyzPoints[index].y)
                            && std::isfinite(xyzPoints[index].z)
                            // Remove outliers
                            && (reprojectionErrors[index] < 5 * reprojectionErrorTotal
                                && reprojectionErrors[index] < reprojectionMaxAcceptable))
                        {
                            const auto baseIndex = indexesUsed[index] * lastChannelLength;
                            keypoints3D[baseIndex] = xyzPoints[index].x;
                            keypoints3D[baseIndex + 1] = xyzPoints[index].y;
                            keypoints3D[baseIndex + 2] = xyzPoints[index].z;
                            keypoints3D[baseIndex + 3] = 1.f;
                            atLeastOnePointProjected = true;
                        }
                    }
                    if (!atLeastOnePointProjected || reprojectionErrorTotal > 60)
                        log("Unusual high re-projection error (averaged over #keypoints) of value "
                            + std::to_string(reprojectionErrorTotal) + " pixels, while the average for a good OpenPose"
                            " detection from 4 cameras is about 2-3 pixels. It might be simply a wrong OpenPose"
                            " detection. If this message appears very frequently, your calibration parameters"
                            " might be wrong.", Priority::High);
                }
                // log("Reprojection error: " + std::to_string(reprojectionErrorTotal)); // To debug reprojection error
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    PoseTriangulation::PoseTriangulation(const int minViews3d) :
        mMinViews3d{minViews3d}
    {
        try
        {
            // Sanity check
            if (0 <= mMinViews3d && mMinViews3d < 2)
                error("Minimum number of views must be at least 2 (e.g., `--3d_min_views 2`) or negative.",
                      __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    PoseTriangulation::~PoseTriangulation()
    {
    }

    void PoseTriangulation::initializationOnThread()
    {
    }

    Array<float> PoseTriangulation::reconstructArray(const std::vector<Array<float>>& keypointsVector,
                                                     const std::vector<cv::Mat>& cameraMatrices,
                                                     const std::vector<Point<int>>& imageSizes) const
    {
        try
        {
            return reconstructArray(std::vector<std::vector<Array<float>>>{keypointsVector},
                                    cameraMatrices, imageSizes).at(0);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }

    std::vector<Array<float>> PoseTriangulation::reconstructArray(
        const std::vector<std::vector<Array<float>>>& keypointsVectors,
        const std::vector<cv::Mat>& cameraMatrices,
        const std::vector<Point<int>>& imageSizes) const
    {
        try
        {
            std::vector<Array<float>> keypoints3Ds(keypointsVectors.size());
            // std::vector<std::thread> threads(keypointsVectors.size()-1);
            for (auto i = 0u; i < keypointsVectors.size()-1; i++)
            {
                // // Multi-thread option - ~15% slower
                // // Ceres seems to be super slow if run concurrently in different threads
                // threads.at(i) = std::thread{&reconstructArrayThread,
                //                             &keypoints3Ds[i], keypointsVectors[i], cameraMatrices,
                //                             imageSizes, mMinViews3d};
                // Single-thread option
                reconstructArrayThread(&keypoints3Ds[i], keypointsVectors[i], cameraMatrices,
                                       imageSizes, mMinViews3d);
            }
            reconstructArrayThread(&keypoints3Ds.back(), keypointsVectors.back(), cameraMatrices,
                                   imageSizes, mMinViews3d);
            // // Close threads
            // for (auto& thread : threads)
            //     if (thread.joinable())
            //         thread.join();
            // Return results
            return keypoints3Ds;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }
}
