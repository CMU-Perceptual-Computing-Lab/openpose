#include <openpose/calibration/cameraParameterEstimation.hpp>
#include <fstream>
#include <numeric> // std::accumulate
#ifdef USE_CERES
    #include <ceres/ceres.h>
    #include <ceres/rotation.h>
#endif
#ifdef USE_EIGEN
    #include <Eigen/Dense>
    #include <opencv2/core/eigen.hpp>
#endif
#include <openpose/3d/cameraParameterReader.hpp>
#include <openpose/filestream/fileStream.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose_private/3d/poseTriangulationPrivate.hpp>
#include <openpose_private/calibration/gridPatternFunctions.hpp>
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp>

namespace op
{
    // Private variables
    const long double PI = 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844;

    // Private functions
    struct Intrinsics
    {
        cv::Mat cameraMatrix;
        cv::Mat distortionCoefficients;

        Intrinsics() :
            cameraMatrix(cv::Mat::eye(3, 3, CV_64F)),
            distortionCoefficients(cv::Mat::zeros(14, 1, CV_64F))
        {
        }

        Intrinsics(const cv::Mat& cameraMatrix, const cv::Mat& distortionCoefficients) :
            cameraMatrix{cameraMatrix.clone()}, // cv::Mat::eye(3, 3, CV_64F)
            distortionCoefficients{distortionCoefficients.clone()} // cv::Mat::zeros(14||12||8||5, 1, CV_64F)
        {
        }

        bool empty()
        {
            return cameraMatrix.empty() || distortionCoefficients.empty();
        }
    };

    #ifdef USE_EIGEN
        struct Extrinsics
        {
            Eigen::Matrix3d Rotation;
            Eigen::Vector3d translationMm;
            std::vector<cv::Point2f> points2DVector;
            std::vector<cv::Point3f> objects3DVector;

            Extrinsics()
            {
                Rotation.setIdentity();
                translationMm.setZero();
            }
        };
    #endif

    std::vector<std::string> getImagePaths(const std::string& imageDirectoryPath)
    {
        try
        {
            // Get files on directory with the desired extensions
            const auto imagePaths = getFilesOnDirectory(imageDirectoryPath, Extensions::Images);
            // Check #files > 0
            if (imagePaths.empty())
                error("No images were found on `" + imageDirectoryPath + "`.", __LINE__, __FUNCTION__, __FILE__);
            // Return result
            return imagePaths;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<std::pair<cv::Mat, std::string>> getImageAndPaths(const std::string& imageDirectoryPath)
    {
        try
        {
            // Get images on directory
            const auto imagePaths = getImagePaths(imageDirectoryPath);
            // Check #files > 0
            if (imagePaths.empty())
                error("No images were found on `" + imageDirectoryPath + "`.", __LINE__, __FUNCTION__, __FILE__);
            // Read images
            std::vector<std::pair<cv::Mat, std::string>> imageAndPaths;
            for (const auto& imagePath : imagePaths)
            {
                imageAndPaths.emplace_back(std::make_pair(cv::imread(imagePath, CV_LOAD_IMAGE_COLOR), imagePath));
                if (imageAndPaths.back().first.empty())
                    error("Image could not be opened from path `" + imagePath + "`.",
                          __LINE__, __FUNCTION__, __FILE__);
            }
            // Return result
            return imageAndPaths;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::pair<double, std::vector<double>> calcReprojectionErrors(
        const std::vector<std::vector<cv::Point3f>>& objects3DVectors,
        const std::vector<std::vector<cv::Point2f>>& points2DVectors, const std::vector<cv::Mat>& rVecs,
        const std::vector<cv::Mat>& tVecs, const Intrinsics& intrinsics)
    {
        try
        {
            std::vector<double> perViewErrors(objects3DVectors.size());

            std::vector<cv::Point2f> points2DVectors2;
            unsigned long long totalPoints = 0;
            double totalErr = 0.;

            for (auto i = 0ull; i < objects3DVectors.size(); ++i )
            {
                cv::projectPoints(
                    cv::Mat(objects3DVectors.at(i)), rVecs.at(i), tVecs.at(i), intrinsics.cameraMatrix,
                    intrinsics.distortionCoefficients, points2DVectors2);
                const auto err = cv::norm(cv::Mat(points2DVectors.at(i)), cv::Mat(points2DVectors2), CV_L2);

                const auto n = objects3DVectors.at(i).size();
                perViewErrors.at(i) = {std::sqrt(err*err/n)};
                totalErr        += err*err;
                totalPoints     += n;
            }
            // Return results
            const auto reprojectionError = std::sqrt(totalErr/totalPoints);
            return std::make_pair(reprojectionError, perViewErrors);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_pair(0., std::vector<double>{});
        }
    }

    Intrinsics calcIntrinsicParameters(
        const cv::Size& imageSize, const std::vector<std::vector<cv::Point2f>>& points2DVectors,
        const std::vector<std::vector<cv::Point3f>>& objects3DVectors, const int calibrateCameraFlags)
    {
        try
        {
            opLog("\nCalibrating camera (intrinsics) with points from " + std::to_string(points2DVectors.size())
                + " images...", Priority::High);

            //Find intrinsic and extrinsic camera parameters
            Intrinsics intrinsics;
            std::vector<cv::Mat> rVecs;
            std::vector<cv::Mat> tVecs;
            const auto rms = cv::calibrateCamera(
                objects3DVectors, points2DVectors, imageSize, intrinsics.cameraMatrix,
                intrinsics.distortionCoefficients, rVecs, tVecs, calibrateCameraFlags);

            // cv::checkRange checks that every array element is neither NaN nor infinite
            const auto calibrationIsCorrect = cv::checkRange(intrinsics.cameraMatrix)
                                            && cv::checkRange(intrinsics.distortionCoefficients);
            if (!calibrationIsCorrect)
                error("Unvalid cameraMatrix and/or distortionCoefficients.", __LINE__, __FUNCTION__, __FILE__);

            double totalAvgErr;
            std::vector<double> reprojectionErrors;
            std::tie(totalAvgErr, reprojectionErrors) = calcReprojectionErrors(
                objects3DVectors, points2DVectors, rVecs, tVecs, intrinsics);

            opLog("\nIntrinsics:", Priority::High);
            opLog("Re-projection error - cv::calibrateCamera vs. calcReprojectionErrors:\t" + std::to_string(rms)
                + " vs. " + std::to_string(totalAvgErr), Priority::High);
            opLog("Intrinsics_K:", Priority::High);
            opLog(intrinsics.cameraMatrix, Priority::High);
            opLog("Intrinsics_distCoeff:", Priority::High);
            opLog(intrinsics.distortionCoefficients, Priority::High);
            opLog(" ", Priority::High);

            return intrinsics;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Intrinsics{};
        }
    }

    double setAngleInRangeZeroTwoPi(const double angle)
    {
        try
        {
            // Return angle in range [0,2pi)
            return std::fmod(angle, 360);    // floating-point remainder
            // double result{angle};

            // Return angle in range [0,2pi)
            // const auto twoPi = 2 * PI;
            // while (result >= 2*PI)
            //     result -= twoPi;
            // while (result < 0)
            //     result += twoPi;

            // return result;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0;
        }
    }

    double setAngleInRangePlusMinusPi(const double angle)
    {
        try
        {
            auto result = setAngleInRangeZeroTwoPi(angle);

            // Return angle in range (-pi,pi]
            const auto twoPi = 2 * PI;
            if (result > PI)
                result -= twoPi;

            // // Return angle in range (-pi,pi]
            // const auto twoPi = 2 * PI;
            // while (result > PI)
            //     result -= twoPi;
            // while (result <= -PI)
            //     result += twoPi;

            return result;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0;
        }
    }

    #ifdef USE_EIGEN
        cv::Mat getRodriguesVector(const Eigen::Matrix3d& rotationMatrix)
        {
            try
            {
                // Rotation matrix as cv::Mat
                cv::Mat rotationMatrixCv;
                cv::eigen2cv(rotationMatrix, rotationMatrixCv);
                // Rotation as vector
                cv::Mat rotationVector;
                cv::Rodrigues(rotationMatrixCv, rotationVector);
                return rotationVector;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return cv::Mat();
            }
        }

        std::pair<bool, double> estimateAverageAngle(const std::vector<double>& angles)
        {
            try
            {
                // Idea:
                    // Average(-40, 40) = 0
                    // Average(-179, 179) = 180 (not 0!)
                    // Average(90, 270) = 0 || 180??? We will assume outliers and return false

                // Sanity check
                if (angles.empty())
                    error("Variables `angles` is empty when calling estimateAverageAngle().",
                        __LINE__, __FUNCTION__, __FILE__);

                // angles in range (-pi, pi]
                auto anglesNormalized = angles;
                for (auto& angle : anglesNormalized)
                    angle = setAngleInRangePlusMinusPi(angle);

                // If the difference between them is > 180 degrees, then we turn them in the range [0, 360) (so
                // now the difference between them is < 180) and return the traditional average. Examples:
                //     - If one in range [90, 180] and the other in range (-180, -90]
                //     - If one is 179 degrees and the other one -179 degrees -> average should be 180 no 0! So:
                //       [179 + (360-179)] / 2 = 180
                //     - Etc.
                // Math equivalent:
                //     We want:         ( maxAngle + (minAngle + 360) )   /2
                //     Equivalent to:   ( maxAngle + minAngle + 360 )   /2     =   (maxAngle + minAngle) / 2 + 180
                //                      =   (angleA + angleB) / 2 + 180,   in radians: (angleA + angleB) / 2 + PI
                auto minElement = *std::min_element(anglesNormalized.begin(), anglesNormalized.end());
                auto maxElement = *std::max_element(anglesNormalized.begin(), anglesNormalized.end());
                if (maxElement - minElement >= PI)
                    for (auto& angle : anglesNormalized)
                        angle = setAngleInRangeZeroTwoPi(angle);

                // If after normalizing range between min and max is still >180 degrees --> there are outliers
                minElement = *std::min_element(anglesNormalized.begin(), anglesNormalized.end());
                maxElement = *std::max_element(anglesNormalized.begin(), anglesNormalized.end());
                auto resultIsOK = true;
                if (maxElement - minElement >= PI)
                {
                    resultIsOK = {false};
                    opLog("There are outliers in the angles.", Priority::High);
                }

                // If the difference between them is <= 180 degrees, then we just return the traditional average.
                // Examples:
                //     - If both have the same signs, i.e., both in range [0, 180] or both in range (-180, 0)
                //     - If one in range [0, 90] and the other in range [-90, 0]
                //     - Etc.
                auto average = std::accumulate(anglesNormalized.begin(), anglesNormalized.end(), 0.)
                             / (double)anglesNormalized.size();
                average = setAngleInRangePlusMinusPi(average);

                return std::make_pair(resultIsOK, average);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return std::make_pair(false, 0.);
            }
        }

        Eigen::Matrix4d getMAverage(const std::vector<Eigen::Matrix4d>& MsToAverage,
                                    const Eigen::Matrix4d& noisyApproximatedM = Eigen::Matrix4d::Zero())
        {
            try
            {
                auto MsToAverageRobust = MsToAverage;
                // Clean noisy outputs
                if (noisyApproximatedM.norm() > 0.1)
                {
                    MsToAverageRobust.clear();
                    for (const auto& matrix : MsToAverage)
                    {
                        bool addElement = true;
                        for (auto col = 0 ; col < 3 ; col++)
                        {
                            for (auto row = 0 ; row < 3 ; row++)
                            {
                                if (std::abs(matrix(col, row) - noisyApproximatedM(col, row)) > 0.25)
                                {
                                    addElement = false;
                                    break;
                                }
                            }
                        }
                        if (addElement)
                            MsToAverageRobust.push_back(matrix);
                    }
                }

                // Projection matrix
                Eigen::Matrix4d averagedProjectionMatrix;
                averagedProjectionMatrix.setIdentity();

                // Average translation
                for (const auto& matrix : MsToAverageRobust)
                    averagedProjectionMatrix.block<3,1>(0,3) += matrix.block<3,1>(0,3);
                averagedProjectionMatrix.block<3,1>(0,3) /= (double)MsToAverageRobust.size();

                // Average rotation
                std::array<std::vector<double>, 3> rotationVectors;
                for (const auto& matrix : MsToAverageRobust)
                {
                    const auto rVec = getRodriguesVector(matrix.block<3,3>(0,0));
                    for (auto i = 0u ; i < rotationVectors.size() ; i++)
                        rotationVectors.at(i).emplace_back(rVec.at<double>(i,0));
                }
                cv::Mat rotationVector = cv::Mat::zeros(3, 1, CV_64F);
                for (auto i = 0u ; i < rotationVectors.size() ; i++)
                {
                    const auto pairAverageAngle = estimateAverageAngle(rotationVectors.at(i));
                    if (!pairAverageAngle.first)
                        opLog("Outlies in the result. Something went wrong when estimating the average of different"
                              " projection matrices.", Priority::High);
                    rotationVector.at<double>(i,0) = {pairAverageAngle.second};
                }
                cv::Mat rotationMatrix;
                cv::Rodrigues(rotationVector, rotationMatrix);
                Eigen::Matrix3d rotEigen;
                cv::cv2eigen(rotationMatrix, rotEigen);
                averagedProjectionMatrix.block<3,3>(0,0) = rotEigen;

                // Result
                return averagedProjectionMatrix;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return Eigen::Matrix4d{};
            }
        }

        Eigen::Matrix4d getMFromRt(const Eigen::Matrix3d& Rot, const Eigen::Vector3d& trans)
        {
            try
            {
                // projectionMatrix
                Eigen::Matrix4d projectionMatrix = Eigen::Matrix4d::Identity();
                projectionMatrix.block<3,3>(0,0) = Rot;
                projectionMatrix.block<3,1>(0,3) = trans;
                return projectionMatrix;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return Eigen::Matrix4d{};
            }
        }

        Eigen::Matrix4d getMFromRt(const cv::Mat& Rot, const cv::Mat& trans)
        {
            try
            {
                if (Rot.cols != 3 || Rot.rows != 3)
                    error("Rotation matrix does not have 3 cols and/or 3 rows.", __LINE__, __FUNCTION__, __FILE__);
                if (trans.cols != 1 || trans.rows != 3)
                    error("Translation vector does not have 1 col and/or 3 rows.", __LINE__, __FUNCTION__, __FILE__);

                Eigen::Matrix3d RotEigen;
                cv::cv2eigen(Rot, RotEigen);
                Eigen::Vector3d transEigen;
                cv::cv2eigen(trans, transEigen);

                // projectionMatrix
                return getMFromRt(RotEigen, transEigen);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return Eigen::Matrix4d{};
            }
        }
        std::pair<cv::Mat, cv::Mat> solveCorrespondences2D3D(
            const cv::Mat& cameraMatrix, const cv::Mat& distortionCoefficients,
            const std::vector<cv::Point3f>& objects3DVector, const std::vector<cv::Point2f>& points2DVector)
        {
            try
            {
                // opLog("Solving 2D-3D correspondences (extrinsics)", Priority::High);
                cv::Mat rVec(3, 1, cv::DataType<double>::type);
                cv::Mat tVec(3, 1, cv::DataType<double>::type);

                // VERY IMPORTANT
                // So far, the only one that gives precise results is: cv::SOLVEPNP_DLS without RANSCAC

                // Non-RANSAC mode
                // It only can use 4 points
                // cv::solvePnP(fourObjects3DVector, fourPoints2DVector, cameraMatrix, distortionCoefficients, rVec, tVec, false, cv::SOLVEPNP_P3P);
                // More robust against outliers
                // const auto found = cv::solvePnP(objects3DVector, points2DVector, cameraMatrix, distortionCoefficients, rVec, tVec, false, cv::SOLVEPNP_ITERATIVE);
                // Default cv::SOLVEPNP_ITERATIVE for OpenCV 3
                const auto found = cv::solvePnP(objects3DVector, points2DVector, cameraMatrix, distortionCoefficients, rVec, tVec, false);
                // More fragile against outliers
                // cv::solvePnP(objects3DVector, points2DVector, cameraMatrix, distortionCoefficients, rVec, tVec, false, cv::SOLVEPNP_EPNP);
                // More robust against outliers
                // cv::solvePnP(objects3DVector, points2DVector, cameraMatrix, distortionCoefficients, rVec, tVec, false, cv::SOLVEPNP_DLS);
                // More robust against outliers
                // cv::solvePnP(objects3DVector, points2DVector, cameraMatrix, distortionCoefficients, rVec, tVec, false, cv::SOLVEPNP_UPNP);

                // RANSAC mode
                // It gives really bad results
                // This one is the best, but it doest care about initial guesses
                // cv::solvePnPRansac(objects3DVector, points2DVector, cameraMatrix, distortionCoefficients, rVec, tVec, false, 1000, 8.0, 0.99, (int)objects3DVector.size() + 1, cv::noArray(), CV_EPNP);
                // This one is the best, but it doest care about initial guesses
                // cv::solvePnPRansac(objects3DVector, points2DVector, cameraMatrix, distortionCoefficients, rVec, tVec, false, 1000, 8.0, 0.99, cv::noArray(), cv::SOLVEPNP_ITERATIVE);
                if (!found)
                    error("Correspondences could not be found.", __LINE__, __FUNCTION__, __FILE__);

                return std::make_pair(rVec, tVec);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return std::make_pair(cv::Mat(), cv::Mat());
            }
        }

        std::tuple<cv::Mat, cv::Mat, std::vector<cv::Point2f>, std::vector<cv::Point3f>> calcExtrinsicParametersOpenCV(
            const cv::Mat& image, const cv::Mat& cameraMatrix, const cv::Mat& distortionCoefficients,
            const cv::Size& gridInnerCorners, const float gridSquareSizeMm)
        {
            try
            {
                // Finding accurate chessboard corner positions
                bool found{false};
                std::vector<cv::Point2f> points2DVector;
                std::tie(found, points2DVector) = findAccurateGridCorners(image, gridInnerCorners);
                if (!found)
                    return std::make_tuple(cv::Mat(), cv::Mat(), std::vector<cv::Point2f>(),
                                           std::vector<cv::Point3f>());

                // Reordering points2DVector to have the first point at the top left position (top right in
                // case the chessboard is mirrored)
                reorderPoints(points2DVector, gridInnerCorners, image);

                // Generate objects3DVector from gridSquareSizeMm
                const auto objects3DVector = getObjects3DVector(gridInnerCorners, gridSquareSizeMm);

                // Solving correspondences 2D-3D
                cv::Mat rVec;
                cv::Mat tVec;
                std::tie(rVec, tVec) = solveCorrespondences2D3D(cameraMatrix, distortionCoefficients, objects3DVector,
                                                                points2DVector);

                return std::make_tuple(rVec, tVec, points2DVector, objects3DVector);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return std::make_tuple(cv::Mat(), cv::Mat(), std::vector<cv::Point2f>(), std::vector<cv::Point3f>());
            }
        }

        std::tuple<bool, Extrinsics> calcExtrinsicParameters(const cv::Mat& image,
                                                             const cv::Size& gridInnerCorners,
                                                             const float gridSquareSizeMm,
                                                             const cv::Mat& cameraMatrix,
                                                             const cv::Mat& distortionCoefficients)
        {
            try
            {
                Extrinsics extrinsics;
                cv::Mat rVecOpenCV;
                cv::Mat RotOpenCV;
                cv::Mat tMmOpenCV;
                std::tie(rVecOpenCV, tMmOpenCV, extrinsics.points2DVector, extrinsics.objects3DVector)
                        = calcExtrinsicParametersOpenCV(
                            image, cameraMatrix, distortionCoefficients, gridInnerCorners, gridSquareSizeMm);
                if (rVecOpenCV.empty())
                    return std::make_tuple(false, Extrinsics{});

                cv::Rodrigues(rVecOpenCV, RotOpenCV);
                cv::cv2eigen(RotOpenCV, extrinsics.Rotation);
                cv::cv2eigen(tMmOpenCV, extrinsics.translationMm);

                return std::make_tuple(true, extrinsics);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return std::make_tuple(false, Extrinsics{});
            }
        }

        std::tuple<bool, Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d> getExtrinsicParameters(
            const std::vector<std::string>& cameraPaths, const cv::Size& gridInnerCorners, const float gridSquareSizeMm,
            const bool coutAndPlotGridCorners, const std::vector<cv::Mat>& intrinsics, const std::vector<cv::Mat>& distortions)
        {
            try
            {
                // Sanity check
                if (intrinsics.size() != 2 || distortions.size() != 2 || cameraPaths.size() != 2)
                    error("Found that (intrinsics.size() != 2 || distortions.size() != 2 || cameraPaths.size() != 2).",
                          __LINE__, __FUNCTION__, __FILE__);

                std::vector<Extrinsics> extrinsicss(2);
                for (auto i = 0u ; i < cameraPaths.size() ; i++)
                {
                    if (coutAndPlotGridCorners)
                        opLog("getExtrinsicParameters(...), iteration with: " + cameraPaths[i], Priority::High);
                    // Loading images
                    const cv::Mat image = cv::imread(cameraPaths[i]);
                    if (image.empty())
                        error("No image found in the path `" + cameraPaths[i] + "`.",
                              __LINE__, __FUNCTION__, __FILE__);
                    bool valid;
                    std::tie(valid, extrinsicss[i]) = calcExtrinsicParameters(
                        image, gridInnerCorners, gridSquareSizeMm, intrinsics.at(i), distortions.at(i));
                    if (!valid)
                        return std::make_tuple(false, Eigen::Matrix3d{}, Eigen::Vector3d{}, Eigen::Matrix3d{},
                                               Eigen::Vector3d{});
                    if (coutAndPlotGridCorners)
                        plotGridCorners(gridInnerCorners, extrinsicss[i].points2DVector, cameraPaths[i], image);
                }

                return std::make_tuple(
                    true,
                    extrinsicss.at(0).Rotation,
                    extrinsicss.at(0).translationMm,
                    extrinsicss.at(1).Rotation,
                    extrinsicss.at(1).translationMm
                );
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return std::make_tuple(
                    false, Eigen::Matrix3d{}, Eigen::Vector3d{}, Eigen::Matrix3d{}, Eigen::Vector3d{});
            }
        }

        Eigen::Matrix4d getMFrom2Ms(const Eigen::Matrix4d& MGridToCam0,
                                    const Eigen::Matrix4d& MGridToCam1)
        {
            try
            {
                // MCam1ToGrid
                // Non-efficient equivalent:
                    // MCam1ToGrid = MGridToCam1.inverse()
                // Efficient version:
                    // M * M.inverse()  =    I
                    // [R t] [R -R't]   =  [I 0]
                    // [0 1] [0   1 ]   =  [0 1]
                    // Conclusion:
                    //     R_inv = R^-1 = R^T
                    //     t_inv = -R^T t
                Eigen::Matrix4d MCam1ToGrid = Eigen::Matrix4d::Identity();
                MCam1ToGrid.block<3,3>(0,0) = MGridToCam1.block<3,3>(0,0).transpose();
                MCam1ToGrid.block<3,1>(0,3) = - MCam1ToGrid.block<3,3>(0,0) * MGridToCam1.block<3,1>(0,3);
                // MCam1ToCam0 = MGridToCam0 * inv(MGridToCam1)
                // I.e., position of camera 1 w.r.t. camera 0
                return Eigen::Matrix4d{MGridToCam0 * MCam1ToGrid};
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return Eigen::Matrix4d{};
            }
        }

        Eigen::Matrix4d getMFromCam1ToCam0(const Eigen::Matrix3d& RGridToMainCam0,
                                           const Eigen::Vector3d& tGridToMainCam0,
                                           const Eigen::Matrix3d& RGridToMainCam1,
                                           const Eigen::Vector3d& tGridToMainCam1,
                                           const bool coutAndImshowVerbose)
        {
            const auto MGridToCam0 = getMFromRt(RGridToMainCam0, tGridToMainCam0);
            const auto MGridToCam1 = getMFromRt(RGridToMainCam1, tGridToMainCam1);
            // const auto MCam1ToCam0 = getMFrom2Ms(MGridToCam0, MGridToCam1);
            const auto MCam1ToCam0 = getMFrom2Ms(MGridToCam1, MGridToCam0);

            if (coutAndImshowVerbose)
            {
                const Eigen::Vector3d tCam1WrtCam0 = MCam1ToCam0.block<3,1>(0,3) / MCam1ToCam0(3,3);
                const Eigen::Matrix3d RCam1WrtCam0 = MCam1ToCam0.block<3,3>(0,0);
                opLog("M_gb:", Priority::High);
                opLog(MGridToCam1, Priority::High);
                opLog("M_gf:", Priority::High);
                opLog(MGridToCam0, Priority::High);
                opLog("M_bf:", Priority::High);
                opLog(MCam1ToCam0, Priority::High);

                opLog("########## Secondary camera position w.r.t. main camera ##########", Priority::High);
                opLog("tCam1WrtCam0:", Priority::High);
                opLog(tCam1WrtCam0, Priority::High);
                opLog("RCam1WrtCam0:", Priority::High);
                opLog(RCam1WrtCam0, Priority::High);
                opLog("MCam0WrtCam1:", Priority::High);
                opLog((- RCam1WrtCam0.transpose() * tCam1WrtCam0), Priority::High);
            }

            return MCam1ToCam0;
        }
    #endif

    const int SIFT_NAME = ('S' + ('I' << 8) + ('F' << 16) + ('T' << 24));
    // const int MSER_NAME = ('M' + ('S' << 8) + ('E' << 16) + ('R' << 24));
    // const int RECT_NAME = ('R' + ('E' << 8) + ('C' << 16) + ('T' << 24));
    const int SIFT_VERSION_4 = ('V' + ('4' << 8) + ('.' << 16) + ('0' << 24));
    const int SIFT_EOF = (0xff + ('E' << 8) + ('O' << 16) + ('F' << 24));
    void writeVisualSFMSiftGPU(const std::string& fileName, const std::vector<cv::Point2f>& points2DVector)
    {
        const int siftName = SIFT_NAME;
        const int siftVersion = SIFT_VERSION_4;
        const int keyDimension = 5;
        const int descDimension = 128;
        const int siftEOF = SIFT_EOF;
        const int nSift = (int)points2DVector.size();
        std::ofstream ofstreamSift;
        ofstreamSift.open(fileName, std::ios::binary);
        // Can write
        if (ofstreamSift.is_open())
        {
            ofstreamSift.write(reinterpret_cast<const char*>(&siftName), sizeof(int));
            ofstreamSift.write(reinterpret_cast<const char*>(&siftVersion), sizeof(int));
            ofstreamSift.write(reinterpret_cast<const char*>(&nSift), sizeof(int));
            ofstreamSift.write(reinterpret_cast<const char*>(&keyDimension), sizeof(int));
            ofstreamSift.write(reinterpret_cast<const char*>(&descDimension), sizeof(int));
            // for (int j = 0; j < nSift; ++j)
            for (auto i = 0u; i < points2DVector.size(); i++)
            {
                // const float x = KeyPts[4 * j];
                // const float y = KeyPts[4 * j + 1];
                const float x = points2DVector[i].x;
                const float y = points2DVector[i].y;
                const float dummy = 0.f;
                const float scale = 1.f;
                const float orientation = 0.f;
                ofstreamSift.write(reinterpret_cast<const char*>(&x), sizeof(float));
                ofstreamSift.write(reinterpret_cast<const char*>(&y), sizeof(float));
                ofstreamSift.write(reinterpret_cast<const char*>(&dummy), sizeof(float));
                // ofstreamSift.write(reinterpret_cast<const char*>(&KeyPts[4 * j + 2]), sizeof(float));
                // ofstreamSift.write(reinterpret_cast<const char*>(&KeyPts[4 * j + 3]), sizeof(float));
                ofstreamSift.write(reinterpret_cast<const char*>(&scale), sizeof(float));
                ofstreamSift.write(reinterpret_cast<const char*>(&orientation), sizeof(float));
            }

            // // Extra argument: const unsigned char* const desc
            // for (int j = 0; j < nSift; ++j)
            //     for (int i = 0; i < descDimension; i++)
            //         ofstreamSift.write(reinterpret_cast<const char*>(&desc[j * descDimension + i]),
            //                            sizeof(unsigned char));
            const unsigned char dummy = 0u;
            for (auto i = 0; i < nSift * descDimension; i++)
                ofstreamSift.write(reinterpret_cast<const char*>(&dummy), sizeof(unsigned char));

            ofstreamSift.write(reinterpret_cast<const char*>(&siftEOF), sizeof(int));
            ofstreamSift.close();
        }
        // Couldn't write
        else
            opLog("Cannot write on " + fileName, Priority::High);
    }

    std::string getFileNameFromCameraIndex(const int cameraIndex)
    {
        try
        {
            // Sanity check
            if (cameraIndex >= 100)
                error("Only implemented for up to 99 cameras.", __LINE__, __FUNCTION__, __FILE__);
            // Return result
            return (cameraIndex < 10 ? "00_0" : "00_") + std::to_string(cameraIndex);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    void estimateAndSaveSiftFileSubThread(
        std::vector<cv::Point2f>* points2DExtrinsicPtr, std::vector<unsigned int>* matchIndexesCameraPtr,
        const int cameraIndex, const int numberCameras, const int numberCorners, const unsigned int numberViews,
        const bool saveImagesWithCorners, const std::string& imageFolder, const cv::Size& gridInnerCornersCvSize,
        const cv::Size& imageSize, const std::vector<std::pair<cv::Mat, std::string>>& imageAndPaths,
        const bool saveSIFTFile)
    {
        try
        {
            // Sanity checks
            if (points2DExtrinsicPtr == nullptr || matchIndexesCameraPtr == nullptr)
                error("Make sure than points2DExtrinsicPtr != nullptr && matchIndexesCameraPtr != nullptr.",
                      __LINE__, __FUNCTION__, __FILE__);
            if (!points2DExtrinsicPtr->empty() || !matchIndexesCameraPtr->empty())
                error("Variables points2DExtrinsicPtr and matchIndexesCameraPtr must be empty.",
                      __LINE__, __FUNCTION__, __FILE__);
            // Estimate and save SIFT file
            std::vector<cv::Point2f>& points2DExtrinsic = *points2DExtrinsicPtr;
            std::vector<unsigned int>& matchIndexesCamera = *matchIndexesCameraPtr;
            std::vector<cv::Mat> imagesWithCorners;
            for (auto viewIndex = 0u ; viewIndex < numberViews ; viewIndex++)
            {
                // Get right image
                const auto& imageAndPath = imageAndPaths.at(viewIndex * numberCameras + cameraIndex);
                const auto& image = imageAndPath.first;

                if (viewIndex % std::max(1, int(numberViews/4)) == 0)
                    opLog("Camera " + std::to_string(cameraIndex) + " - Image view "
                        + std::to_string(viewIndex+1) + "/" + std::to_string(numberViews),
                        Priority::High);

                // Sanity check
                if (imageSize.width != image.cols || imageSize.height != image.rows)
                    error("Detected images with different sizes in `" + imageFolder + "` All images"
                          " must have the same resolution.", __LINE__, __FUNCTION__, __FILE__);

                // Find grid corners
                bool found;
                std::vector<cv::Point2f> points2DVector;
                std::tie(found, points2DVector) = findAccurateGridCorners(image, gridInnerCornersCvSize);

                // Reorder & save 2D pixels points
                if (found)
                {
                    reorderPoints(points2DVector, gridInnerCornersCvSize, image);
                    for (auto i = 0 ; i < numberCorners ; i++)
                        matchIndexesCamera.emplace_back(viewIndex * numberCorners + i);
                }
                else
                {
                    points2DVector.clear();
                    points2DVector.resize(numberCorners, cv::Point2f{-1.f,-1.f});
                    opLog("Camera " + std::to_string(cameraIndex) + " - Image view "
                        + std::to_string(viewIndex+1) + "/" + std::to_string(numberViews)
                        + " - Chessboard not found.", Priority::High);
                }
                points2DExtrinsic.insert(points2DExtrinsic.end(), points2DVector.begin(), points2DVector.end());

                // Show image (with chessboard corners if found)
                if (saveImagesWithCorners)
                {
                    cv::Mat imageToPlot = image.clone();
                    if (found)
                        drawGridCorners(imageToPlot, gridInnerCornersCvSize, points2DVector);
                    imagesWithCorners.emplace_back(imageToPlot);
                }
            }

            // Save *.sift file for camera
            if (saveSIFTFile)
            {
                // const auto fileName = getFullFilePathNoExtension(imageAndPaths.at(cameraIndex).second) + ".sift";
                const auto fileName = getFileParentFolderPath(imageAndPaths.at(cameraIndex).second)
                                    + getFileNameFromCameraIndex(cameraIndex) + ".sift";
                writeVisualSFMSiftGPU(fileName, points2DExtrinsic);
            }

            // Save images with corners
            if (saveImagesWithCorners)
            {
                const auto folderWhereSavingImages = imageFolder + "images_with_corners/";
                // Create directory in case it did not exist
                makeDirectory(folderWhereSavingImages);
                const auto pathWhereSavingImages = folderWhereSavingImages + std::to_string(cameraIndex) + "_";
                // Save new images
                const std::string extension{".png"};
                auto fileRemoved = true;
                for (auto i = 0u ; i < imagesWithCorners.size() || fileRemoved; i++)
                {
                    const auto finalPath = pathWhereSavingImages + std::to_string(i+1) + extension;
                    // remove leftovers/previous files
                    // Note: If file is not deleted before cv::imwrite, Windows considers that the file
                    // was "only" modified at that time, not created
                    fileRemoved = {remove(finalPath.c_str()) == 0};
                    // save images on hhd in the desired place
                    if (i < imagesWithCorners.size())
                    {
                        const auto opMat = OP_CV2OPMAT(imagesWithCorners.at(i));
                        saveImage(opMat, finalPath);
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    const std::string sEmptyErrorMessage = "No chessboard was found in any of the images. Are you sure you"
        " are using the right value for `--grid_number_inner_corners`? Remember that it corresponds to the"
        " number of inner corners on the image (not the total number of corners!). I.e., it corresponds to"
        " the number of squares on each side minus 1! (or the number of total corners on each side - 1)."
        " It follows the OpenCV notation.";





    // Public functions
    void estimateAndSaveIntrinsics(
        const Point<int>& gridInnerCorners, const float gridSquareSizeMm, const int flags,
        const std::string& outputParameterFolder, const std::string& imageFolder, const std::string& serialNumber,
        const bool saveImagesWithCorners)
    {
        try
        {
            // Point<int> --> cv::Size
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            const cv::Size gridInnerCornersCvSize{gridInnerCorners.x, gridInnerCorners.y};

            // Read images in folder
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            const auto imageAndPaths = getImageAndPaths(imageFolder);

            // Get 2D grid corners of each image
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            std::vector<std::vector<cv::Point2f>> points2DVectors;
            std::vector<cv::Mat> imagesWithCorners;
            const auto imageSize = imageAndPaths.at(0).first.size();
            for (auto i = 0u ; i < imageAndPaths.size() ; i++)
            {
                opLog("\nImage " + std::to_string(i+1) + "/" + std::to_string(imageAndPaths.size()), Priority::High);
                const auto& image = imageAndPaths.at(i).first;

                // Sanity check
                if (imageSize.width != image.cols || imageSize.height != image.rows)
                    error("Detected images with different sizes in `" + imageFolder + "` All images"
                          " must have the same resolution.", __LINE__, __FUNCTION__, __FILE__);

                // Find grid corners
                bool found;
                std::vector<cv::Point2f> points2DVector;
                std::tie(found, points2DVector) = findAccurateGridCorners(image, gridInnerCornersCvSize);

                // Reorder & save 2D pixels points
                if (found)
                {
                    // For intrinsics order is irrelevant, so I do not care if it fails
                    const auto showWarning = false;
                    reorderPoints(points2DVector, gridInnerCornersCvSize, image, showWarning);
                    points2DVectors.emplace_back(points2DVector);
                }
                else
                    opLog("Chessboard not found in image " + imageAndPaths.at(i).second + ".", Priority::High);

                // Debugging (optional) - Show image (with chessboard corners if found)
                if (saveImagesWithCorners)
                {
                    cv::Mat imageToPlot = image.clone();
                    if (found)
                        drawGridCorners(imageToPlot, gridInnerCornersCvSize, points2DVector);
                    // cv::pyrDown(imageToPlot, imageToPlot);
                    // cv::imshow("Image View", imageToPlot);
                    // cv::waitKey(delayMilliseconds);
                    imagesWithCorners.emplace_back(imageToPlot);
                }
            }
            // Sanity check
            if (points2DVectors.empty())
                error(sEmptyErrorMessage, __LINE__, __FUNCTION__, __FILE__);

            // Run calibration
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // objects3DVector is the same one for each image
            const std::vector<std::vector<cv::Point3f>> objects3DVectors(
                points2DVectors.size(), getObjects3DVector(gridInnerCornersCvSize, gridSquareSizeMm));
            const auto intrinsics = calcIntrinsicParameters(imageSize, points2DVectors, objects3DVectors, flags);

            // Save intrinsics/results
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            Matrix opCameraMatrix = OP_CV2OPMAT(intrinsics.cameraMatrix);
            Matrix opDistortionCoefficients = OP_CV2OPMAT(intrinsics.distortionCoefficients);
            CameraParameterReader cameraParameterReader{
                serialNumber, opCameraMatrix, opDistortionCoefficients };
            cameraParameterReader.writeParameters(outputParameterFolder);

            // Debugging (optional) - Save images with corners
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            if (saveImagesWithCorners)
            {
                const auto folderWhereSavingImages = imageFolder + "images_with_corners/";
                // Create directory in case it did not exist
                makeDirectory(folderWhereSavingImages);
                // Save new images
                const std::string extension{".png"};
                auto fileRemoved = true;
                for (auto i = 0u ; i < imagesWithCorners.size() || fileRemoved; i++)
                {
                    const auto finalPath = folderWhereSavingImages + std::to_string(i+1) + extension;
                    // remove leftovers/previous files
                    // Note: If file is not deleted before cv::imwrite, Windows considers that the file
                    // was "only" modified at that time, not created
                    fileRemoved = {remove(finalPath.c_str()) == 0};
                    // save images on hhd in the desired place
                    if (i < imagesWithCorners.size())
                    {
                        const auto opMat = OP_CV2OPMAT(imagesWithCorners.at(i));
                        saveImage(opMat, finalPath);
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void estimateAndSaveExtrinsics(
        const std::string& parameterFolder, const std::string& imageFolder, const Point<int>& gridInnerCorners,
        const float gridSquareSizeMm, const int index0, const int index1, const bool imagesAreUndistorted,
        const bool combineCam0Extrinsics)
    {
        try
        {
            #ifdef USE_EIGEN
                // For debugging
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                const auto coutResults = false;
                // const auto coutResults = true;
                const bool coutAndImshowVerbose = false;

                // Point<int> --> cv::Size
                const cv::Size gridInnerCornersCvSize{gridInnerCorners.x, gridInnerCorners.y};

                // Load intrinsic parameters
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                CameraParameterReader cameraParameterReader;
                cameraParameterReader.readParameters(parameterFolder);
                const auto cameraSerialNumbers = cameraParameterReader.getCameraSerialNumbers();
                const auto opRealCameraDistortions = cameraParameterReader.getCameraDistortions();
                OP_OP2CVVECTORMAT(realCameraDistortions, opRealCameraDistortions)
                auto opCameraIntrinsicsSubset = cameraParameterReader.getCameraIntrinsics();
                OP_OP2CVVECTORMAT(cameraIntrinsicsSubset, opCameraIntrinsicsSubset)
                auto cameraDistortionsSubset = (imagesAreUndistorted ?
                    std::vector<cv::Mat>{realCameraDistortions.size()}
                    : realCameraDistortions);
                // Only use the 2 desired ones
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                cameraIntrinsicsSubset = {cameraIntrinsicsSubset.at(index0), cameraIntrinsicsSubset.at(index1)};
                cameraDistortionsSubset = {cameraDistortionsSubset.at(index0), cameraDistortionsSubset.at(index1)};
                // Base extrinsics
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                cv::Mat extrinsicsCam0 = cv::Mat::eye(4, 4, realCameraDistortions.at(0).type());
                bool cam0IsOrigin = true;
                if (combineCam0Extrinsics)
                {
                    const cv::Mat cameraExtrinsicsAtIndex0 = OP_OP2CVCONSTMAT(
                        cameraParameterReader.getCameraExtrinsics().at(index0));
                    cameraExtrinsicsAtIndex0.copyTo(extrinsicsCam0(cv::Rect{0,0,4,3}));
                    cam0IsOrigin = cv::norm(extrinsicsCam0 - cv::Mat::eye(4, 4, extrinsicsCam0.type())) < 1e-9;
                }

                // Number cameras and image paths
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                const auto numberCameras = cameraParameterReader.getNumberCameras();
                opLog("\nDetected " + std::to_string(numberCameras) + " cameras from your XML files on:\n"
                    + parameterFolder + "\nRemove wrong/extra XML files if this number of cameras does not"
                    + " correspond with the number of cameras recorded in:\n" + imageFolder + "\n",
                    Priority::High);
                const auto imagePaths = getImagePaths(imageFolder);
                // Sanity check
                if (imagePaths.size() % numberCameras != 0)
                    error("You indicated that there are " + std::to_string(numberCameras)
                          + " cameras. However, we found a total of " + std::to_string(imagePaths.size())
                          + " images, which should be possible to divide into the number of cameras with no"
                          " remainder.",
                          __LINE__, __FUNCTION__, __FILE__);

                // Estimate extrinsic parameters per image
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                opLog("Calibrating camera " + cameraSerialNumbers.at(index1) + " with respect to camera "
                    + cameraSerialNumbers.at(index0) + "...", Priority::High);
                const auto numberViews = imagePaths.size() / numberCameras;
                auto counterValidImages = 0u;
                std::vector<Eigen::Matrix4d> MCam1ToCam0s;
                for (auto i = 0u ; i < imagePaths.size() ; i+=numberCameras)
                {
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    const auto pathCam0 = imagePaths[i+index0];
                    const auto pathCam1 = imagePaths[i+index1];
                    if (coutResults || i/numberCameras % int(numberViews/10) == 0)
                        opLog("It " + std::to_string(i/numberCameras+1) + "/" + std::to_string(numberViews) + ": "
                            + getFileNameAndExtension(pathCam0) + " & "
                            + getFileNameAndExtension(pathCam1) + "...", Priority::High);

                    // Extrinsic parameters extractor
                    Eigen::Matrix3d RGridToMainCam0;
                    Eigen::Matrix3d RGridToMainCam1;
                    Eigen::Vector3d tGridToMainCam0;
                    Eigen::Vector3d tGridToMainCam1;
                    bool valid = true;
                    std::tie(valid, RGridToMainCam0, tGridToMainCam0, RGridToMainCam1, tGridToMainCam1)
                        = getExtrinsicParameters({pathCam0, pathCam1}, gridInnerCornersCvSize, gridSquareSizeMm,
                                                 false,
                                                 // coutAndImshowVerbose, // It'd display all images with grid
                                                 cameraIntrinsicsSubset, cameraDistortionsSubset);
                    if (valid)
                    {
                        counterValidImages++;
                        if (coutAndImshowVerbose)
                        {
                            opLog("########## Extrinsic parameters extractor ##########", Priority::High);
                            opLog("R_gf", Priority::High);
                            opLog(RGridToMainCam0, Priority::High);
                            opLog("t_gf", Priority::High);
                            opLog(tGridToMainCam0, Priority::High);
                            opLog("R_gb", Priority::High);
                            opLog(RGridToMainCam1, Priority::High);
                            opLog("t_gb", Priority::High);
                            opLog(tGridToMainCam1, Priority::High);
                            opLog("\n", Priority::High);
                        }

                        // MCam1ToCam0 - Projection matrix estimator
                        if (coutAndImshowVerbose)
                            opLog("########## Projection Matrix from secondary camera to main camera ##########",
                                Priority::High);
                        MCam1ToCam0s.emplace_back(
                            getMFromCam1ToCam0(RGridToMainCam0, tGridToMainCam0, RGridToMainCam1, tGridToMainCam1,
                                coutAndImshowVerbose));
                        if (coutResults)
                        {
                            if (coutAndImshowVerbose)
                                opLog("M_bg:", Priority::High);
                            opLog(MCam1ToCam0s.back(), Priority::High);
                            opLog(" ", Priority::High);
                        }
                    }
                    else
                    {
                        if (coutResults)
                            opLog("Invalid frame (chessboard not found).", Priority::High);
                    }
                }
                // Sanity check
                if (MCam1ToCam0s.empty())
                    error(sEmptyErrorMessage, __LINE__, __FUNCTION__, __FILE__);
                opLog("Finished processing images.", Priority::High);

                // Pseudo RANSAC calibration
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                const auto MCam1ToCam0Noisy = getMAverage(MCam1ToCam0s);
                opLog("Estimated initial (noisy?) projection matrix.", Priority::High);
                auto MCam1ToCam0 = getMAverage(MCam1ToCam0s, MCam1ToCam0Noisy);
                while ((MCam1ToCam0 - getMAverage(MCam1ToCam0s, MCam1ToCam0)).norm() > 1e-3)
                {
                    if (coutResults)
                        opLog("Repeated robustness method...", Priority::High);
                    MCam1ToCam0 = getMAverage(MCam1ToCam0s, MCam1ToCam0);
                }
                opLog("Estimated robust projection matrix.", Priority::High);
                opLog("norm(M_robust-M_noisy): " + std::to_string((MCam1ToCam0Noisy - MCam1ToCam0).norm()),
                    Priority::High);



                // Show errors
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                if (coutAndImshowVerbose)
                {
                    opLog("\n-----------------------------------------------------------------------------------"
                        "-------------------\nErrors:", Priority::High);
                    // Errors
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    for (auto i = 0u ; i < MCam1ToCam0s.size() ; i++)
                    {
                        opLog("tCam1WrtCam0:", Priority::High);
                        opLog(MCam1ToCam0s.at(i).block<3,1>(0,3).transpose(), Priority::High);
                    }
                    opLog(" ", Priority::High);

                    opLog("tCam1WrtCam0:", Priority::High);
                    opLog(MCam1ToCam0.block<3,1>(0,3).transpose(), Priority::High);
                    opLog(" ", Priority::High);

                    // Rotation matrix in degrees Rodrigues(InputArray src, OutputArray dst
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    const auto rad2deg = 180 / PI;
                    for (auto i = 0u ; i < MCam1ToCam0s.size() ; i++)
                    {
                        Eigen::Matrix3d R_secondaryToMain = MCam1ToCam0s.at(i).block<3,3>(0,0);
                        opLog("rodrigues:", Priority::High);
                        opLog((getRodriguesVector(R_secondaryToMain).t() * rad2deg), Priority::High);
                    }
                    Eigen::Matrix3d R_secondaryToMain = MCam1ToCam0.block<3,3>(0,0);
                    opLog("rodrigues:", Priority::High);
                    opLog((getRodriguesVector(R_secondaryToMain).t() * rad2deg), Priority::High);
                }

                // Show final result
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                if (coutResults)
                {
                    opLog("\n\n\n---------------------------------------------------------------------------"
                        "---------------------------", Priority::High);
                    opLog(std::to_string(counterValidImages) + " valid images.", Priority::High);
                    opLog("Initial (noisy?) projection matrix:", Priority::High);
                    opLog(MCam1ToCam0Noisy, Priority::High);
                    opLog("\nFinal projection matrix (mm):", Priority::High);
                    opLog(MCam1ToCam0, Priority::High);
                    opLog(" ", Priority::High);
                }

                // mm --> m
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                MCam1ToCam0.block<3,1>(0,3) *= 1e-3;

                // Eigen --> cv::Mat
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                cv::Mat cvMatExtrinsics;
                Eigen::MatrixXd eigenExtrinsics = MCam1ToCam0.block<3,4>(0,0);
                cv::eigen2cv(eigenExtrinsics, cvMatExtrinsics);

                // pos_target_origen = pos_target_cam1 * pos_cam1_origin
                // Example: pos_31 = pos_32 * pos_21
                if (!cam0IsOrigin)
                    cvMatExtrinsics *= extrinsicsCam0;

                // Final projection matrix
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                opLog("\nFinal projection matrix w.r.t. global origin (meters):", Priority::High);
                opLog(cvMatExtrinsics, Priority::High);
                opLog(" ", Priority::High);

                // Save result
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                CameraParameterReader cameraParameterReaderFinal{
                    cameraSerialNumbers.at(index1),
                    OP_CV2OPMAT(cameraIntrinsicsSubset.at(1)),
                    OP_CV2OPMAT(realCameraDistortions.at(index1)),
                    OP_CV2OPMAT(cvMatExtrinsics)
                };
                cameraParameterReaderFinal.writeParameters(parameterFolder);

                // Let the rendered image to be displayed
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                if (coutAndImshowVerbose)
                    cv::waitKey(0);
            #else
                UNUSED(parameterFolder);
                UNUSED(imageFolder);
                UNUSED(gridInnerCorners);
                UNUSED(gridSquareSizeMm);
                UNUSED(index0);
                UNUSED(index1);
                UNUSED(imagesAreUndistorted);
                UNUSED(combineCam0Extrinsics);
                error("CMake flag `USE_EIGEN` required when compiling OpenPose`.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    #if defined(USE_CERES) && defined(USE_EIGEN)
        double computeReprojectionErrorInPixels(
            const std::vector<std::vector<cv::Point2f>>& points2DVectorsExtrinsic, const Eigen::MatrixXd& BAValid,
            const Eigen::Matrix<double, 3, Eigen::Dynamic>& points3D, const std::vector<cv::Mat> cameraExtrinsics,
            const std::vector<cv::Mat> cameraIntrinsics, const bool verbose = true)
        {
            try
            {
                // compute the average reprojection error
                const unsigned int numberCameras = cameraIntrinsics.size();
                const unsigned int numberPoints = points2DVectorsExtrinsic[0].size();
                double sumError = 0.;
                int sumPoint = 0;
                double maxError = 0.;
                int maxCamIdx = -1;
                int maxPtIdx = -1;
                for (auto cameraIndex = 0u; cameraIndex < numberCameras; cameraIndex++)
                {
                    const cv::Mat cameraMatrix = cameraIntrinsics[cameraIndex] * cameraExtrinsics[cameraIndex];
                    for (auto i = 0u; i < numberPoints; i++)
                    {
                        if (!BAValid(cameraIndex, i))
                            continue;
                        const auto& point3d = &points3D.data()[3*i];
                        const double KX = cameraMatrix.at<double>(0, 0) * point3d[0]
                                        + cameraMatrix.at<double>(0, 1) * point3d[1]
                                        + cameraMatrix.at<double>(0, 2) * point3d[2]
                                        + cameraMatrix.at<double>(0, 3);
                        const double KY = cameraMatrix.at<double>(1, 0) * point3d[0]
                                        + cameraMatrix.at<double>(1, 1) * point3d[1]
                                        + cameraMatrix.at<double>(1, 2) * point3d[2]
                                        + cameraMatrix.at<double>(1, 3);
                        const double KZ = cameraMatrix.at<double>(2, 0) * point3d[0]
                                        + cameraMatrix.at<double>(2, 1) * point3d[1]
                                        + cameraMatrix.at<double>(2, 2) * point3d[2]
                                        + cameraMatrix.at<double>(2, 3);
                        const double xDiff = KX / KZ - points2DVectorsExtrinsic[cameraIndex][i].x;
                        const double yDiff = KY / KZ - points2DVectorsExtrinsic[cameraIndex][i].y;
                        const double error = sqrt(xDiff*xDiff + yDiff*yDiff);
                        sumError += error;
                        sumPoint++;
                        if (error > maxError)
                        {
                            maxError = error;
                            maxPtIdx = i;
                            maxCamIdx = cameraIndex;
                        }
                    }
                }
                if (sumPoint == 0)
                    error("Number of inlier points is 0 (with " + std::to_string(numberCameras)
                          + " cameras and " + std::to_string(numberPoints) + " total points).",
                          __LINE__, __FUNCTION__, __FILE__);
                // Debugging
                if (verbose)
                    opLog("Reprojection Error info: Max error: " + std::to_string(maxError) + ";\t in cam idx "
                        + std::to_string(maxCamIdx) + " with pt idx: " + std::to_string(maxPtIdx) + " & pt 2D: "
                        + std::to_string(points2DVectorsExtrinsic[maxCamIdx][maxPtIdx].x) + "x"
                        + std::to_string(points2DVectorsExtrinsic[maxCamIdx][maxPtIdx].y), Priority::High);
                return sumError / sumPoint;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return -1;
            }
        }

        void removeOutliersReprojectionError(
            std::vector<std::vector<cv::Point2f>>& points2DVectorsExtrinsic, Eigen::MatrixXd& BAValid,
            const Eigen::Matrix<double, 3, Eigen::Dynamic>& points3D, const std::vector<cv::Mat> cameraExtrinsics,
            const std::vector<cv::Mat> cameraIntrinsics, const double errorThreshold = 5.)
        {
            try
            {
                std::vector<unsigned int> indexesToRemove;
                const unsigned int numberCameras = cameraIntrinsics.size();
                const unsigned int numberPoints = points2DVectorsExtrinsic[0].size();
                for (auto cameraIndex = 0u; cameraIndex < numberCameras; cameraIndex++)
                {
                    const cv::Mat cameraMatrix = cameraIntrinsics[cameraIndex] * cameraExtrinsics[cameraIndex];
                    for (auto i = 0u; i < numberPoints; i++)
                    {
                        if (!BAValid(cameraIndex, i))
                            continue;
                        const auto& point3d = &points3D.data()[3*i];
                        const double KX = cameraMatrix.at<double>(0, 0) * point3d[0]
                                        + cameraMatrix.at<double>(0, 1) * point3d[1]
                                        + cameraMatrix.at<double>(0, 2) * point3d[2]
                                        + cameraMatrix.at<double>(0, 3);
                        const double KY = cameraMatrix.at<double>(1, 0) * point3d[0]
                                        + cameraMatrix.at<double>(1, 1) * point3d[1]
                                        + cameraMatrix.at<double>(1, 2) * point3d[2]
                                        + cameraMatrix.at<double>(1, 3);
                        const double KZ = cameraMatrix.at<double>(2, 0) * point3d[0]
                                        + cameraMatrix.at<double>(2, 1) * point3d[1]
                                        + cameraMatrix.at<double>(2, 2) * point3d[2]
                                        + cameraMatrix.at<double>(2, 3);
                        const double xDiff = KX / KZ - points2DVectorsExtrinsic[cameraIndex][i].x;
                        const double yDiff = KY / KZ - points2DVectorsExtrinsic[cameraIndex][i].y;
                        const double error = sqrt(xDiff*xDiff + yDiff*yDiff);
                        if (error > errorThreshold)
                        {
                            indexesToRemove.emplace_back(i);
                            BAValid(cameraIndex, i) = 0;
                        }
                    }
                }
                // Sort + Remove duplicates
                std::sort(indexesToRemove.begin(), indexesToRemove.end());
                indexesToRemove.erase(
                    std::unique(indexesToRemove.begin(), indexesToRemove.end()), indexesToRemove.end());
                // Sanity check
                if (numberPoints <= indexesToRemove.size())
                    error("All samples are considered outliers, no images left ("
                          + std::to_string(numberPoints) + " total points vs. "
                          + std::to_string(indexesToRemove.size()) + " outliers).",
                          __LINE__, __FUNCTION__, __FILE__);
                // Pros / cons:
                // - Pros: It reduces the size of points (faster).
                // - Cons: Scaling does not work.
                // // Remove outliers (if any)
                // if (!indexesToRemove.empty())
                // {
                //     const unsigned int numberPointsRansac = numberPoints - indexesToRemove.size();
                //     std::vector<std::vector<cv::Point2f>> points2DVectorsExtrinsicRansac(
                //         numberCameras, std::vector<cv::Point2f>(numberPointsRansac));
                //     Eigen::Matrix<double, 3, Eigen::Dynamic> points3DRansac(3, numberPointsRansac);
                //     Eigen::MatrixXd BAValidRansac = Eigen::MatrixXd::Zero(numberCameras, numberPointsRansac);
                //     auto counterRansac = 0u;
                //     for (auto i = 0u ; i < indexesToRemove.size()+1 ; i++)
                //     {
                //         // Otherwise, it would not get the points after the last outlier
                //         const auto& indexToRemove = (
                //             i < indexesToRemove.size() ? indexesToRemove[i] : numberPoints);
                //         while (counterRansac < indexToRemove && counterRansac < numberPoints)
                //         {
                //             // Fill 2D coordinate
                //             for (auto cameraIndex = 0u; cameraIndex < numberCameras; cameraIndex++)
                //                 points2DVectorsExtrinsicRansac[cameraIndex][counterRansac-i]
                //                     = points2DVectorsExtrinsic[cameraIndex][counterRansac];
                //                 // points2DVectorsExtrinsicRansac.at(cameraIndex).at(counterRansac-i)
                //                 //     = points2DVectorsExtrinsic.at(cameraIndex).at(counterRansac);
                //             // Fill 3D coordinate
                //             const auto* const point3D = &points3D.data()[3*counterRansac];
                //             auto* point3DRansac = &points3DRansac.data()[3*(counterRansac-i)];
                //             point3DRansac[0] = point3D[0];
                //             point3DRansac[1] = point3D[1];
                //             point3DRansac[2] = point3D[2];
                //             // Fill BAValidRansac
                //             BAValidRansac.col(counterRansac-i) = BAValid.col(counterRansac);
                //             // Update counter
                //             counterRansac++;
                //         }
                //         counterRansac++;
                //     }
                //     // Asign back
                //     std::swap(points2DVectorsExtrinsic, points2DVectorsExtrinsicRansac);
                //     std::swap(points3D, points3DRansac);
                //     std::swap(BAValid, BAValidRansac);
                // }
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        Eigen::MatrixXd getInlierAndOutliers(
            const std::vector<std::vector<cv::Point2f>>& points2DVectorsExtrinsic)
        {
            try
            {
                const auto numberCameras = points2DVectorsExtrinsic.size();
                // Update inliers & outliers
                // This is a valid reprojection term
                Eigen::MatrixXd BAValid = Eigen::MatrixXd::Zero(numberCameras, points2DVectorsExtrinsic[0].size());
                for (auto i = 0u; i < points2DVectorsExtrinsic[0].size(); i++)
                {
                    auto visibleViewCounter = 0u;
                    // std::vector<cv::Mat> pointCameraMatrices;
                    for (auto cameraIndex = 0u ; cameraIndex < numberCameras ; cameraIndex++)
                    {
                        if (points2DVectorsExtrinsic[cameraIndex][i].x >= 0)  // visible in this camera
                        {
                            visibleViewCounter++;
                            if (visibleViewCounter > 1u)
                                break;
                        }
                    }
                    // If visible in >1 camera, point used in bundle adjustment
                    if (visibleViewCounter > 1u)
                        for (auto cameraIndex = 0u ; cameraIndex < numberCameras ; cameraIndex++)
                            if (points2DVectorsExtrinsic[cameraIndex][i].x >= 0)
                                BAValid(cameraIndex, i) = 1;
                }
                return BAValid;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return Eigen::MatrixXd{};
            }
        }

        void removeOutliersReprojectionErrorIterative(
            std::vector<std::vector<cv::Point2f>>& points2DVectorsExtrinsic, Eigen::MatrixXd& BAValid,
            const Eigen::Matrix<double, 3, Eigen::Dynamic>& points3D, const std::vector<cv::Mat> cameraExtrinsics,
            const std::vector<cv::Mat> cameraIntrinsics, const double errorThresholdRelative = 5.)
        {
            try
            {
                // Outlier removal
                auto reprojectionError = computeReprojectionErrorInPixels(
                    points2DVectorsExtrinsic, BAValid, points3D, cameraExtrinsics, cameraIntrinsics, false);
                auto reprojectionErrorPrevious = reprojectionError+1;
                while (reprojectionError != reprojectionErrorPrevious)
                {
                    reprojectionErrorPrevious = reprojectionError;
                    // 1 pixels is a lot for HD images...
                    const auto errorThreshold = fastMax(2*reprojectionError, errorThresholdRelative);
                    removeOutliersReprojectionError(
                        points2DVectorsExtrinsic, BAValid, points3D, cameraExtrinsics, cameraIntrinsics,
                        errorThreshold);
                    reprojectionError = computeReprojectionErrorInPixels(
                        points2DVectorsExtrinsic, BAValid, points3D, cameraExtrinsics, cameraIntrinsics);
                    // Verbose
                    opLog("Reprojection Error (after outlier removal iteration): "
                        + std::to_string(reprojectionError) + " pixels,\twith error threshold of "
                        + std::to_string(errorThreshold) + " pixels.", Priority::High);
                }
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        Eigen::Matrix<double, 3, Eigen::Dynamic> reconstruct3DPoints(
            const std::vector<std::vector<cv::Point2f>>& points2DVectorsExtrinsic,
            const std::vector<cv::Mat>& cameraIntrinsics, const std::vector<cv::Mat>& cameraExtrinsics,
            const int numberCameras, const cv::Size& imageSize)
        {
            try
            {
                // Initialize to 0
                Eigen::Matrix<double, 3, Eigen::Dynamic> points3D(3, points2DVectorsExtrinsic[0].size());
                points3D.setZero();
                // Compute the initial camera matrices
                std::vector<cv::Mat> cameraMatrices(numberCameras);
                for (auto cameraIndex = 0 ; cameraIndex < numberCameras ; cameraIndex++)
                    cameraMatrices[cameraIndex] = cameraIntrinsics[cameraIndex] * cameraExtrinsics[cameraIndex];
                const auto imageRatio = std::sqrt(imageSize.area() / 1310720.);
                const auto reprojectionMaxAcceptable = 25 * imageRatio;
                for (auto i = 0u; i < points2DVectorsExtrinsic[0].size(); i++)
                {
                    std::vector<cv::Mat> pointCameraMatrices;
                    std::vector<cv::Point2d> pointsOnEachCamera;
                    for (auto cameraIndex = 0 ; cameraIndex < numberCameras ; cameraIndex++)
                    {
                        if (points2DVectorsExtrinsic[cameraIndex][i].x >= 0)  // visible in this camera
                        {
                            pointCameraMatrices.emplace_back(cameraMatrices[cameraIndex]);
                            const auto& point2D = points2DVectorsExtrinsic[cameraIndex][i];
                            // cv::Point2f --> cv::Point2d
                            pointsOnEachCamera.emplace_back(cv::Point2d{point2D.x, point2D.y});
                        }
                    }
                    // if visible in one camera, no triangulation and not used in bundle adjustment.
                    if (pointCameraMatrices.size() > 1u)
                    {
                        cv::Mat reconstructedPoint;
                        triangulateWithOptimization(
                            reconstructedPoint, pointCameraMatrices, pointsOnEachCamera, reprojectionMaxAcceptable);
                        auto* points3DPtr = &points3D.data()[3*i];
                        points3DPtr[0] = reconstructedPoint.at<double>(0, 0) / reconstructedPoint.at<double>(3, 0);
                        points3DPtr[1] = reconstructedPoint.at<double>(1, 0) / reconstructedPoint.at<double>(3, 0);
                        points3DPtr[2] = reconstructedPoint.at<double>(2, 0) / reconstructedPoint.at<double>(3, 0);
                    }
                }
                return points3D;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return Eigen::Matrix<double, 3, Eigen::Dynamic>();
            }
        }

        // // BundleAdjustmentCost not used because it was slower than differentiation by parts
        // struct BundleAdjustmentCost
        // {
        //     BundleAdjustmentCost(const std::vector<std::vector<cv::Point2f>>& points2DVectorsExtrinsic,
        //                          const std::vector<cv::Mat>& cameraIntrinsicsCvMat,
        //                          const Eigen::MatrixXd& BAValid) :
        //         points2DVectorsExtrinsic(points2DVectorsExtrinsic), BAValid(BAValid)
        //     {
        //         numberCameras = cameraIntrinsicsCvMat.size();
        //         if (points2DVectorsExtrinsic.size() != numberCameras)
        //             error("#view of points != #camera intrinsics.", __LINE__, __FUNCTION__, __FILE__);
        //         numberPoints = points2DVectorsExtrinsic[0].size();
        //         numberProjection = BAValid.sum();
        //         for (auto cameraIndex = 0u; cameraIndex < numberCameras; cameraIndex++)
        //         {
        //             if (cameraIntrinsicsCvMat[cameraIndex].cols!=3 || cameraIntrinsicsCvMat[cameraIndex].rows!=3)
        //                 error("Intrinsics passed in are not 3 x 3.", __LINE__, __FUNCTION__, __FILE__);
        //             cameraIntrinsics.resize(numberCameras);
        //             for (auto x = 0; x < 3; x++)
        //                 for (auto y = 0; y < 3; y++)
        //                     cameraIntrinsics[cameraIndex](x,y) = cameraIntrinsicsCvMat[cameraIndex].at<double>(x,y);
        //         }
        //     }

        //     template <typename T>
        //     bool operator()(T const* const* parameters, T* residuals) const;

        //     const std::vector<std::vector<cv::Point2f>>& points2DVectorsExtrinsic;
        //     const Eigen::MatrixXd& BAValid;
        //     std::vector<Eigen::Matrix<double, 3, 3>> cameraIntrinsics;
        //     uint numberCameras, numberPoints, numberProjection;
        // };

        // template <typename T>
        // bool BundleAdjustmentCost::operator()(T const* const* parameters, T* residuals) const
        // {
        //     // cameraExtrinsics: angle axis + translation (6 x (#Cam - 1)), camera 0 is always [I | 0]
        //     // pt3D: 3D points (3 x #Points)
        //     const T* cameraExtrinsics = parameters[0];
        //     const T* pt3D = parameters[1];
        //     const Eigen::Map< const Eigen::Matrix<T, 3, Eigen::Dynamic> > pt3DWorld(pt3D, 3, numberPoints);
        //     uint countProjection = 0u;
        //     for (auto cameraIndex = 0u; cameraIndex < numberCameras; cameraIndex++)
        //     {
        //         Eigen::Matrix<T, 3, Eigen::Dynamic> pt3DCamera = pt3DWorld;
        //         if (cameraIndex > 0u)
        //         {
        //             const Eigen::Map< const Eigen::Matrix<T, 3, 1> > translation(
        //             cameraExtrinsics + 6 * (cameraIndex - 1) + 3);  // minus 1!
        //             Eigen::Matrix<T, 3, 3> rotation;
        //             ceres::AngleAxisToRotationMatrix(cameraExtrinsics + 6 * (cameraIndex - 1), rotation.data());//-1!
        //             pt3DCamera = rotation * pt3DCamera;
        //             pt3DCamera.colwise() += translation;
        //         }
        //         const Eigen::Matrix<T, 3, Eigen::Dynamic> pt2DHomogeneous = cameraIntrinsics[cameraIndex].cast<T>()
        //                                                                   * pt3DCamera;
        //         const Eigen::Matrix<T, 1, Eigen::Dynamic> ptx = pt2DHomogeneous.row(0).cwiseQuotient(
        //             pt2DHomogeneous.row(2));
        //         const Eigen::Matrix<T, 1, Eigen::Dynamic> pty = pt2DHomogeneous.row(1).cwiseQuotient(
        //             pt2DHomogeneous.row(2));
        //         for (auto i = 0u; i < numberPoints; i++)
        //         {
        //             if (!BAValid(cameraIndex, i))   // no data for this point
        //                 continue;
        //             residuals[2 * countProjection + 0] = ptx(0, i) - T(points2DVectorsExtrinsic[cameraIndex][i].x);
        //             residuals[2 * countProjection + 1] = pty(0, i) - T(points2DVectorsExtrinsic[cameraIndex][i].y);
        //             countProjection++;
        //         }
        //     }
        //     // sanity check
        //     if (countProjection != numberProjection)
        //         error("Wrong number of constraints in bundle adjustment", __LINE__, __FUNCTION__, __FILE__);
        //     return true;
        // }

        struct BundleAdjustmentUnit
        {
            BundleAdjustmentUnit(const cv::Point2f& pt2d, const cv::Mat& intrinsics) :
                pt2d{pt2d},
                intrinsics{intrinsics}
            {
                if (intrinsics.cols != 3 || intrinsics.rows != 3)
                    error("Intrinsics passed in are not 3 x 3.", __LINE__, __FUNCTION__, __FILE__);
                if (intrinsics.type() != CV_64FC1)
                    error("Intrinsics passed in must be in double.", __LINE__, __FUNCTION__, __FILE__);
                cv::Mat pt2DHomogeneous(3, 1, CV_64FC1);
                pt2DHomogeneous.at<double>(0, 0) = pt2d.x;
                pt2DHomogeneous.at<double>(1, 0) = pt2d.y;
                pt2DHomogeneous.at<double>(2, 0) = 1;
                const cv::Mat calibrated = intrinsics.inv() * pt2DHomogeneous;
                pt2dCalibrated.x = calibrated.at<double>(0, 0) / calibrated.at<double>(2, 0);
                pt2dCalibrated.y = calibrated.at<double>(1, 0) / calibrated.at<double>(2, 0);
            }
            const cv::Point2f& pt2d;
            const cv::Mat& intrinsics;
            cv::Point2f pt2dCalibrated;

            template <typename T>
            bool operator()(const T* camera, const T* point, T* residuals) const
            {
                // camera (6): angle axis + translation4
                // point (3): X, Y, Z
                // residuals (2): x, y
                T P[3];
                ceres::AngleAxisRotatePoint(camera, point, P);
                Eigen::Matrix<T, 3, 3, Eigen::ColMajor> R;
                P[0] += camera[3]; P[1] += camera[4]; P[2] += camera[5];

                residuals[0] = P[0] / P[2] - T(pt2dCalibrated.x);
                residuals[1] = P[1] / P[2] - T(pt2dCalibrated.y);
                return true;
            }

            template <typename T>
            bool operator()(const T* point, T* residuals) const
            {
                // point (3): X, Y, Z
                // residuals (2): x, y

                residuals[0] = point[0] / point[2] - T(pt2dCalibrated.x);
                residuals[1] = point[1] / point[2] - T(pt2dCalibrated.y);
                return true;
            }
        };

        // // Compute the Jacobian of rotation matrix w.r.t angle axis (rotation matrix in column major order)
        // void AngleAxisToRotationMatrixDerivative(const double* pose, double* dR_data, const int idj = 0,
        //                                          const int numberColumns = 3)
        // {
        //     Eigen::Map< Eigen::Matrix<double, 9, Eigen::Dynamic, Eigen::RowMajor> > dR(
        //         dR_data, 9, numberColumns);
        //     std::fill(dR_data, dR_data + 9 * numberColumns, 0.0);
        //     const double theta2 = pose[0] * pose[0] + pose[1] * pose[1] + pose[2] * pose[2];
        //     if (theta2 > std::numeric_limits<double>::epsilon())
        //     {
        //         const double theta = sqrt(theta2);
        //         const double s = sin(theta);
        //         const double c = cos(theta);
        //         const Eigen::Map< const Eigen::Matrix<double, 3, 1> > u(pose);
        //         Eigen::VectorXd e(3);
        //         e[0] = pose[0] / theta; e[1] = pose[1] / theta; e[2] = pose[2] / theta;

        //         // dR / dtheta
        //         Eigen::Matrix<double, 9, 1> dRdth(9, 1);
        //         Eigen::Map< Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > dRdth_(dRdth.data());
        //         // skew symmetric
        //         dRdth_ << 0.0, -e[2], e[1],
        //                   e[2], 0.0, -e[0],
        //                   -e[1], e[0], 0.0;
        //         // dRdth_ = dRdth_ * c - Matrix<double, 3, 3>::Identity() * s + s * e * e.transpose();
        //         dRdth_ = - dRdth_ * c - Eigen::Matrix<double, 3, 3>::Identity() * s + s * e * e.transpose();

        //         // dR / de
        //         Eigen::Matrix<double, 9, 3, Eigen::RowMajor> dRde(9, 3);
        //         // d(ee^T) / de
        //         dRde <<
        //             2 * e[0], 0., 0.,
        //             e[1], e[0], 0.,
        //             e[2], 0., e[0],
        //             e[1], e[0], 0.,
        //             0., 2 * e[1], 0.,
        //             0., e[2], e[1],
        //             e[2], 0., e[0],
        //             0., e[2], e[1],
        //             0., 0., 2 * e[2];
        //         Eigen::Matrix<double, 9, 3, Eigen::RowMajor> dexde(9, 3);
        //         dexde <<
        //             0, 0, 0,
        //             0, 0, -1,
        //             0, 1, 0,
        //             0, 0, 1,
        //             0, 0, 0,
        //             -1, 0, 0,
        //             0, -1, 0,
        //             1, 0, 0,
        //             0, 0, 0;
        //         // dRde = dRde * (1. - c) + c * dexde;
        //         dRde = dRde * (1. - c) - s * dexde;
        //         Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dedu
        //             = Eigen::Matrix<double, 3, 3>::Identity() / theta - u * u.transpose() / theta2 / theta;

        //         dR.block(0, 3 * idj, 9, 3) = dRdth * e.transpose() + dRde * dedu;
        //     }
        //     else
        //     {
        //         dR(1, 3 * idj + 2) = 1;
        //         dR(2, 3 * idj + 1) = -1;
        //         dR(3, 3 * idj + 2) = -1;
        //         dR(5, 3 * idj) = 1;
        //         dR(6, 3 * idj + 1) = 1;
        //         dR(7, 3 * idj) = -1;
        //     }
        // }

        // // Compute the derivative w.r.t AB from dA
        // void SparseProductDerivative(const double* const dA_data, const double* const B_data,
        //                              const std::vector<int>& parentIndexes, double* dAB_data,
        //                              const int numberColumns = 3)
        // {
        //     // d(AB) = AdB + (dA)B
        //     // Sparse for loop form
        //     std::fill(dAB_data, dAB_data + 3 * numberColumns, 0.0);
        //     for (int r = 0; r < 3; r++)
        //     {
        //         const int baseIndex = 3*r;
        //         for (const auto& parentIndex : parentIndexes)
        //         {
        //             const auto parentOffset = 3*parentIndex;
        //             for (int subIndex = 0; subIndex < 3; subIndex++)
        //             {
        //                 const auto finalOffset = parentOffset + subIndex;
        //                 dAB_data[numberColumns*r + finalOffset] +=
        //                     B_data[0] * dA_data[numberColumns*baseIndex + finalOffset]
        //                     + B_data[1] * dA_data[numberColumns*(baseIndex+1) + finalOffset]
        //                     + B_data[2] * dA_data[numberColumns*(baseIndex+2) + finalOffset];
        //             }
        //         }
        //     }
        // }

        // // BundleAdjustmentUnitJacobian not used because it was slower than automatic differentiation
        // class BundleAdjustmentUnitJacobian : public ceres::CostFunction
        // {
        // public:
        //     BundleAdjustmentUnitJacobian(const cv::Point2f& pt2d, const cv::Mat& intrinsics, const bool solveExt) :
        //         pt2d(pt2d), intrinsics(intrinsics), solveExt(solveExt)
        //     {
        //         if (intrinsics.cols != 3 || intrinsics.rows != 3)
        //             error("Intrinsics passed in are not 3 x 3.", __LINE__, __FUNCTION__, __FILE__);
        //         if (intrinsics.type() != CV_64FC1)
        //             error("Intrinsics passed in must be in double.", __LINE__, __FUNCTION__, __FILE__);

        //         cv::Mat pt2DHomogeneous(3, 1, CV_64FC1);
        //         pt2DHomogeneous.at<double>(0, 0) = pt2d.x;
        //         pt2DHomogeneous.at<double>(1, 0) = pt2d.y;
        //         pt2DHomogeneous.at<double>(2, 0) = 1;
        //         const cv::Mat calibrated = intrinsics.inv() * pt2DHomogeneous;
        //         pt2dCalibrated.x = calibrated.at<double>(0, 0) / calibrated.at<double>(2, 0);
        //         pt2dCalibrated.y = calibrated.at<double>(1, 0) / calibrated.at<double>(2, 0);

        //         CostFunction::set_num_residuals(2);
        //         auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
        //         parameter_block_sizes->clear();
        //         if (solveExt) parameter_block_sizes->push_back(6); // camera extrinsics
        //         parameter_block_sizes->push_back(3);  // 3D points
        //     }

        //     virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
        //     {
        //         double P[3];
        //         const double* ptr = solveExt ? P : parameters[0];
        //         if (solveExt)
        //         {
        //             const double* camera = parameters[0];
        //             ceres::AngleAxisRotatePoint(camera, parameters[1], P);
        //             P[0] += camera[3]; P[1] += camera[4]; P[2] += camera[5];
        //         }

        //         residuals[0] = ptr[0] / ptr[2] - pt2dCalibrated.x;
        //         residuals[1] = ptr[1] / ptr[2] - pt2dCalibrated.y;

        //         if (jacobians)
        //         {
        //             // Q = RP + t, L = [Lx ; Ly], Lx = Qx / Qz, Ly = Qy / Qz
        //             Eigen::Matrix<double, 2, 3, Eigen::RowMajor> dQ;
        //             // x = X / Z -> dx/dX = 1/Z, dx/dY = 0, dx/dZ = -X / Z^2;
        //             dQ.data()[0] = 1 / ptr[2];
        //             dQ.data()[1] = 0;
        //             dQ.data()[2] = -ptr[0] / ptr[2] / ptr[2];
        //             // y = Y / Z -> dy/dX = 0, dy/dY = 1/Z, dy/dZ = -Y / Z^2;
        //             dQ.data()[3] = 0;
        //             dQ.data()[4] = 1 / ptr[2];
        //             dQ.data()[5] = -ptr[1] / ptr[2] / ptr[2];

        //             if (solveExt)
        //             {
        //                 if (jacobians[0])   // Jacobian of output [x, y] w.r.t. input [angle axis, translation]
        //                 {
        //                     Eigen::Map< Eigen::Matrix<double, 2, 6, Eigen::RowMajor> > dRt(jacobians[0]);
        //                     // dt
        //                     dRt.block<2, 3>(0, 3) = dQ;
        //                     // dL/dR = dL/dQ * dQ/dR * dR/d(\theta)
        //                     Eigen::Matrix<double, 9, 3, Eigen::RowMajor> dRdtheta;
        //                     AngleAxisToRotationMatrixDerivative(parameters[0], dRdtheta.data());
        //                     // switch from column major (R) to row major
        //                     Eigen::Matrix<double, 1, 3> tmp = dRdtheta.row(1);
        //                     dRdtheta.row(1) = dRdtheta.row(3);
        //                     dRdtheta.row(3) = tmp;
        //                     tmp = dRdtheta.row(2);
        //                     dRdtheta.row(2) = dRdtheta.row(6);
        //                     dRdtheta.row(6) = tmp;
        //                     tmp = dRdtheta.row(5);
        //                     dRdtheta.row(5) = dRdtheta.row(7);
        //                     dRdtheta.row(7) = tmp;
        //                     Eigen::Matrix<double, 3, 3, Eigen::RowMajor> dQdtheta;
        //                     SparseProductDerivative(
        //                         dRdtheta.data(), parameters[1], std::vector<int>(1, 0), dQdtheta.data());
        //                     dRt.block<2, 3>(0, 0) = dQ * dQdtheta;
        //                 }
        //                 if (jacobians[1])   // Jacobian of output [x, y] w.r.t input [X, Y, Z]
        //                 {
        //                     // dL/dP = dL/dQ * dQ/dP = dL/dQ * R
        //                     Eigen::Matrix<double, 3, 3> R;
        //                     ceres::AngleAxisToRotationMatrix(parameters[0], R.data());
        //                     Eigen::Map< Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > dP(jacobians[1]);
        //                     dP = dQ * R;
        //                 }
        //             }
        //             else
        //             {
        //                 if (jacobians[0])   // Jacobian of output [x, y] w.r.t input [X, Y, Z]
        //                     std::copy(dQ.data(), dQ.data() + 6, jacobians[0]);
        //             }
        //         }
        //         return true;
        //     }
        // private:
        //     const cv::Point2f& pt2d;
        //     cv::Point2f pt2dCalibrated;
        //     const cv::Mat& intrinsics;
        //     const bool solveExt;
        // };

        void runBundleAdjustment(
            std::vector<cv::Mat>& refinedExtrinsics, Eigen::Matrix<double, 3, Eigen::Dynamic>& points3D,
            const std::vector<std::vector<cv::Point2f>>& points2DVectorsExtrinsic, const Eigen::MatrixXd& BAValid,
            const std::vector<cv::Mat>& cameraIntrinsics, const int numberCameras)
        {
            try
            {
                // Sanity check
                auto normCam0Identity = cv::norm(
                    refinedExtrinsics[0] - cv::Mat::eye(3, 4, refinedExtrinsics[0].type()));
                if (normCam0Identity > 1e-9) // std::cout prints exactly 0
                    error("Camera 0 should be [I, 0] for this algorithm to run. Norm cam0-[I,0] = "
                          + std::to_string(normCam0Identity), __LINE__, __FUNCTION__, __FILE__);
                // Prepare the camera extrinsics
                Eigen::Matrix<double, 6, Eigen::Dynamic> cameraRt(6, numberCameras); // Angle axis + translation
                for (auto cameraIndex = 0 ; cameraIndex < numberCameras ; cameraIndex++)
                {
                    cameraRt.data()[6 * cameraIndex + 3] = refinedExtrinsics[cameraIndex].at<double>(0, 3);
                    cameraRt.data()[6 * cameraIndex + 4] = refinedExtrinsics[cameraIndex].at<double>(1, 3);
                    cameraRt.data()[6 * cameraIndex + 5] = refinedExtrinsics[cameraIndex].at<double>(2, 3);
                    Eigen::Matrix<double, 3, 3> rotation;   // Column major!
                    for (auto x = 0; x < 3; x++)
                        for (auto y = 0; y < 3; y++)
                            rotation(x, y) = refinedExtrinsics[cameraIndex].at<double>(x, y);
                    ceres::RotationMatrixToAngleAxis(rotation.data(), cameraRt.data() + 6 * cameraIndex);
                }
                ceres::Problem problem;
                ceres::Solver::Options options;
                // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                // options.linear_solver_type = ceres::DENSE_QR;
                options.linear_solver_type = ceres::DENSE_SCHUR;
                options.use_nonmonotonic_steps = true;
                options.minimizer_progress_to_stdout = true;
                options.num_threads = 1;
                // // Option 1/3) Computing things together
                // const int numResiduals = 2 * BAValid.sum();  // x and y
                // BundleAdjustmentCost* ptr_BA = new BundleAdjustmentCost(
                //     points2DVectorsExtrinsic, cameraIntrinsics, BAValid);
                // ceres::DynamicAutoDiffCostFunction<BundleAdjustmentCost>* costFunction
                //     = new ceres::DynamicAutoDiffCostFunction<BundleAdjustmentCost>(ptr_BA);
                // costFunction->AddParameterBlock(6 * (numberCameras - 1));  // R + t
                // costFunction->AddParameterBlock(3 * points2DVectorsExtrinsic[0].size());
                // costFunction->SetNumResiduals(numResiduals);
                // problem.AddResidualBlock(
                //     costFunction, new ceres::HuberLoss(2.0), cameraRt.data() + 6, points3D.data());
                // Option 2/3) Computing things separately (automatic differentiation)
                for (auto cameraIndex = 0; cameraIndex < numberCameras; cameraIndex++)
                {
                    if (cameraIndex != 0u)
                    {
                        for (auto i = 0u; i < points2DVectorsExtrinsic[cameraIndex].size(); i++)
                        {
                            if (!BAValid(cameraIndex, i)) continue;
                            BundleAdjustmentUnit* ptr_BA = new BundleAdjustmentUnit(
                                points2DVectorsExtrinsic[cameraIndex][i], cameraIntrinsics[cameraIndex]);
                            auto* costFunction = new ceres::AutoDiffCostFunction<BundleAdjustmentUnit, 2, 6, 3>(
                                ptr_BA);
                            problem.AddResidualBlock(
                                costFunction, new ceres::HuberLoss(2.0), cameraRt.data() + 6 * cameraIndex,
                                points3D.data() + 3 * i);
                        }
                    }
                    else
                    {
                        for (auto i = 0u; i < points2DVectorsExtrinsic[cameraIndex].size(); i++)
                        {
                            if (!BAValid(cameraIndex, i)) continue;
                            BundleAdjustmentUnit* ptr_BA = new BundleAdjustmentUnit(
                                points2DVectorsExtrinsic[cameraIndex][i], cameraIntrinsics[cameraIndex]);
                            auto* costFunction = new ceres::AutoDiffCostFunction<BundleAdjustmentUnit, 2, 3>(ptr_BA);
                            problem.AddResidualBlock(costFunction, new ceres::HuberLoss(2.0), points3D.data() + 3 * i);
                        }
                    }
                }
                // No need to delete ptr_BA or costFunction; Ceres::Problem takes care of them.
                // // Option 3/3) Computing things separately (manual differentiation)
                // for (auto cameraIndex = 0; cameraIndex < numberCameras; cameraIndex++)
                // {
                //     if (cameraIndex != 0u)
                //         for (auto i = 0u; i < points2DVectorsExtrinsic[cameraIndex].size(); i++)
                //         {
                //             if (!BAValid(cameraIndex, i)) continue;
                //             ceres::CostFunction* costFunction = new BundleAdjustmentUnitJacobian(
                //                 points2DVectorsExtrinsic[cameraIndex][i], cameraIntrinsics[cameraIndex], true);
                //             problem.AddResidualBlock(
                //                 costFunction, new ceres::HuberLoss(2.0), cameraRt.data() + 6 * cameraIndex,
                //                 points3D.data() + 3 * i);
                //         }
                //     else
                //         for (auto i = 0u; i < points2DVectorsExtrinsic[cameraIndex].size(); i++)
                //         {
                //             if (!BAValid(cameraIndex, i)) continue;
                //             ceres::CostFunction* costFunction = new BundleAdjustmentUnitJacobian(
                //                 points2DVectorsExtrinsic[cameraIndex][i], cameraIntrinsics[cameraIndex], false);
                //             problem.AddResidualBlock(costFunction, new ceres::HuberLoss(2.0), points3D.data() + 3 * i);
                //         }
                // }
                // Ceres verbose
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                opLog(summary.FullReport(), Priority::High);
                // Sanity check
                normCam0Identity = cv::norm(
                    refinedExtrinsics[0] - cv::Mat::eye(3, 4, refinedExtrinsics[0].type()));
                if (normCam0Identity > 1e-9) // std::cout prints exactly 0
                    error("Camera 0 should not be modified by our implementation of bundle adjustment, notify us!"
                          " Norm: " + std::to_string(normCam0Identity), __LINE__, __FUNCTION__, __FILE__);
                // The first one should be [I | 0] and it is not modified by our Ceres optimization
                for (auto cameraIndex = 1 ; cameraIndex < numberCameras ; cameraIndex++)
                {
                    cv::Mat extrinsics(3, 4, refinedExtrinsics[0].type());
                    extrinsics.at<double>(0, 3) = cameraRt.data()[6 * cameraIndex + 3];
                    extrinsics.at<double>(1, 3) = cameraRt.data()[6 * cameraIndex + 4];
                    extrinsics.at<double>(2, 3) = cameraRt.data()[6 * cameraIndex + 5];
                    Eigen::Matrix<double, 3, 3> rotation;
                    ceres::AngleAxisToRotationMatrix(cameraRt.data() + 6 * cameraIndex, rotation.data());
                    for (auto x = 0; x < 3; x++)
                        for (auto y = 0; y < 3; y++)
                            extrinsics.at<double>(x, y) = rotation(x, y);
                    refinedExtrinsics[cameraIndex] = extrinsics;
                }
                const auto reprojectionError = computeReprojectionErrorInPixels(
                    points2DVectorsExtrinsic, BAValid, points3D, refinedExtrinsics, cameraIntrinsics);
                opLog("Reprojection Error (after Bundle Adjustment): " + std::to_string(reprojectionError)
                    + " pixels.", Priority::High);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void rescaleExtrinsicsAndPoints3D(
            std::vector<cv::Mat>& refinedExtrinsics, Eigen::Matrix<double, 3, Eigen::Dynamic>& points3D,
            const std::vector<std::vector<cv::Point2f>>& points2DVectorsExtrinsic, const Eigen::MatrixXd& BAValid,
            const std::vector<cv::Mat>& cameraIntrinsics, const int numberCameras, const int numberCorners,
            const float gridSquareSizeMm, const Point<int>& gridInnerCorners)
        {
            try
            {
                // Rescale the 3D points and translation based on the grid size
                if (points2DVectorsExtrinsic[0].size() % numberCorners != 0)
                    error("The number of points should be divided by number of corners in the image.",
                          __LINE__, __FUNCTION__, __FILE__);
                const int numTimeStep = points2DVectorsExtrinsic[0].size() / numberCorners;
                double sumLength = 0.;
                double sumSquareLength = 0.;
                double maxLength = -1.;
                double minLength = std::numeric_limits<double>::max();
                for (auto t = 0; t < numTimeStep; t++)
                {
                    // Horizontal edges
                    for (auto x = 0; x < gridInnerCorners.x - 1; x++)
                        for (auto y = 0; y < gridInnerCorners.y; y++)
                        {
                            const int startPerFrame = x + y * gridInnerCorners.x;
                            const int startIndex = startPerFrame + t * numberCorners;
                            const int endPerFrame = x + 1 + y * gridInnerCorners.x;
                            const int endIndex = endPerFrame + t * numberCorners;
                            // These points are used for BA, must have been constructed.
                            if (BAValid.col(startIndex).any() && BAValid.col(endIndex).any())
                            {
                                const double length = (points3D.col(startIndex) - points3D.col(endIndex)).norm();
                                sumSquareLength += length * length;
                                sumLength += length;
                                if (length < minLength)
                                    minLength = length;
                                if (length > maxLength)
                                    maxLength = length;
                            }
                        }

                    // Vertical edges
                    for (auto x = 0; x < gridInnerCorners.x; x++)
                        for (auto y = 0; y < gridInnerCorners.y - 1; y++)
                        {
                            const int startPerFrame = x + y * gridInnerCorners.x;
                            const int startIndex = startPerFrame + t * numberCorners;
                            const int endPerFrame = x + (y + 1) * gridInnerCorners.x;
                            const int endIndex = endPerFrame + t * numberCorners;
                            // These points are used for BA, must have been constructed.
                            if (BAValid.col(startIndex).any() && BAValid.col(endIndex).any())
                            {
                                const double length = (points3D.col(startIndex) - points3D.col(endIndex)).norm();
                                sumSquareLength += length * length;
                                sumLength += length;
                                if (length < minLength)
                                    minLength = length;
                                if (length > maxLength)
                                    maxLength = length;
                            }
                        }
                }
                const double scalingFactor = 0.001f * gridSquareSizeMm * sumLength / sumSquareLength;
                opLog("Scaling factor: " + std::to_string(scalingFactor) + ",\tMin grid length: "
                    + std::to_string(minLength) + ",\tMax grid length: " + std::to_string(maxLength), Priority::High);
                // Scale extrinsics: Scale the translation (and the 3D point)
                for (auto cameraIndex = 1; cameraIndex < numberCameras; cameraIndex++)
                {
                    refinedExtrinsics[cameraIndex].at<double>(0, 3) *= scalingFactor;
                    refinedExtrinsics[cameraIndex].at<double>(1, 3) *= scalingFactor;
                    refinedExtrinsics[cameraIndex].at<double>(2, 3) *= scalingFactor;
                }
                // Scale 3D points
                points3D *= scalingFactor;
                // Final reprojection error
                const auto reprojectionError = computeReprojectionErrorInPixels(
                    points2DVectorsExtrinsic, BAValid, points3D, refinedExtrinsics, cameraIntrinsics);
                opLog("Reprojection Error (after rescaling): " + std::to_string(reprojectionError) + " pixels.",
                    Priority::High);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void runBundleAdjustmentWithOutlierRemoval(
            std::vector<cv::Mat>& refinedExtrinsics, Eigen::Matrix<double, 3, Eigen::Dynamic>& points3D,
            std::vector<std::vector<cv::Point2f>>& points2DVectorsExtrinsic, Eigen::MatrixXd& BAValid,
            const std::vector<cv::Mat>& cameraIntrinsics, const int numberCameras, const double pixelThreshold,
            const bool printInitialReprojection)
        {
            try
            {
                // Update inliers & outliers
                BAValid = getInlierAndOutliers(points2DVectorsExtrinsic);

                // Initial reprojection error
                if (printInitialReprojection)
                {
                    const auto reprojectionError = computeReprojectionErrorInPixels(
                        points2DVectorsExtrinsic, BAValid, points3D, refinedExtrinsics, cameraIntrinsics);
                    opLog("Reprojection Error (initial): " + std::to_string(reprojectionError), Priority::High);
                    opLog(" ", Priority::High);
                }

                // Outlier removal
                opLog("Applying outlier removal...", Priority::High);
                removeOutliersReprojectionErrorIterative(
                    points2DVectorsExtrinsic, BAValid, points3D, refinedExtrinsics, cameraIntrinsics, pixelThreshold);
                opLog(" ", Priority::High);

                // Bundle Adjustment
                opLog("Running bundle adjustment...", Priority::High);
                runBundleAdjustment(
                    refinedExtrinsics, points3D, points2DVectorsExtrinsic, BAValid, cameraIntrinsics, numberCameras);
                opLog(" ", Priority::High);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void cameraXAsOrigin(
            std::vector<cv::Mat>& cameraExtrinsics, cv::Mat& cameraOriginInv, const cv::Mat& cameraOrigin)
        {
            try
            {
                // Extrinsics = cameraOrigin will turn into [I | 0]
                // All the others are transformed accordingly by multiplying them by inv(cameraOrigin)
                cameraOriginInv = cv::Mat::eye(4, 4, cameraOrigin.type());
                for (auto i = 0 ; i < 3 ; i++)
                    for (auto j = 0 ; j < 4 ; j++)
                        cameraOriginInv.at<double>(i, j) = cameraOrigin.at<double>(i, j);
                cameraOriginInv = cameraOriginInv.inv();
                for (auto& cameraExtrinsic : cameraExtrinsics)
                    cameraExtrinsic = cameraExtrinsic * cameraOriginInv;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }
    #endif

    void refineAndSaveExtrinsics(
        const std::string& parameterFolder, const std::string& imageFolder, const Point<int>& gridInnerCorners,
        const float gridSquareSizeMm, const int numberCameras, const bool imagesAreUndistorted,
        const bool saveImagesWithCorners)
    {
        try
        {
            #if defined(USE_CERES) && defined(USE_EIGEN)
                // Debugging
                const auto saveVisualSFMFiles = false;
                // const auto saveVisualSFMFiles = true;

                // Sanity check
                if (!imagesAreUndistorted)
                    error("This mode assumes that the images are already undistorted (add flag `--omit_distortion`).",
                          __LINE__, __FUNCTION__, __FILE__);

                opLog("Loading images...", Priority::High);
                const auto imageAndPaths = getImageAndPaths(imageFolder);
                opLog("Images loaded.", Priority::High);

                // Point<int> --> cv::Size
                const cv::Size gridInnerCornersCvSize{gridInnerCorners.x, gridInnerCorners.y};

                // Load parameters (distortion, intrinsics, initial extrinsics)
                opLog("Loading parameters...", Priority::High);
                CameraParameterReader cameraParameterReader;
                cameraParameterReader.readParameters(parameterFolder);
                const auto opCameraExtrinsicsInitial = cameraParameterReader.getCameraExtrinsicsInitial();
                // Sanity check
                if (opCameraExtrinsicsInitial.empty())
                    error("Camera intrinsics could not be loaded from " + parameterFolder
                        + ". Are they in the right path? Remember than the XML must contain the right intrinsic"
                        + " parameters before using this function. ", __LINE__, __FUNCTION__, __FILE__);
                bool initialEmpty = false;
                for (const auto& cameraExtrinsicInitial : opCameraExtrinsicsInitial)
                {
                    if (cameraExtrinsicInitial.empty())
                    {
                        initialEmpty = true;
                        break;
                    }
                }
                opLog("Parameters loaded.", Priority::High);
                // Camera extrinsics
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                auto opCameraExtrinsics = (initialEmpty
                    ? cameraParameterReader.getCameraExtrinsics() : opCameraExtrinsicsInitial);
                OP_OP2CVVECTORMAT(cameraExtrinsics, opCameraExtrinsics)
                // The first one should be [I | 0]: Multiply them all by inv(camera 0 extrinsics)
                cv::Mat cameraOriginInv;
                cameraXAsOrigin(cameraExtrinsics, cameraOriginInv, cameraExtrinsics.at(0).clone());
                // Camera intrinsics and distortion
                const auto opCameraIntrinsics = cameraParameterReader.getCameraIntrinsics();
                OP_OP2CVVECTORMAT(cameraIntrinsics, opCameraIntrinsics);
                const auto cameraDistortions = (
                    imagesAreUndistorted
                    ? std::vector<Matrix>{cameraIntrinsics.size()} : cameraParameterReader.getCameraDistortions());
                // Read images in folder
                opLog("Reading images in folder...", Priority::High);
                const auto numberCorners = gridInnerCorners.area();
                std::vector<std::vector<cv::Point2f>> points2DVectorsExtrinsic(numberCameras); // camera - keypoints
                std::vector<std::vector<unsigned int>> matchIndexes(numberCameras); // camera - indixes found
                if (imageAndPaths.empty())
                    error("imageAndPaths.empty()!.", __LINE__, __FUNCTION__, __FILE__);
                opLog("Images read.", Priority::High);
                // Get 2D grid corners of each image
                std::vector<cv::Mat> imagesWithCorners;
                const auto imageSize = imageAndPaths.at(0).first.size();
                const auto numberViews = (unsigned int)(imageAndPaths.size() / numberCameras);
                opLog("Processing cameras...", Priority::High);
                std::vector<std::thread> threads;
                for (auto cameraIndex = 0 ; cameraIndex < numberCameras ; cameraIndex++)
                {
                    auto* points2DExtrinsic = &points2DVectorsExtrinsic[cameraIndex];
                    auto* matchIndexesCamera = &matchIndexes[cameraIndex];
                    // Threaded version
                    threads.emplace_back(estimateAndSaveSiftFileSubThread, points2DExtrinsic,
                                         matchIndexesCamera, cameraIndex, numberCameras,
                                         numberCorners, numberViews, saveImagesWithCorners, imageFolder,
                                         gridInnerCornersCvSize, imageSize, imageAndPaths, saveVisualSFMFiles);
                    // // Non-threaded version
                    // estimateAndSaveSiftFileSubThread(points2DExtrinsic, matchIndexesCamera, cameraIndex, numberCameras,
                    //                                  numberCorners, numberViews, saveImagesWithCorners, imageFolder,
                    //                                  gridInnerCornersCvSize, imageSize, imageAndPaths,
                    //                                  saveVisualSFMFiles);
                }
                // Threaded version
                for (auto& thread : threads)
                    if (thread.joinable())
                        thread.join();

                // Matching file
                if (saveVisualSFMFiles)
                {
                    std::ofstream ofstreamMatches{
                        getFileParentFolderPath(imageAndPaths.at(0).second) + "FeatureMatches.txt"};
                    for (auto cameraIndex = 0 ; cameraIndex < numberCameras ; cameraIndex++)
                    {
                        for (auto cameraIndex2 = cameraIndex+1 ; cameraIndex2 < numberCameras ; cameraIndex2++)
                        {
                            std::vector<unsigned int> matchIndexesIntersection;
                            std::set_intersection(matchIndexes[cameraIndex].begin(), matchIndexes[cameraIndex].end(),
                                                  matchIndexes[cameraIndex2].begin(), matchIndexes[cameraIndex2].end(),
                                                  std::back_inserter(matchIndexesIntersection));

                            ofstreamMatches << getFileNameFromCameraIndex(cameraIndex) << ".jpg"
                                            << " " << getFileNameFromCameraIndex(cameraIndex2) << ".jpg"
                            // ofstreamMatches << getFileNameAndExtension(imageAndPaths.at(cameraIndex).second)
                            //                 << " " << getFileNameAndExtension(imageAndPaths.at(cameraIndex2).second)
                                            << " " << matchIndexesIntersection.size() << "\n";
                            for (auto reps = 0 ; reps < 2 ; reps++)
                            {
                                for (auto i = 0u ; i < matchIndexesIntersection.size() ; i++)
                                    ofstreamMatches << matchIndexesIntersection[i] << " ";
                                ofstreamMatches << "\n";
                            }
                            ofstreamMatches << "\n";
                        }
                    }
                }
                // ofstreamMatches.close();
                opLog("Number points (i.e., timestamps) fully obtained: "
                    + std::to_string(points2DVectorsExtrinsic[0].size()), Priority::High);
                opLog("Number views (i.e., cameras) fully obtained: "
                    + std::to_string(points2DVectorsExtrinsic[0].size() / numberCorners), Priority::High);
                opLog(" ", Priority::High);

                // Sanity check
                for (auto i = 1 ; i < numberCameras ; i++)
                    if (points2DVectorsExtrinsic[i].size() != points2DVectorsExtrinsic[0].size())
                        error("Something went wrong. Notify us:"
                              " points2DVectorsExtrinsic[i].size() != points2DVectorsExtrinsic[0].size().",
                              __LINE__, __FUNCTION__, __FILE__);

                // Note:
                // Extrinsics for each camera: std::vector<cv::Mat> cameraExtrinsics (translation in meters)
                // Intrinsics for each camera: std::vector<cv::Mat> cameraIntrinsics
                // Distortions assumed to be 0 (for now...)
                // 3D coordinates: gridSquareSizeMm (in mm not meters!) is the size of each chessboard square side
                // 2D coordinates:
                //     - matchIndexes[cameraIndex] are the coordinates matched (so found) in camera cameraIndex.
                //     - matchIndexesIntersection shows you how to get the intersection of 2 pair of cameras.
                //     - points2DVectorsExtrinsic[cameraIndex] has the 2D coordinates of the chessboard for camera
                //       cameraIndex.
                // Please, do not make changes to the code above this line (unless you ask me first), given that this code
                // is the same than VisualSFM uses, so we can easily compare results with both of them. If you wanna
                // re-write the 2D matching format, just modify it or duplicate it, but do not remove or edit
                // `matchIndexesIntersection`.
                // Last note: For quick debugging, set saveVisualSFMFiles = true and check the generated FeatureMatches.txt
                // (note that *.sift files are actually in binary format, so quite hard to read.)

                opLog("Estimating initial 3D points...", Priority::High);
                // Run triangulation to obtain the initial 3D points
                const auto initialPoints3D = reconstruct3DPoints(
                    points2DVectorsExtrinsic, cameraIntrinsics, cameraExtrinsics, numberCameras, imageSize);

                auto refinedExtrinsics = cameraExtrinsics;
                auto points3D = initialPoints3D;
                Eigen::MatrixXd BAValid;

                // Update inliers & outliers + Outlier removal + Bundle Adjustment with 1.0 threshold
                runBundleAdjustmentWithOutlierRemoval(
                    refinedExtrinsics, points3D, points2DVectorsExtrinsic, BAValid, cameraIntrinsics,
                    numberCameras, 1.0, true);

                // Update inliers & outliers + Outlier removal + Bundle Adjustment with 0.5 threshold
                runBundleAdjustmentWithOutlierRemoval(
                    refinedExtrinsics, points3D, points2DVectorsExtrinsic, BAValid, cameraIntrinsics,
                    numberCameras, 0.5, false);

                // Rescale the 3D points and translation based on the grid size
                rescaleExtrinsicsAndPoints3D(
                    refinedExtrinsics, points3D, points2DVectorsExtrinsic, BAValid, cameraIntrinsics, numberCameras,
                    numberCorners, gridSquareSizeMm, gridInnerCorners);

                // Revert back to refinedExtrinsics[0] = cameraExtrinsics[0] (rather than [I,0])
                // Note: Given that inv([R,t;0,1]) is another [R',t';0,1], scaling is maintained
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                cv::Mat cameraOriginInv2;
                cameraXAsOrigin(refinedExtrinsics, cameraOriginInv2, cameraOriginInv);
                // Sanity check
                auto normCam0Identity = cv::norm(refinedExtrinsics[0] - cameraExtrinsics[0]);
                if (normCam0Identity > 1e-9) // std::cout prints exactly 0
                    error("Unexpected error, notify us. Norm difference: "
                          + std::to_string(normCam0Identity), __LINE__, __FUNCTION__, __FILE__);

                // Final projection matrix
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                opLog("\nFinal projection matrix w.r.t. global origin (meters):", Priority::High);
                for (auto cameraIndex = 0; cameraIndex < numberCameras; cameraIndex++)
                {
                    opLog("Camera " + std::to_string(cameraIndex) + ":", Priority::High);
                    opLog(refinedExtrinsics[cameraIndex], Priority::High);
                    // opLog("Initial camera " + std::to_string(cameraIndex) + ":", Priority::High);
                    // opLog(cameraExtrinsics[cameraIndex], Priority::High);
                    const auto normDifference = cv::norm(
                        refinedExtrinsics[cameraIndex] - cameraExtrinsics[cameraIndex]);
                    opLog("Norm difference w.r.t. original extrinsics: " + std::to_string(normDifference),
                        Priority::High);
                }

                // Save new extrinsics
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                const auto cameraSerialNumbers = cameraParameterReader.getCameraSerialNumbers();
                const auto opRealCameraDistortions = cameraParameterReader.getCameraDistortions();
                for (auto i = 0 ; i < numberCameras ; i++)
                {
                    CameraParameterReader cameraParameterReaderFinal{
                        cameraSerialNumbers.at(i),
                        OP_CV2OPCONSTMAT(cameraIntrinsics.at(i)),
                        opRealCameraDistortions.at(i),
                        OP_CV2OPCONSTMAT(refinedExtrinsics.at(i)),
                        (initialEmpty ? OP_CV2OPCONSTMAT(cameraExtrinsics.at(i)) : opCameraExtrinsicsInitial.at(i))};
                    cameraParameterReaderFinal.writeParameters(parameterFolder);
                }
                opLog(" ", Priority::High);
            #else
                UNUSED(parameterFolder);
                UNUSED(imageFolder);
                UNUSED(gridInnerCorners);
                UNUSED(gridSquareSizeMm);
                UNUSED(numberCameras);
                UNUSED(imagesAreUndistorted);
                UNUSED(saveImagesWithCorners);
                error("CMake flags `USE_CERES` and `USE_EIGEN` required when compiling OpenPose`.",
                      __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void estimateAndSaveSiftFile(const Point<int>& gridInnerCorners, const std::string& imageFolder,
                                 const int numberCameras, const bool saveImagesWithCorners)
    {
        try
        {
            opLog("Loading images...", Priority::High);
            const auto imageAndPaths = getImageAndPaths(imageFolder);
            opLog("Images loaded.", Priority::High);

            // Point<int> --> cv::Size
            const cv::Size gridInnerCornersCvSize{gridInnerCorners.x, gridInnerCorners.y};

            // Read images in folder
            const auto numberCorners = gridInnerCorners.area();
            std::vector<std::vector<cv::Point2f>> points2DVectorsExtrinsic(numberCameras); // camera - keypoints
            std::vector<std::vector<unsigned int>> matchIndexes(numberCameras); // camera - indixes found
            if (imageAndPaths.empty())
                error("imageAndPaths.empty()!.", __LINE__, __FUNCTION__, __FILE__);

            // Get 2D grid corners of each image
            std::vector<cv::Mat> imagesWithCorners;
            const auto imageSize = imageAndPaths.at(0).first.size();
            const auto numberViews = (unsigned int)(imageAndPaths.size() / numberCameras);
            opLog("Processing cameras...", Priority::High);
            std::vector<std::thread> threads;
            for (auto cameraIndex = 0 ; cameraIndex < numberCameras ; cameraIndex++)
            {
                auto* points2DExtrinsic = &points2DVectorsExtrinsic[cameraIndex];
                auto* matchIndexesCamera = &matchIndexes[cameraIndex];
                // Threaded version
                const auto saveSIFTFile = true;
                threads.emplace_back(estimateAndSaveSiftFileSubThread, points2DExtrinsic,
                                     matchIndexesCamera, cameraIndex, numberCameras,
                                     numberCorners, numberViews, saveImagesWithCorners, imageFolder,
                                     gridInnerCornersCvSize, imageSize, imageAndPaths, saveSIFTFile);
                // // Non-threaded version
                // estimateAndSaveSiftFileSubThread(points2DExtrinsic, matchIndexesCamera, cameraIndex, numberCameras,
                //                                  numberCorners, numberViews, saveImagesWithCorners, imageFolder,
                //                                  gridInnerCornersCvSize, imageSize, imageAndPaths, saveSIFTFile);
            }
            // Threaded version
            for (auto& thread : threads)
                if (thread.joinable())
                    thread.join();

            // Matching file
            std::ofstream ofstreamMatches{getFileParentFolderPath(imageAndPaths.at(0).second) + "FeatureMatches.txt"};
            for (auto cameraIndex = 0 ; cameraIndex < numberCameras ; cameraIndex++)
            {
                for (auto cameraIndex2 = cameraIndex+1 ; cameraIndex2 < numberCameras ; cameraIndex2++)
                {
                    std::vector<unsigned int> matchIndexesIntersection;
                    std::set_intersection(matchIndexes[cameraIndex].begin(), matchIndexes[cameraIndex].end(),
                                          matchIndexes[cameraIndex2].begin(), matchIndexes[cameraIndex2].end(),
                                          std::back_inserter(matchIndexesIntersection));

                    ofstreamMatches << getFileNameFromCameraIndex(cameraIndex) << ".jpg"
                                    << " " << getFileNameFromCameraIndex(cameraIndex2) << ".jpg"
                    // ofstreamMatches << getFileNameAndExtension(imageAndPaths.at(cameraIndex).second)
                    //                 << " " << getFileNameAndExtension(imageAndPaths.at(cameraIndex2).second)
                                    << " " << matchIndexesIntersection.size() << "\n";
                    for (auto reps = 0 ; reps < 2 ; reps++)
                    {
                        for (auto i = 0u ; i < matchIndexesIntersection.size() ; i++)
                            ofstreamMatches << matchIndexesIntersection[i] << " ";
                        ofstreamMatches << "\n";
                    }
                    ofstreamMatches << "\n";
                }
            }
            ofstreamMatches.close();
            opLog("Number points fully obtained: " + std::to_string(points2DVectorsExtrinsic[0].size()), Priority::High);
            opLog("Number views fully obtained: " + std::to_string(points2DVectorsExtrinsic[0].size() / numberCorners),
                Priority::High);
            // Sanity check
            for (auto i = 1 ; i < numberCameras ; i++)
                if (points2DVectorsExtrinsic[i].size() != points2DVectorsExtrinsic[0].size())
                    error("Something went wrong. Notify us.", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
