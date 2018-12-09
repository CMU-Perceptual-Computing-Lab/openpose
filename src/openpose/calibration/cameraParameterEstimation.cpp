#include <fstream>
#include <numeric> // std::accumulate
#include <opencv2/core/core.hpp>
#ifdef USE_EIGEN
    #include <Eigen/Dense>
    #include <opencv2/core/eigen.hpp>
#endif
#include <openpose/3d/cameraParameterReader.hpp>
#include <openpose/calibration/gridPatternFunctions.hpp>
#include <openpose/filestream/fileStream.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/calibration/cameraParameterEstimation.hpp>
#include <openpose/3d/poseTriangulation.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

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
            const std::vector<std::string> extensions{
                // Completely supported by OpenCV
                "bmp", "dib", "pbm", "pgm", "ppm", "sr", "ras",
                // Most of them supported by OpenCV
                "jpg", "jpeg", "png"};
            const auto imagePaths = getFilesOnDirectory(imageDirectoryPath, extensions);
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
            double reprojectionError;
            std::vector<double> perViewErrors(objects3DVectors.size());

            std::vector<cv::Point2f> points2DVectors2;
            unsigned long long totalPoints = 0;
            double totalErr = 0;

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

            reprojectionError = {std::sqrt(totalErr/totalPoints)};

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
            log("\nCalibrating camera (intrinsics) with points from " + std::to_string(points2DVectors.size())
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

            log("\nIntrinsics:", Priority::High);
            log("Re-projection error - cv::calibrateCamera vs. calcReprojectionErrors:\t" + std::to_string(rms)
                + " vs. " + std::to_string(totalAvgErr), Priority::High);
            log("Intrinsics_K:", Priority::High);
            log(intrinsics.cameraMatrix, Priority::High);
            log("Intrinsics_distCoeff:", Priority::High);
            log(intrinsics.distortionCoefficients, Priority::High);
            log(" ", Priority::High);

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
                    error("Variables `angles` is empty.", __LINE__, __FUNCTION__, __FILE__);

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
                    log("There are outliers in the angles.", Priority::High);
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
                        log("Outlies in the result. Something went wrong when estimating the average of different"
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
                // log("Solving 2D-3D correspondences (extrinsics)", Priority::High);
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
                reorderPoints(points2DVector, gridInnerCorners, Points2DOrigin::TopLeft);

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
                        log("getExtrinsicParameters(...), iteration with: " + cameraPaths[i], Priority::High);
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
                log("M_gb:", Priority::High);
                log(MGridToCam1, Priority::High);
                log("M_gf:", Priority::High);
                log(MGridToCam0, Priority::High);
                log("M_bf:", Priority::High);
                log(MCam1ToCam0, Priority::High);

                log("########## Secondary camera position w.r.t. main camera ##########", Priority::High);
                log("tCam1WrtCam0:", Priority::High);
                log(tCam1WrtCam0, Priority::High);
                log("RCam1WrtCam0:", Priority::High);
                log(RCam1WrtCam0, Priority::High);
                log("MCam0WrtCam1:", Priority::High);
                log((- RCam1WrtCam0.transpose() * tCam1WrtCam0), Priority::High);
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
            log("Cannot write on " + fileName, Priority::High);
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

                if (viewIndex % std::max(1, int(numberViews/6)) == 0)
                    log("Camera " + std::to_string(cameraIndex) + " - Image view "
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
                    reorderPoints(points2DVector, gridInnerCornersCvSize, Points2DOrigin::TopLeft);
                    for (auto i = 0 ; i < numberCorners ; i++)
                        matchIndexesCamera.emplace_back(viewIndex * numberCorners + i);
                }
                else
                {
                    points2DVector.clear();
                    points2DVector.resize(numberCorners, cv::Point2f{-1.f,-1.f});
                    log("Camera " + std::to_string(cameraIndex) + " - Image view "
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
                        saveImage(imagesWithCorners.at(i), finalPath);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }





    // Public functions
    void estimateAndSaveIntrinsics(
        const Point<int>& gridInnerCorners, const float gridSquareSizeMm, const int flags,
        const std::string& outputParameterFolder, const std::string& imageFolder, const std::string& serialNumber,
        const bool saveImagesWithCorners)
    {
        try
        {
            // Point<int> --> cv::Size
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            const cv::Size gridInnerCornersCvSize{gridInnerCorners.x, gridInnerCorners.y};

            // Read images in folder
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            const auto imageAndPaths = getImageAndPaths(imageFolder);

            // Get 2D grid corners of each image
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            std::vector<std::vector<cv::Point2f>> points2DVectors;
            std::vector<cv::Mat> imagesWithCorners;
            const auto imageSize = imageAndPaths.at(0).first.size();
            for (auto i = 0u ; i < imageAndPaths.size() ; i++)
            {
                log("\nImage " + std::to_string(i+1) + "/" + std::to_string(imageAndPaths.size()), Priority::High);
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
                    reorderPoints(points2DVector, gridInnerCornersCvSize, Points2DOrigin::TopLeft);
                    points2DVectors.emplace_back(points2DVector);
                }
                else
                    log("Chessboard not found in image " + imageAndPaths.at(i).second + ".", Priority::High);

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

            // Run calibration
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // objects3DVector is the same one for each image
            const std::vector<std::vector<cv::Point3f>> objects3DVectors(
                points2DVectors.size(), getObjects3DVector(gridInnerCornersCvSize, gridSquareSizeMm));
            const auto intrinsics = calcIntrinsicParameters(imageSize, points2DVectors, objects3DVectors, flags);

            // Save intrinsics/results
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            CameraParameterReader cameraParameterReader{
                serialNumber, intrinsics.cameraMatrix, intrinsics.distortionCoefficients};
            cameraParameterReader.writeParameters(outputParameterFolder);

            // Debugging (optional) - Save images with corners
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
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
                        saveImage(imagesWithCorners.at(i), finalPath);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void estimateAndSaveExtrinsics(const std::string& parameterFolder,
                                   const std::string& imageFolder,
                                   const Point<int>& gridInnerCorners,
                                   const float gridSquareSizeMm,
                                   const int index0,
                                   const int index1,
                                   const bool imagesAreUndistorted,
                                   const bool combineCam0Extrinsics)
    {
        try
        {
            #ifdef USE_EIGEN
                // For debugging
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                const auto coutResults = false;
                // const auto coutResults = true;
                const bool coutAndImshowVerbose = false;

                // Point<int> --> cv::Size
                const cv::Size gridInnerCornersCvSize{gridInnerCorners.x, gridInnerCorners.y};

                // Load intrinsic parameters
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                CameraParameterReader cameraParameterReader;
                cameraParameterReader.readParameters(parameterFolder);
                const auto cameraSerialNumbers = cameraParameterReader.getCameraSerialNumbers();
                const auto realCameraDistortions = cameraParameterReader.getCameraDistortions();
                auto cameraIntrinsicsSubset = cameraParameterReader.getCameraIntrinsics();
                auto cameraDistortionsSubset = (imagesAreUndistorted ?
                    std::vector<cv::Mat>{realCameraDistortions.size()}
                    : realCameraDistortions);
                // Only use the 2 desired ones
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                cameraIntrinsicsSubset = {cameraIntrinsicsSubset.at(index0), cameraIntrinsicsSubset.at(index1)};
                cameraDistortionsSubset = {cameraDistortionsSubset.at(index0), cameraDistortionsSubset.at(index1)};
                // Base extrinsics
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                cv::Mat extrinsicsCam0 = cv::Mat::eye(4, 4, realCameraDistortions.at(0).type());
                bool cam0IsOrigin = true;
                if (combineCam0Extrinsics)
                {
                    cameraParameterReader.getCameraExtrinsics().at(index0).copyTo(extrinsicsCam0(cv::Rect{0,0,4,3}));
                    cam0IsOrigin = cv::norm(extrinsicsCam0 - cv::Mat::eye(4, 4, extrinsicsCam0.type())) < 1e-9;
                }

                // Number cameras and image paths
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                const auto numberCameras = cameraParameterReader.getNumberCameras();
                log("\nDetected " + std::to_string(numberCameras) + " cameras from your XML files on:\n"
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
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                log("Calibrating camera " + cameraSerialNumbers.at(index1) + " with respect to camera "
                    + cameraSerialNumbers.at(index0) + "...", Priority::High);
                const auto numberViews = imagePaths.size() / numberCameras;
                auto counterValidImages = 0u;
                std::vector<Eigen::Matrix4d> MCam1ToCam0s;
                for (auto i = 0u ; i < imagePaths.size() ; i+=numberCameras)
                {
                    log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    const auto pathCam0 = imagePaths[i+index0];
                    const auto pathCam1 = imagePaths[i+index1];
                    if (coutResults || i/numberCameras % int(numberViews/10) == 0)
                        log("It " + std::to_string(i/numberCameras+1) + "/" + std::to_string(numberViews) + ": "
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
                            log("########## Extrinsic parameters extractor ##########", Priority::High);
                            log("R_gf", Priority::High);
                            log(RGridToMainCam0, Priority::High);
                            log("t_gf", Priority::High);
                            log(tGridToMainCam0, Priority::High);
                            log("R_gb", Priority::High);
                            log(RGridToMainCam1, Priority::High);
                            log("t_gb", Priority::High);
                            log(tGridToMainCam1, Priority::High);
                            log("\n", Priority::High);
                        }

                        // MCam1ToCam0 - Projection matrix estimator
                        if (coutAndImshowVerbose)
                            log("########## Projection Matrix from secondary camera to main camera ##########",
                                Priority::High);
                        MCam1ToCam0s.emplace_back(getMFromCam1ToCam0(RGridToMainCam0, tGridToMainCam0,
                                                                     RGridToMainCam1, tGridToMainCam1,
                                                                     coutAndImshowVerbose));
                        if (coutResults)
                        {
                            if (coutAndImshowVerbose)
                                log("M_bg:", Priority::High);
                            log(MCam1ToCam0s.back(), Priority::High);
                            log(" ", Priority::High);
                        }
                    }
                    else
                    {
                        if (coutResults)
                            log("Invalid frame (chessboard not found).", Priority::High);
                    }
                }
                log("Finished processing images.", Priority::High);

                // Pseudo RANSAC calibration
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                const auto MCam1ToCam0Noisy = getMAverage(MCam1ToCam0s);
                log("Estimated initial (noisy?) projection matrix.", Priority::High);
                auto MCam1ToCam0 = getMAverage(MCam1ToCam0s, MCam1ToCam0Noisy);
                while ((MCam1ToCam0 - getMAverage(MCam1ToCam0s, MCam1ToCam0)).norm() > 1e-3)
                {
                    if (coutResults)
                        log("Repeated robustness method...", Priority::High);
                    MCam1ToCam0 = getMAverage(MCam1ToCam0s, MCam1ToCam0);
                }
                log("Estimated robust projection matrix.", Priority::High);
                log("norm(M_robust-M_noisy): " + std::to_string((MCam1ToCam0Noisy - MCam1ToCam0).norm()),
                    Priority::High);



                // Show errors
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                if (coutAndImshowVerbose)
                {
                    log("\n-----------------------------------------------------------------------------------"
                        "-------------------\nErrors:", Priority::High);
                    // Errors
                    log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    for (auto i = 0u ; i < MCam1ToCam0s.size() ; i++)
                    {
                        log("tCam1WrtCam0:", Priority::High);
                        log(MCam1ToCam0s.at(i).block<3,1>(0,3).transpose(), Priority::High);
                    }
                    log(" ", Priority::High);

                    log("tCam1WrtCam0:", Priority::High);
                    log(MCam1ToCam0.block<3,1>(0,3).transpose(), Priority::High);
                    log(" ", Priority::High);

                    // Rotation matrix in degrees Rodrigues(InputArray src, OutputArray dst
                    log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    const auto rad2deg = 180 / PI;
                    for (auto i = 0u ; i < MCam1ToCam0s.size() ; i++)
                    {
                        Eigen::Matrix3d R_secondaryToMain = MCam1ToCam0s.at(i).block<3,3>(0,0);
                        log("rodrigues:", Priority::High);
                        log((getRodriguesVector(R_secondaryToMain).t() * rad2deg), Priority::High);
                    }
                    Eigen::Matrix3d R_secondaryToMain = MCam1ToCam0.block<3,3>(0,0);
                    log("rodrigues:", Priority::High);
                    log((getRodriguesVector(R_secondaryToMain).t() * rad2deg), Priority::High);
                }

                // Show final result
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                if (coutResults)
                {
                    log("\n\n\n---------------------------------------------------------------------------"
                        "---------------------------", Priority::High);
                    log(std::to_string(counterValidImages) + " valid images.", Priority::High);
                    log("Initial (noisy?) projection matrix:", Priority::High);
                    log(MCam1ToCam0Noisy, Priority::High);
                    log("\nFinal projection matrix (mm):", Priority::High);
                    log(MCam1ToCam0, Priority::High);
                    log(" ", Priority::High);
                }

                // mm --> m
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                MCam1ToCam0.block<3,1>(0,3) *= 1e-3;

                // Eigen --> cv::Mat
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                cv::Mat cvMatExtrinsics;
                Eigen::MatrixXd eigenExtrinsics = MCam1ToCam0.block<3,4>(0,0);
                cv::eigen2cv(eigenExtrinsics, cvMatExtrinsics);

                // pos_target_origen = pos_target_cam1 * pos_cam1_origin
                // Example: pos_31 = pos_32 * pos_21
                if (!cam0IsOrigin)
                    cvMatExtrinsics *= extrinsicsCam0;

                // Final projection matrix
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                log("\nFinal projection matrix w.r.t. global origin (m):", Priority::High);
                log(cvMatExtrinsics, Priority::High);
                log(" ", Priority::High);

                // Save result
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                CameraParameterReader camera2ParameterReader{
                    cameraSerialNumbers.at(index1),
                    cameraIntrinsicsSubset.at(1),
                    realCameraDistortions.at(index1),
                    cvMatExtrinsics};
                camera2ParameterReader.writeParameters(parameterFolder);

                // Let the rendered image to be displayed
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
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

    // defined by Donglai, compute the Jacobian of rotation matrix w.r.t angle axis (rotation matrix in column major order)
    void AngleAxisToRotationMatrixDerivative(const double* pose, double* dR_data, const int idj=0, const int numberColumns=3)
    {
        Eigen::Map< Eigen::Matrix<double, 9, Eigen::Dynamic, Eigen::RowMajor> > dR(dR_data, 9, numberColumns);
        std::fill(dR_data, dR_data + 9 * numberColumns, 0.0);
        const double theta2 = pose[0] * pose[0] + pose[1] * pose[1] + pose[2] * pose[2];
        if (theta2 > std::numeric_limits<double>::epsilon())
        {
            const double theta = sqrt(theta2);
            const double s = sin(theta);
            const double c = cos(theta);
            const Eigen::Map< const Eigen::Matrix<double, 3, 1> > u(pose);
            Eigen::VectorXd e(3);
            e[0] = pose[0] / theta; e[1] = pose[1] / theta; e[2] = pose[2] / theta;

            // dR / dtheta
            Eigen::Matrix<double, 9, 1> dRdth(9, 1);
            Eigen::Map< Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > dRdth_(dRdth.data());
            // skew symmetric
            dRdth_ << 0.0, -e[2], e[1],
                      e[2], 0.0, -e[0],
                      -e[1], e[0], 0.0;
            // dRdth_ = dRdth_ * c - Matrix<double, 3, 3>::Identity() * s + s * e * e.transpose();
            dRdth_ = - dRdth_ * c - Eigen::Matrix<double, 3, 3>::Identity() * s + s * e * e.transpose();

            // dR / de
            Eigen::Matrix<double, 9, 3, Eigen::RowMajor> dRde(9, 3);
            // d(ee^T) / de
            dRde <<
                2 * e[0], 0., 0.,
                e[1], e[0], 0.,
                e[2], 0., e[0],
                e[1], e[0], 0.,
                0., 2 * e[1], 0.,
                0., e[2], e[1],
                e[2], 0., e[0],
                0., e[2], e[1],
                0., 0., 2 * e[2];
            Eigen::Matrix<double, 9, 3, Eigen::RowMajor> dexde(9, 3);
            dexde <<
                0, 0, 0,
                0, 0, -1,
                0, 1, 0,
                0, 0, 1,
                0, 0, 0,
                -1, 0, 0,
                0, -1, 0,
                1, 0, 0,
                0, 0, 0;
            // dRde = dRde * (1. - c) + c * dexde;
            dRde = dRde * (1. - c) - s * dexde;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dedu = Eigen::Matrix<double, 3, 3>::Identity() / theta - u * u.transpose() / theta2 / theta;

            dR.block(0, 3 * idj, 9, 3) = dRdth * e.transpose() + dRde * dedu;
        }
        else
        {
            dR(1, 3 * idj + 2) = 1;
            dR(2, 3 * idj + 1) = -1;
            dR(3, 3 * idj + 2) = -1;
            dR(5, 3 * idj) = 1;
            dR(6, 3 * idj + 1) = 1;
            dR(7, 3 * idj) = -1;
        }
    }

    // compute the derivative w.r.t AB from dA
    void SparseProductDerivative(const double* const dA_data, const double* const B_data,
                             const std::vector<int>& parentIndexes, double* dAB_data, const int numberColumns=3)
    {
        // d(AB) = AdB + (dA)B
        // Sparse for loop form
        std::fill(dAB_data, dAB_data + 3 * numberColumns, 0.0);
        for (int r = 0; r < 3; r++)
        {
            const int baseIndex = 3*r;
            for (const auto& parentIndex : parentIndexes)
            {
                const auto parentOffset = 3*parentIndex;
                for (int subIndex = 0; subIndex < 3; subIndex++)
                {
                    const auto finalOffset = parentOffset + subIndex;
                    dAB_data[numberColumns*r + finalOffset] +=
                        B_data[0] * dA_data[numberColumns*baseIndex + finalOffset]
                        + B_data[1] * dA_data[numberColumns*(baseIndex+1) + finalOffset]
                        + B_data[2] * dA_data[numberColumns*(baseIndex+2) + finalOffset];
                }
            }
        }
    }

    // defined by Donglai
    struct bundleAdjustmentCost
    {
        bundleAdjustmentCost(const std::vector<std::vector<cv::Point2f>>& points2DVectorsExtrinsic, const std::vector<cv::Mat>& cameraIntrinsicsCvMat,
            const Eigen::MatrixXd& BAValid)
            :points2DVectorsExtrinsic(points2DVectorsExtrinsic), BAValid(BAValid)
        {
            numberCameras = cameraIntrinsicsCvMat.size();
            if (points2DVectorsExtrinsic.size() != numberCameras)
                error("#view of points != #camera intrinsics.", __LINE__, __FUNCTION__, __FILE__);
            numberPoints = points2DVectorsExtrinsic[0].size();
            numberProjection = BAValid.sum();
            for (auto cameraIndex = 0u; cameraIndex < numberCameras; cameraIndex++)
            {
                if (cameraIntrinsicsCvMat[cameraIndex].cols != 3 || cameraIntrinsicsCvMat[cameraIndex].rows != 3)
                    error("Intrinsics passed in are not 3 x 3.", __LINE__, __FUNCTION__, __FILE__);
                cameraIntrinsics.resize(numberCameras);
                for (auto x = 0; x < 3; x++)
                    for (auto y = 0; y < 3; y++)
                        cameraIntrinsics[cameraIndex](x, y) = cameraIntrinsicsCvMat[cameraIndex].at<double>(x, y);
            }
        }

        template <typename T>
        bool operator()(T const* const* parameters, T* residuals) const;

        const std::vector<std::vector<cv::Point2f>>& points2DVectorsExtrinsic;
        const Eigen::MatrixXd& BAValid;
        std::vector<Eigen::Matrix<double, 3, 3>> cameraIntrinsics;
        uint numberCameras, numberPoints, numberProjection;
    };

    // defined by Donglai
    template <typename T>
    bool bundleAdjustmentCost::operator()(T const* const* parameters, T* residuals) const
    {
        // cameraExtrinsics: angle axis + translation (6 x (#Cam - 1)), camera 0 is always [I | 0]
        // pt3D: 3D points (3 x #Points)
        const T* cameraExtrinsics = parameters[0];
        const T* pt3D = parameters[1];
        const Eigen::Map< const Eigen::Matrix<T, 3, Eigen::Dynamic> > pt3DWorld(pt3D, 3, numberPoints);
        uint countProjection = 0u;
        for (auto cameraIndex = 0u; cameraIndex < numberCameras; cameraIndex++)
        {
            Eigen::Matrix<T, 3, Eigen::Dynamic> pt3DCamera = pt3DWorld;
            if (cameraIndex > 0u)
            {
                const Eigen::Map< const Eigen::Matrix<T, 3, 1> > translation(cameraExtrinsics + 6 * (cameraIndex - 1) + 3);  // minus 1!
                Eigen::Matrix<T, 3, 3> rotation;
                ceres::AngleAxisToRotationMatrix(cameraExtrinsics + 6 * (cameraIndex - 1), rotation.data());  // minus 1!
                pt3DCamera = rotation * pt3DCamera;
                pt3DCamera.colwise() += translation;
            }
            const Eigen::Matrix<T, 3, Eigen::Dynamic> pt2DHomogeneous = cameraIntrinsics[cameraIndex].cast<T>() * pt3DCamera;
            const Eigen::Matrix<T, 1, Eigen::Dynamic> ptx = pt2DHomogeneous.row(0).cwiseQuotient(pt2DHomogeneous.row(2));
            const Eigen::Matrix<T, 1, Eigen::Dynamic> pty = pt2DHomogeneous.row(1).cwiseQuotient(pt2DHomogeneous.row(2));
            for (auto i = 0u; i < numberPoints; i++)
            {
                if (!BAValid(cameraIndex, i))   // no data for this point
                    continue;
                residuals[2 * countProjection + 0] = ptx(0, i) - T(points2DVectorsExtrinsic[cameraIndex][i].x);
                residuals[2 * countProjection + 1] = pty(0, i) - T(points2DVectorsExtrinsic[cameraIndex][i].y);
                countProjection++;
            }
        }
        // sanity check
        if (countProjection != numberProjection)
            error("Wrong number of constraints in bundle adjustment", __LINE__, __FUNCTION__, __FILE__);
        return true;
    }

    // defined by Donglai
    struct bundleAdjustmentUnit
    {
        bundleAdjustmentUnit(const cv::Point2f& pt2d, const cv::Mat& intrinsics): pt2d(pt2d), intrinsics(intrinsics)
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
            // camera (6): angle axis + translation 
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

    // defined by Donglai
    class bundleAdjustmentUnitJacobian: public ceres::CostFunction
    {
    public:
        bundleAdjustmentUnitJacobian(const cv::Point2f& pt2d, const cv::Mat& intrinsics, const bool solveExt): pt2d(pt2d), intrinsics(intrinsics), solveExt(solveExt)
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

            CostFunction::set_num_residuals(2);
            auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
            parameter_block_sizes->clear();
            if (solveExt) parameter_block_sizes->push_back(6); // camera extrinsics
            parameter_block_sizes->push_back(3);  // 3D points
        }

        virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
        {
            double P[3];
            const double* ptr = solveExt ? P : parameters[0];
            if (solveExt)
            {
                const double* camera = parameters[0];
                ceres::AngleAxisRotatePoint(camera, parameters[1], P);
                P[0] += camera[3]; P[1] += camera[4]; P[2] += camera[5];
            }

            residuals[0] = ptr[0] / ptr[2] - pt2dCalibrated.x;
            residuals[1] = ptr[1] / ptr[2] - pt2dCalibrated.y;

            if (jacobians)
            {
                // Q = RP + t, L = [Lx ; Ly], Lx = Qx / Qz, Ly = Qy / Qz
                Eigen::Matrix<double, 2, 3, Eigen::RowMajor> dQ;
                // x = X / Z -> dx/dX = 1/Z, dx/dY = 0, dx/dZ = -X / Z^2;
                dQ.data()[0] = 1 / ptr[2];
                dQ.data()[1] = 0;
                dQ.data()[2] = -ptr[0] / ptr[2] / ptr[2];
                // y = Y / Z -> dy/dX = 0, dy/dY = 1/Z, dy/dZ = -Y / Z^2;
                dQ.data()[3] = 0;
                dQ.data()[4] = 1 / ptr[2];
                dQ.data()[5] = -ptr[1] / ptr[2] / ptr[2];

                if (solveExt)
                {
                    if (jacobians[0])   // Jacobian of output [x, y] w.r.t. input [angle axis, translation]
                    {
                        Eigen::Map< Eigen::Matrix<double, 2, 6, Eigen::RowMajor> > dRt(jacobians[0]);
                        // dt
                        dRt.block<2, 3>(0, 3) = dQ;
                        // dL/dR = dL/dQ * dQ/dR * dR/d(\theta)
                        Eigen::Matrix<double, 9, 3, Eigen::RowMajor> dRdtheta;
                        AngleAxisToRotationMatrixDerivative(parameters[0], dRdtheta.data());
                        // switch from column major (R) to row major
                        Eigen::Matrix<double, 1, 3> tmp = dRdtheta.row(1);
                        dRdtheta.row(1) = dRdtheta.row(3);
                        dRdtheta.row(3) = tmp;
                        tmp = dRdtheta.row(2);
                        dRdtheta.row(2) = dRdtheta.row(6);
                        dRdtheta.row(6) = tmp;
                        tmp = dRdtheta.row(5);
                        dRdtheta.row(5) = dRdtheta.row(7);
                        dRdtheta.row(7) = tmp;
                        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> dQdtheta;
                        SparseProductDerivative(dRdtheta.data(), parameters[1], std::vector<int>(1, 0), dQdtheta.data());
                        dRt.block<2, 3>(0, 0) = dQ * dQdtheta;
                    }
                    if (jacobians[1])   // Jacobian of output [x, y] w.r.t input [X, Y, Z]
                    {
                        // dL/dP = dL/dQ * dQ/dP = dL/dQ * R
                        Eigen::Matrix<double, 3, 3> R;
                        ceres::AngleAxisToRotationMatrix(parameters[0], R.data());
                        Eigen::Map< Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > dP(jacobians[1]);
                        dP = dQ * R;
                    }
                }
                else
                {
                    if (jacobians[0])   // Jacobian of output [x, y] w.r.t input [X, Y, Z]
                        std::copy(dQ.data(), dQ.data() + 6, jacobians[0]);
                }
            }
            return true;
        }
    private:
        const cv::Point2f& pt2d;
        cv::Point2f pt2dCalibrated;
        const cv::Mat& intrinsics;
        const bool solveExt;
    };

    double computeReprojectionError(
        const std::vector<std::vector<cv::Point2f>>& points2DVectorsExtrinsic,
        const Eigen::Matrix<double, 3, Eigen::Dynamic>& points3D,
        const Eigen::MatrixXd& BAValid, const std::vector<cv::Mat> cameraExtrinsics,
        const std::vector<cv::Mat> cameraIntrinsics)
    {
        try
        {
            // compute the average reprojection error
            const unsigned int numberCameras = cameraIntrinsics.size();
            const unsigned int numberPoints = points2DVectorsExtrinsic[0].size();
            double sumError = 0;
            int sumPoint = 0;
            double maxError = 0;
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
                    const double KX = cameraMatrix.at<double>(0, 0) * point3d[0] + cameraMatrix.at<double>(0, 1) * point3d[1] + cameraMatrix.at<double>(0, 2) * point3d[2] + cameraMatrix.at<double>(0, 3);
                    const double KY = cameraMatrix.at<double>(1, 0) * point3d[0] + cameraMatrix.at<double>(1, 1) * point3d[1] + cameraMatrix.at<double>(1, 2) * point3d[2] + cameraMatrix.at<double>(1, 3);
                    const double KZ = cameraMatrix.at<double>(2, 0) * point3d[0] + cameraMatrix.at<double>(2, 1) * point3d[1] + cameraMatrix.at<double>(2, 2) * point3d[2] + cameraMatrix.at<double>(2, 3);
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
            std::cout << "\tReprojection Error info: Max error: " << maxError << ";\t in cam idx " << maxCamIdx
                      << " with pt idx: " << maxPtIdx
                      << " & pt 2D: " << points2DVectorsExtrinsic[maxCamIdx][maxPtIdx].x
                            << " " << points2DVectorsExtrinsic[maxCamIdx][maxPtIdx].y << std::endl;
            return sumError / sumPoint;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    void removeOutliersReprojectionError(
        std::vector<std::vector<cv::Point2f>>& points2DVectorsExtrinsic,
        Eigen::Matrix<double, 3, Eigen::Dynamic>& points3D,
        Eigen::MatrixXd& BAValid, const std::vector<cv::Mat> cameraExtrinsics,
        const std::vector<cv::Mat> cameraIntrinsics, const double errorThreshold = 10)
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
                    const double KX = cameraMatrix.at<double>(0, 0) * point3d[0] + cameraMatrix.at<double>(0, 1) * point3d[1] + cameraMatrix.at<double>(0, 2) * point3d[2] + cameraMatrix.at<double>(0, 3);
                    const double KY = cameraMatrix.at<double>(1, 0) * point3d[0] + cameraMatrix.at<double>(1, 1) * point3d[1] + cameraMatrix.at<double>(1, 2) * point3d[2] + cameraMatrix.at<double>(1, 3);
                    const double KZ = cameraMatrix.at<double>(2, 0) * point3d[0] + cameraMatrix.at<double>(2, 1) * point3d[1] + cameraMatrix.at<double>(2, 2) * point3d[2] + cameraMatrix.at<double>(2, 3);
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
                      + std::to_string(numberPoints) + " total points vs. " + std::to_string(indexesToRemove.size())
                      + " outliers).", __LINE__, __FUNCTION__, __FILE__);
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
            //         const auto& indexToRemove = (i < indexesToRemove.size() ? indexesToRemove[i] : numberPoints);
            //         while (counterRansac < indexToRemove && counterRansac < numberPoints)
            //         {
            //             // Fill 2D coordinate
            //             for (auto cameraIndex = 0u; cameraIndex < numberCameras; cameraIndex++)
            //                 points2DVectorsExtrinsicRansac[cameraIndex][counterRansac-i]
            //                     = points2DVectorsExtrinsic[cameraIndex][counterRansac];
            //                 // points2DVectorsExtrinsicRansac.at(cameraIndex).at(counterRansac-i) = points2DVectorsExtrinsic.at(cameraIndex).at(counterRansac);
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

    void refineAndSaveExtrinsics(
        const std::string& parameterFolder, const std::string& imageFolder, const Point<int>& gridInnerCorners,
        const float gridSquareSizeMm, const int numberCameras, const bool imagesAreUndistorted,
        const bool saveImagesWithCorners)
    {
        try
        {
            // Sanity check
            if (!imagesAreUndistorted)
                error("This mode assumes that the images are already undistorted (add flag `--omit_distortion`).", __LINE__, __FUNCTION__, __FILE__);

            log("Loading images...", Priority::High);
            const auto imageAndPaths = getImageAndPaths(imageFolder);
            log("Images loaded.", Priority::High);

            // Point<int> --> cv::Size
            const cv::Size gridInnerCornersCvSize{gridInnerCorners.x, gridInnerCorners.y};

            // Load intrinsic parameters
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            CameraParameterReader cameraParameterReader;
            cameraParameterReader.readParameters(parameterFolder);
            // const auto cameraSerialNumbers = cameraParameterReader.getCameraSerialNumbers();
            const auto cameraExtrinsics = cameraParameterReader.getCameraExtrinsics();
            const auto cameraIntrinsics = cameraParameterReader.getCameraIntrinsics();
            const auto cameraDistortions = (
                imagesAreUndistorted
                ? std::vector<cv::Mat>{cameraIntrinsics.size()} : cameraParameterReader.getCameraDistortions());

            // Read images in folder
            const auto numberCorners = gridInnerCorners.area();
            std::vector<std::vector<cv::Point2f>> points2DVectorsExtrinsic(numberCameras); // camera - keypoints
            std::vector<std::vector<unsigned int>> matchIndexes(numberCameras); // camera - indixes found
            if (imageAndPaths.empty())
                error("imageAndPaths.empty()!.", __LINE__, __FUNCTION__, __FILE__);

            // Debugging
            // const auto saveVisualSFMFiles = false;
            const auto saveVisualSFMFiles = true;
            // Get 2D grid corners of each image
            std::vector<cv::Mat> imagesWithCorners;
            const auto imageSize = imageAndPaths.at(0).first.size();
            const auto numberViews = imageAndPaths.size() / numberCameras;
            log("Processing cameras...", Priority::High);
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
            }
            // ofstreamMatches.close();
            log("Number points fully obtained: " + std::to_string(points2DVectorsExtrinsic[0].size()), Priority::High);
            log("Number views fully obtained: " + std::to_string(points2DVectorsExtrinsic[0].size() / numberCorners),
                Priority::High);
            // Sanity check
            for (auto i = 1 ; i < numberCameras ; i++)
                if (points2DVectorsExtrinsic[i].size() != points2DVectorsExtrinsic[0].size())
                    error("Something went wrong. Notify us.", __LINE__, __FUNCTION__, __FILE__);

            // Note:
            // Extrinsics for each camera: std::vector<cv::Mat> cameraExtrinsics (translation in meters)
            // Intrinsics for each camera: std::vector<cv::Mat> cameraIntrinsics
            // Distortions assumed to be 0 (for now...)
            // 3D coordinates: gridSquareSizeMm (in mm not meters!) is the size of each chessboard square side
            // 2D coordinates:
            //     - matchIndexes[cameraIndex] are the coordinates matched (so found) in camera cameraIndex.
            //     - matchIndexesIntersection shows you how to get the intersection of 2 pair of cameras.
            //     - points2DVectorsExtrinsic[cameraIndex] has the 2D coordinates of the chessboard for camera cameraIndex.
            // Please, do not make changes to the code above this line (unless you ask me first), given that this code
            // is the same than VisualSFM uses, so we can easily compare results with both of them. If you wanna
            // re-write the 2D matching format, just modify it or duplicate it, but do not remove or edit
            // `matchIndexesIntersection`.
            // Last note: For quick debugging, set saveVisualSFMFiles = true and check the generated FeatureMatches.txt
            // (note that *.sift files are actually in binary format, so quite hard to read.)
            log("3D square size:");
            log(gridSquareSizeMm); // Just temporary to avoid warning of unused variable

            // print out things
            for (auto i = 0u; i < cameraExtrinsics.size(); i++)
            {
                std::cout << i << "\n" << cameraExtrinsics[i] << std::endl;
            }
            for (auto i = 0u; i < cameraIntrinsics.size(); i++)
            {
                std::cout << i << "\n" << cameraIntrinsics[i] << std::endl;
            }
            for (auto i = 0u; i < points2DVectorsExtrinsic.size(); i++)
            {
                std::cout << i << "\n" << points2DVectorsExtrinsic[i].size() << std::endl;
            }

            std::cout << "---------------------------\n";

            // compute the initial camera matrices
            std::vector<cv::Mat> cameraMatrices(numberCameras);
            for (auto cameraIndex = 0 ; cameraIndex < numberCameras ; cameraIndex++)
                cameraMatrices[cameraIndex] = cameraIntrinsics[cameraIndex] * cameraExtrinsics[cameraIndex];
            // Run triangulation to obtain the initial 3D points
            Eigen::MatrixXd BAValid = Eigen::MatrixXd::Zero(numberCameras, points2DVectorsExtrinsic[0].size());  // this is a valid reprojection term
            Eigen::Matrix<double, 3, Eigen::Dynamic> initialPoints3D(3, points2DVectorsExtrinsic[0].size());
            initialPoints3D.setZero();
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
                        pointsOnEachCamera.emplace_back(cv::Point2d{point2D.x, point2D.y}); // cv::Point2f --> cv::Point2d
                    }
                }
                if (pointCameraMatrices.size() < 2u)  // if visible in one camera, no triangulation and not used in bundle adjustment.
                    continue;
                for (auto cameraIndex = 0 ; cameraIndex < numberCameras ; cameraIndex++)
                    if (points2DVectorsExtrinsic[cameraIndex][i].x >= 0)  // this 2D term is used for optimization
                        BAValid(cameraIndex, i) = 1;
                cv::Mat reconstructedPoint;
                const float reprojectionError = triangulateWithOptimization(
                    reconstructedPoint, pointCameraMatrices, pointsOnEachCamera, reprojectionMaxAcceptable);
                UNUSED(reprojectionError);
                initialPoints3D.data()[3 * i + 0] = reconstructedPoint.at<double>(0, 0) / reconstructedPoint.at<double>(3, 0);
                initialPoints3D.data()[3 * i + 1] = reconstructedPoint.at<double>(1, 0) / reconstructedPoint.at<double>(3, 0);
                initialPoints3D.data()[3 * i + 2] = reconstructedPoint.at<double>(2, 0) / reconstructedPoint.at<double>(3, 0);
            }
            // std::cout << "---------------------------------" << std::endl;
            // for (int x = 432; x < 432 + 54; x++)
            //     std::cout << points2DVectorsExtrinsic[0][x].x << " " << points2DVectorsExtrinsic[0][x].y << std::endl;
            // std::cout << "---------------------------------" << std::endl;
            // for (int x = 432; x < 432 + 54; x++)
            //     std::cout << points2DVectorsExtrinsic[1][x].x << " " << points2DVectorsExtrinsic[1][x].y << std::endl;
            // std::cout << "---------------------------------" << std::endl;
            // for (int x = 432; x < 432 + 54; x++)
            //     std::cout << points2DVectorsExtrinsic[2][x].x << " " << points2DVectorsExtrinsic[2][x].y << std::endl;
            // std::cout << "---------------------------------" << std::endl;
            // for (int x = 432; x < 432 + 54; x++)
            //     std::cout << points2DVectorsExtrinsic[3][x].x << " " << points2DVectorsExtrinsic[3][x].y << std::endl;
            auto reprojectionError = computeReprojectionError(
                points2DVectorsExtrinsic, initialPoints3D, BAValid, cameraExtrinsics, cameraIntrinsics);
            std::cout << "Reprojection Error (initial): " << reprojectionError << std::endl;

            // Outlier removal
            auto reprojectionErrorPrevious = reprojectionError+1;
            while (reprojectionError != reprojectionErrorPrevious)
            {
                reprojectionErrorPrevious = reprojectionError;
                // 10 pixels is a lot for full HD images...
                const auto errorThreshold = fastMax(2*reprojectionError, 1.);
                removeOutliersReprojectionError(
                    points2DVectorsExtrinsic, initialPoints3D, BAValid, cameraExtrinsics, cameraIntrinsics,
                    errorThreshold);
                reprojectionError = computeReprojectionError(
                    points2DVectorsExtrinsic, initialPoints3D, BAValid, cameraExtrinsics, cameraIntrinsics);
                std::cout << "Reprojection Error (after outlier removal iteration): " << reprojectionError
                          << ",\twith error threshold of " << errorThreshold << std::endl;
            }
            std::cout << "Reprojection Error (after outlier removal): " << computeReprojectionError(
                points2DVectorsExtrinsic, initialPoints3D, BAValid, cameraExtrinsics, cameraIntrinsics) << std::endl;
            std::cout << "Number of total 3D points " << initialPoints3D.size() << std::endl;

            // Start bundle adjustment.
            Eigen::Matrix<double, 3, Eigen::Dynamic> points3D = initialPoints3D;
            Eigen::Matrix<double, 6, Eigen::Dynamic> cameraRt(6, numberCameras);   // angle axis + translation
            // prepare the camera intrinsics
            for (auto cameraIndex = 0 ; cameraIndex < numberCameras ; cameraIndex++)
            {
                cameraRt.data()[6 * cameraIndex + 3] = cameraExtrinsics[cameraIndex].at<double>(0, 3);
                cameraRt.data()[6 * cameraIndex + 4] = cameraExtrinsics[cameraIndex].at<double>(1, 3);
                cameraRt.data()[6 * cameraIndex + 5] = cameraExtrinsics[cameraIndex].at<double>(2, 3);
                Eigen::Matrix<double, 3, 3> rotation;   // column major!
                for (auto x = 0; x < 3; x++)
                    for (auto y = 0; y < 3; y++)
                        rotation(x, y) = cameraExtrinsics[cameraIndex].at<double>(x, y);
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
            // computing things together
            /*
            const int numResiduals = 2 * BAValid.sum();  // x and y
            bundleAdjustmentCost* ptr_BA = new bundleAdjustmentCost(points2DVectorsExtrinsic, cameraIntrinsics, BAValid);
            ceres::DynamicAutoDiffCostFunction<bundleAdjustmentCost>* costFunction = new ceres::DynamicAutoDiffCostFunction<bundleAdjustmentCost>(ptr_BA);
            costFunction->AddParameterBlock(6 * (numberCameras - 1));  // R + t
            costFunction->AddParameterBlock(3 * points2DVectorsExtrinsic[0].size());
            costFunction->SetNumResiduals(numResiduals);
            problem.AddResidualBlock(costFunction, new ceres::HuberLoss(2.0), cameraRt.data() + 6, points3D.data());
            */

            // computing things separately  (automatic differentiation)
            for (auto cameraIndex = 0; cameraIndex < numberCameras; cameraIndex++)
            {
                if (cameraIndex != 0u)
                    for (auto i = 0u; i < points2DVectorsExtrinsic[cameraIndex].size(); i++)
                    {
                        if (!BAValid(cameraIndex, i)) continue;
                        bundleAdjustmentUnit* ptr_BA = new bundleAdjustmentUnit(points2DVectorsExtrinsic[cameraIndex][i], cameraIntrinsics[cameraIndex]);
                        ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<bundleAdjustmentUnit, 2, 6, 3>(ptr_BA);
                        problem.AddResidualBlock(costFunction, new ceres::HuberLoss(2.0), cameraRt.data() + 6 * cameraIndex, points3D.data() + 3 * i);
                    }
                else
                    for (auto i = 0u; i < points2DVectorsExtrinsic[cameraIndex].size(); i++)
                    {
                        if (!BAValid(cameraIndex, i)) continue;
                        bundleAdjustmentUnit* ptr_BA = new bundleAdjustmentUnit(points2DVectorsExtrinsic[cameraIndex][i], cameraIntrinsics[cameraIndex]);
                        ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<bundleAdjustmentUnit, 2, 3>(ptr_BA);
                        problem.AddResidualBlock(costFunction, new ceres::HuberLoss(2.0), points3D.data() + 3 * i);
                    }
            }

            // computing things separately (manual differentiation)
            // for (auto cameraIndex = 0; cameraIndex < numberCameras; cameraIndex++)
            // {
            //     if (cameraIndex != 0u)
            //         for (auto i = 0u; i < points2DVectorsExtrinsic[cameraIndex].size(); i++)
            //         {
            //             if (!BAValid(cameraIndex, i)) continue;
            //             ceres::CostFunction* costFunction = new bundleAdjustmentUnitJacobian(points2DVectorsExtrinsic[cameraIndex][i], cameraIntrinsics[cameraIndex], true);
            //             problem.AddResidualBlock(costFunction, new ceres::HuberLoss(2.0), cameraRt.data() + 6 * cameraIndex, points3D.data() + 3 * i);
            //         }
            //     else
            //         for (auto i = 0u; i < points2DVectorsExtrinsic[cameraIndex].size(); i++)
            //         {
            //             if (!BAValid(cameraIndex, i)) continue;
            //             ceres::CostFunction* costFunction = new bundleAdjustmentUnitJacobian(points2DVectorsExtrinsic[cameraIndex][i], cameraIntrinsics[cameraIndex], false);
            //             problem.AddResidualBlock(costFunction, new ceres::HuberLoss(2.0), points3D.data() + 3 * i);
            //         }
            // }
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.FullReport() << std::endl;

            std::vector<cv::Mat> refinedExtrinsics(numberCameras);
            refinedExtrinsics[0] = cameraExtrinsics[0];
            for (auto cameraIndex = 1; cameraIndex < numberCameras; cameraIndex++)   // the first one is always [I | 0]
            {
                cv::Mat ext(3, 4, CV_64FC1);
                ext.at<double>(0, 3) = cameraRt.data()[6 * cameraIndex + 3];
                ext.at<double>(1, 3) = cameraRt.data()[6 * cameraIndex + 4];
                ext.at<double>(2, 3) = cameraRt.data()[6 * cameraIndex + 5];
                Eigen::Matrix<double, 3, 3> rotation;
                ceres::AngleAxisToRotationMatrix(cameraRt.data() + 6 * cameraIndex, rotation.data());
                for (auto x = 0; x < 3; x++)
                    for (auto y = 0; y < 3; y++)
                        ext.at<double>(x, y) = rotation(x, y);
                refinedExtrinsics[cameraIndex] = ext;
            }
            std::cout << "Reprojection Error (after Bundle Adjustment): "
                      << computeReprojectionError(points2DVectorsExtrinsic, points3D, BAValid, refinedExtrinsics, cameraIntrinsics) << std::endl;
            // no need to delete ptr_BA or costFunction; Ceres::Problem takes care of them.

            // rescale the 3D points and translation based on the grid size
            if (points2DVectorsExtrinsic[0].size() % numberCorners != 0)
                error("The number of points should be divided by number of corners in the image.", __LINE__, __FUNCTION__, __FILE__);
            const int numTimeStep = points2DVectorsExtrinsic[0].size() / numberCorners;
            double sumLength = 0.;
            double sumSquareLength = 0.;
            double maxLength = -1;
            double minLength = std::numeric_limits<double>::max();
            for (auto t = 0; t < numTimeStep; t++)
            {
                // horizontal edges
                for (auto x = 0; x < gridInnerCorners.x - 1; x++)
                    for (auto y = 0; y < gridInnerCorners.y; y++)
                    {
                        const int startPerFrame = x + y * gridInnerCorners.x;
                        const int startIndex = startPerFrame + t * numberCorners;
                        const int endPerFrame = x + 1 + y * gridInnerCorners.x;
                        const int endIndex = endPerFrame + t * numberCorners;
                        if (BAValid.col(startIndex).any() && BAValid.col(endIndex).any())   // These points are used for BA, must have been constructed.
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

                // vertical edges
                for (auto x = 0; x < gridInnerCorners.x; x++)
                    for (auto y = 0; y < gridInnerCorners.y - 1; y++)
                    {
                        const int startPerFrame = x + y * gridInnerCorners.x;
                        const int startIndex = startPerFrame + t * numberCorners;
                        const int endPerFrame = x + (y + 1) * gridInnerCorners.x;
                        const int endIndex = endPerFrame + t * numberCorners;
                        if (BAValid.col(startIndex).any() && BAValid.col(endIndex).any())   // These points are used for BA, must have been constructed.
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
            const double scalingFactor = 0.001 * gridSquareSizeMm * sumLength / sumSquareLength;
            std::cout << "Max grid length: " << maxLength << std::endl << "Min grid length: " << minLength << std::endl;
            std::cout << "Scaling: " << scalingFactor << std::endl;

            for (auto cameraIndex = 1; cameraIndex < numberCameras; cameraIndex++) // scale the translation (and the 3D point)
            {
                refinedExtrinsics[cameraIndex].at<double>(0, 3) *= scalingFactor;
                refinedExtrinsics[cameraIndex].at<double>(1, 3) *= scalingFactor;
                refinedExtrinsics[cameraIndex].at<double>(2, 3) *= scalingFactor;
            }
            points3D *= scalingFactor;
            std::cout << "Reprojection Error (after rescaling): "
                      << computeReprojectionError(points2DVectorsExtrinsic, points3D, BAValid, refinedExtrinsics, cameraIntrinsics) << std::endl;

            std::cout << "Output: -----------------------------------" << std::endl;
            for (auto cameraIndex = 0; cameraIndex < numberCameras; cameraIndex++)
            {
                std::cout << cameraIndex << " ::::::" << std::endl;
                std::cout << refinedExtrinsics[cameraIndex] << std::endl;
            }
// Eigen::Matrix<double, 2, Eigen::Dynamic> residuals(2, int(BAValid.sum()));
// (*ptr_BA)(ptrParameters, residuals.data());
// std::cout << residuals << std::endl;
// delete ptr_BA;
// delete costFunction;
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
            log("Loading images...", Priority::High);
            const auto imageAndPaths = getImageAndPaths(imageFolder);
            log("Images loaded.", Priority::High);

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
            const auto numberViews = imageAndPaths.size() / numberCameras;
            log("Processing cameras...", Priority::High);
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
            log("Number points fully obtained: " + std::to_string(points2DVectorsExtrinsic[0].size()), Priority::High);
            log("Number views fully obtained: " + std::to_string(points2DVectorsExtrinsic[0].size() / numberCorners),
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
