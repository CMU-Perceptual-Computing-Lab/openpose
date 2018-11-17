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
#include <openpose/calibration/cameraParameterEstimation.hpp>

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

    std::pair<double, std::vector<double>> calcReprojectionErrors(const std::vector<std::vector<cv::Point3f>>& objects3DVectors,
                                                                  const std::vector<std::vector<cv::Point2f>>& points2DVectors,
                                                                  const std::vector<cv::Mat>& rVecs,
                                                                  const std::vector<cv::Mat>& tVecs,
                                                                  const Intrinsics& intrinsics)
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
                cv::projectPoints(cv::Mat(objects3DVectors.at(i)),
                                  rVecs.at(i),
                                  tVecs.at(i),
                                  intrinsics.cameraMatrix,
                                  intrinsics.distortionCoefficients,
                                  points2DVectors2);
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

    Intrinsics calcIntrinsicParameters(const cv::Size& imageSize,
                                       const std::vector<std::vector<cv::Point2f>>& points2DVectors,
                                       const std::vector<std::vector<cv::Point3f>>& objects3DVectors,
                                       const int calibrateCameraFlags)
    {
        try
        {
            log("\nCalibrating camera (intrinsics) with points from " + std::to_string(points2DVectors.size())
                + " images...", Priority::High);

            //Find intrinsic and extrinsic camera parameters
            Intrinsics intrinsics;
            std::vector<cv::Mat> rVecs;
            std::vector<cv::Mat> tVecs;
            const auto rms = cv::calibrateCamera(objects3DVectors, points2DVectors, imageSize, intrinsics.cameraMatrix,
                                                 intrinsics.distortionCoefficients, rVecs, tVecs,
                                                 calibrateCameraFlags);

            // cv::checkRange checks that every array element is neither NaN nor infinite
            const auto calibrationIsCorrect = cv::checkRange(intrinsics.cameraMatrix)
                                            && cv::checkRange(intrinsics.distortionCoefficients);
            if (!calibrationIsCorrect)
                error("Unvalid cameraMatrix and/or distortionCoefficients.", __LINE__, __FUNCTION__, __FILE__);

            double totalAvgErr;
            std::vector<double> reprojectionErrors;
            std::tie(totalAvgErr, reprojectionErrors) = calcReprojectionErrors(objects3DVectors, points2DVectors,
                                                                               rVecs, tVecs, intrinsics);

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
        std::pair<cv::Mat, cv::Mat> solveCorrespondences2D3D(const cv::Mat& cameraMatrix,
                                                             const cv::Mat& distortionCoefficients,
                                                             const std::vector<cv::Point3f>& objects3DVector,
                                                             const std::vector<cv::Point2f>& points2DVector)
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

        std::tuple<cv::Mat, cv::Mat, std::vector<cv::Point2f>, std::vector<cv::Point3f>> calcExtrinsicParametersOpenCV(const cv::Mat& image,
                                                                                                                       const cv::Mat& cameraMatrix,
                                                                                                                       const cv::Mat& distortionCoefficients,
                                                                                                                       const cv::Size& gridInnerCorners,
                                                                                                                       const float gridSquareSizeMm)
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

        std::tuple<bool, Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d> getExtrinsicParameters(const std::vector<std::string>& cameraPaths,
                                                                                                                    const cv::Size& gridInnerCorners,
                                                                                                                    const float gridSquareSizeMm,
                                                                                                                    const bool coutAndPlotGridCorners,
                                                                                                                    const std::vector<cv::Mat>& intrinsics,
                                                                                                                    const std::vector<cv::Mat>& distortions)
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
                    {
                        plotGridCorners(gridInnerCorners, extrinsicss[i].points2DVector,
                                        cameraPaths[i], image);
                    }
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
                return std::make_tuple(false, Eigen::Matrix3d{}, Eigen::Vector3d{}, Eigen::Matrix3d{}, Eigen::Vector3d{});
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

    void estimateAndSaveSiftFileSubThread(std::vector<cv::Point2f>* points2DExtrinsicPtr,
                                          std::vector<unsigned int>* matchIndexesCameraPtr,
                                          const int cameraIndex,
                                          const int numberCameras,
                                          const int numberCorners,
                                          const unsigned int numberViews,
                                          const bool saveImagesWithCorners,
                                          const std::string& imagesFolder,
                                          const cv::Size& gridInnerCornersCvSize,
                                          const cv::Size& imageSize,
                                          const std::vector<std::pair<cv::Mat, std::string>>& imageAndPaths)
    {
        try
        {
            // Sanity check
            if (points2DExtrinsicPtr == nullptr || matchIndexesCameraPtr == nullptr)
                error("Make sure than points2DExtrinsicPtr != nullptr && matchIndexesCameraPtr != nullptr.",
                      __LINE__, __FUNCTION__, __FILE__);
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
                    error("Detected images with different sizes in `" + imagesFolder + "` All images"
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
            // const auto fileName = getFullFilePathNoExtension(imageAndPaths.at(cameraIndex).second) + ".sift";
            const auto fileName = getFileParentFolderPath(imageAndPaths.at(cameraIndex).second)
                                + getFileNameFromCameraIndex(cameraIndex) + ".sift";
            writeVisualSFMSiftGPU(fileName, points2DExtrinsic);

            // Save images with corners
            if (saveImagesWithCorners)
            {
                const auto folderWhereSavingImages = imagesFolder + "images_with_corners/";
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
    void estimateAndSaveIntrinsics(const Point<int>& gridInnerCorners,
                                   const float gridSquareSizeMm,
                                   const int flags,
                                   const std::string& outputParameterFolder,
                                   const std::string& imagesFolder,
                                   const std::string& serialNumber,
                                   const bool saveImagesWithCorners)
    {
        try
        {
            // Point<int> --> cv::Size
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            const cv::Size gridInnerCornersCvSize{gridInnerCorners.x, gridInnerCorners.y};

            // Read images in folder
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            std::vector<std::vector<cv::Point2f>> points2DVectors;
            const auto imageAndPaths = getImageAndPaths(imagesFolder);

            // Get 2D grid corners of each image
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            std::vector<cv::Mat> imagesWithCorners;
            const auto imageSize = imageAndPaths.at(0).first.size();
            for (auto i = 0u ; i < imageAndPaths.size() ; i++)
            {
                log("\nImage " + std::to_string(i+1) + "/" + std::to_string(imageAndPaths.size()), Priority::High);
                const auto& image = imageAndPaths.at(i).first;

                // Sanity check
                if (imageSize.width != image.cols || imageSize.height != image.rows)
                    error("Detected images with different sizes in `" + imagesFolder + "` All images"
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
                    std::cerr << "Chessboard not found in this image." << std::endl;

                // Show image (with chessboard corners if found)
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
            const std::vector<std::vector<cv::Point3f>> objects3DVectors(points2DVectors.size(),
                                                                         getObjects3DVector(gridInnerCornersCvSize,
                                                                                            gridSquareSizeMm));
            const auto intrinsics = calcIntrinsicParameters(imageSize, points2DVectors, objects3DVectors, flags);

            // Save intrinsics/results
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            CameraParameterReader cameraParameterReader{serialNumber, intrinsics.cameraMatrix,
                                                        intrinsics.distortionCoefficients};
            cameraParameterReader.writeParameters(outputParameterFolder);

            // Save images with corners
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            if (saveImagesWithCorners)
            {
                const auto folderWhereSavingImages = imagesFolder + "images_with_corners/";
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

    void estimateAndSaveExtrinsics(const std::string& intrinsicsFolder,
                                   const std::string& extrinsicsImagesFolder,
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
                cameraParameterReader.readParameters(intrinsicsFolder);
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
                    + intrinsicsFolder + "\nRemove wrong/extra XML files if this number of cameras does not"
                    + " correspond with the number of cameras recorded in:\n" + extrinsicsImagesFolder + "\n",
                    Priority::High);
                const auto imagePaths = getImagePaths(extrinsicsImagesFolder);
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
                camera2ParameterReader.writeParameters(intrinsicsFolder);

                // Let the rendered image to be displayed
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                if (coutAndImshowVerbose)
                    cv::waitKey(0);
            #else
                UNUSED(intrinsicsFolder);
                UNUSED(extrinsicsImagesFolder);
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

    void estimateAndSaveSiftFile(const Point<int>& gridInnerCorners,
                                 const std::string& imagesFolder,
                                 const int numberCameras,
                                 const bool saveImagesWithCorners)
    {
        try
        {
            log("Loading images...", Priority::High);
            const auto imageAndPaths = getImageAndPaths(imagesFolder);
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
                threads.emplace_back(estimateAndSaveSiftFileSubThread, points2DExtrinsic,
                                     matchIndexesCamera, cameraIndex, numberCameras,
                                     numberCorners, numberViews, saveImagesWithCorners, imagesFolder,
                                     gridInnerCornersCvSize, imageSize, imageAndPaths);
                // // Non-threaded version
                // estimateAndSaveSiftFileSubThread(points2DExtrinsic, matchIndexesCamera, cameraIndex, numberCameras,
                //                                  numberCorners, numberViews, saveImagesWithCorners, imagesFolder,
                //                                  gridInnerCornersCvSize, imageSize, imageAndPaths);
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
