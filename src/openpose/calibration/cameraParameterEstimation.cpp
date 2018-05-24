#include <fstream>
#include <opencv2/core/core.hpp>
#include <openpose/3d/cameraParameterReader.hpp>
#include <openpose/calibration/gridPatternFunctions.hpp>
#include <openpose/filestream/fileStream.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/calibration/cameraParameterEstimation.hpp>

namespace op
{
    // Private functions
    struct Intrinsics
    {
        cv::Mat cameraMatrix;
        cv::Mat distortionCoefficients;

        Intrinsics() :
            cameraMatrix{cv::Mat::eye(3, 3, CV_64F)},
            distortionCoefficients{cv::Mat::zeros(14, 1, CV_64F)}
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

    std::vector<cv::Mat> getImages(const std::string& imageDirectoryPath)
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
            // Read images
            std::vector<cv::Mat> images;
            for (const auto& imagePath : imagePaths)
            {
                images.emplace_back(cv::imread(imagePath, CV_LOAD_IMAGE_COLOR));
                if (images.back().empty())
                    error("Image could not be opened from path `" + imagePath + "`.", __LINE__, __FUNCTION__, __FILE__);
            }
            // Return result
            return images;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::pair<double, std::vector<double>> computeReprojectionErrors(const std::vector<std::vector<cv::Point3f>>& objects3DVectors,
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

    std::tuple<Intrinsics, std::vector<double>, double> calcIntrinsicParameters(const cv::Size& imageSize,
                                                                                const std::vector<std::vector<cv::Point2f>>& points2DVectors,
                                                                                const std::vector<std::vector<cv::Point3f>>& objects3DVectors,
                                                                                const int calibrateCameraFlags)
    {
        try
        {

            std::cout << "\nCalibrating camera (intrinsics)" << std::endl;
            std::vector<double> reprojectionErrors;
            double totalAvgErr;

            //Find intrinsic and extrinsic camera parameters
            Intrinsics intrinsics;
            std::vector<cv::Mat> rVecs;
            std::vector<cv::Mat> tVecs;
            const auto rms = cv::calibrateCamera(objects3DVectors, points2DVectors, imageSize, intrinsics.cameraMatrix,
                                                 intrinsics.distortionCoefficients, rVecs, tVecs, calibrateCameraFlags);

            // cv::checkRange checks that every array element is neither NaN nor infinite
            const auto calibrationIsCorrect = cv::checkRange(intrinsics.cameraMatrix) && cv::checkRange(intrinsics.distortionCoefficients);
            if (!calibrationIsCorrect)
                error("Unvalid cameraMatrix and/or distortionCoefficients.", __LINE__, __FUNCTION__, __FILE__);

            std::tie(totalAvgErr, reprojectionErrors) = computeReprojectionErrors(objects3DVectors, points2DVectors, rVecs, tVecs, intrinsics);

            std::cout << "\nIntrinsics:\n";
            std::cout << "Re-projection error - cv::calibrateCamera vs. computeReprojectionErrors:\t" << rms << " vs. " << totalAvgErr << "\n";
            std::cout << "Intrinsics_K" << intrinsics.cameraMatrix << "\n";
            std::cout << "Intrinsics_distCoeff" << intrinsics.distortionCoefficients << "\n" << std::endl;

            return std::make_tuple(intrinsics, reprojectionErrors, totalAvgErr);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_tuple(Intrinsics{}, std::vector<double>(), 0.);
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
            const cv::Size gridInnerCornersCvSize{gridInnerCorners.x, gridInnerCorners.y};

            // Read images in folder
            std::vector<std::vector<cv::Point2f>> points2DVectors;
            const auto images = getImages(imagesFolder);

            // Get 2D grid corners of each image
            std::vector<cv::Mat> imagesWithCorners;
            const auto imageSize = images.at(0).size();
            for (const auto& image : images)
            {
                // Security check
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
                cv::Mat imageToPlot = image.clone();
                if (found)
                    drawGridCorners(imageToPlot, gridInnerCornersCvSize, points2DVector);
                // cv::pyrDown(imageToPlot, imageToPlot);
                // cv::imshow("Image View", imageToPlot);
                // cv::waitKey(delayMilliseconds);
                imagesWithCorners.emplace_back(imageToPlot);
            }

            // Run calibration
            // objects3DVector is the same one for each image
            const std::vector<std::vector<cv::Point3f>> objects3DVectors(points2DVectors.size(),
                                                                         getObjects3DVector(gridInnerCornersCvSize,
                                                                                            gridSquareSizeMm));
            Intrinsics intrinsics;
            std::vector<double> reprojErrs;
            double totalAvgErr;
            std::tie(intrinsics, reprojErrs, totalAvgErr) = calcIntrinsicParameters(imageSize, points2DVectors, objects3DVectors, flags);

            // Save intrinsics/results
            CameraParameterReader cameraParameterReader(serialNumber, intrinsics.cameraMatrix, intrinsics.distortionCoefficients);
            cameraParameterReader.writeParameters(outputParameterFolder);

            // Save images with corners
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
}
