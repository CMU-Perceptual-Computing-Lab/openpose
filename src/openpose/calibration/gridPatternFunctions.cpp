#include <openpose/utilities/fileSystem.hpp>
#include <openpose/calibration/gridPatternFunctions.hpp>

namespace op
{
    // Private functions
    void improveCornersPositionsAtSubPixelLevel(std::vector<cv::Point2f>& points2DVector, const cv::Mat& image)
    {
        try
        {
            if (!image.empty() && points2DVector.size() > 1)
            {
                // cv::Mat imageGray;
                // cv::cvtColor(image, imageGray, CV_BGR2GRAY);
                const auto winSize = std::max(5,
                    (int)std::round(cv::norm(cv::Mat(points2DVector.at(0) - points2DVector.at(1)), cv::NORM_INF) / 4));
                cv::cornerSubPix(image,
                                 points2DVector,
                                 cv::Size{winSize, winSize}, // Depending on the chessboard size;
                                 // cv::Size{11,11}, // Default in code I got, used above one
                                 cv::Size{-1,-1},
                                 cv::TermCriteria{ CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 1000, 1e-9 });
                                 // cv::TermCriteria{ CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 });  // Default
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::pair<bool, std::vector<cv::Point2f>> lightlyTryToFindGridCorners(const cv::Mat& image,
                                                                          const cv::Size& gridInnerCorners)
    {
        try
        {
            std::vector<cv::Point2f> points2DVector;
            // CALIB_CB_FAST_CHECK -> faster but more false negatives
            const auto chessboardFound = cv::findChessboardCorners(image,
                                                                   gridInnerCorners,
                                                                   points2DVector,
                                                                   CV_CALIB_CB_ADAPTIVE_THRESH
                                                                    | CV_CALIB_CB_NORMALIZE_IMAGE
                                                                    | CV_CALIB_CB_FILTER_QUADS);

            return std::make_pair(chessboardFound, points2DVector);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_pair(false, std::vector<cv::Point2f>());
        }
    }

    std::pair<bool, std::vector<cv::Point2f>> mediumlyTryToFindGridCorners(const cv::Mat& image,
                                                                           const cv::Size& gridInnerCorners)
    {
        try
        {
            bool chessboardFound{false};
            std::vector<cv::Point2f> points2DVector;

            if (!image.empty())
            {
                std::tie(chessboardFound, points2DVector) = lightlyTryToFindGridCorners(image, gridInnerCorners);

                if (!chessboardFound)
                {
                    std::tie(chessboardFound, points2DVector) = lightlyTryToFindGridCorners(image, gridInnerCorners);
                    if (!chessboardFound)
                    {
                        // If not chessboardFound -> try sharpening the image
                        // std::cerr << "Grid not found, trying sharpening" << std::endl;
                        cv::Mat sharperedImage;
                        // hardcoded filter size, to be tested on 50 mm lens
                        cv::GaussianBlur(image, sharperedImage, cv::Size{0,0}, 105);
                        // hardcoded weight, to be tested.
                        cv::addWeighted(image, 1.8, sharperedImage, -0.8, 0, sharperedImage);
                        std::tie(chessboardFound, points2DVector) = lightlyTryToFindGridCorners(
                            sharperedImage, gridInnerCorners);
                    }
                }
            }

            return std::make_pair(chessboardFound, points2DVector);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_pair(false, std::vector<cv::Point2f>());
        }
    }

    std::pair<bool, std::vector<cv::Point2f>> heavilyTryToFindGridCorners(const cv::Mat& image,
                                                                          const cv::Size& gridInnerCorners)
    {
        try
        {
            bool chessboardFound{false};
            std::vector<cv::Point2f> points2DVector;

            // Loading images
            if (!image.empty())
            {
                std::tie(chessboardFound, points2DVector) = mediumlyTryToFindGridCorners(image, gridInnerCorners);

                if (!chessboardFound)
                {
                    cv::Mat tempImage;
                    while (!chessboardFound) // 71 x 71 > 5000
                    {
                        if (!tempImage.empty())
                            cv::pyrDown(tempImage, tempImage);
                        else
                            cv::pyrDown(image, tempImage);
                        std::tie(chessboardFound, points2DVector) = mediumlyTryToFindGridCorners(
                            tempImage, gridInnerCorners);

                        // After next pyrDown if will be area > 5000 < 71 x 71 px image
                        if (tempImage.size().area() <= 20e3)
                            break;
                    }
                    if (chessboardFound && image.size().width != tempImage.size().width)
                    {
                        std::cerr << "Chessboard found at lower resolution: " << tempImage.size() << "px" << std::endl;
                        for (auto& point : points2DVector)
                            point *= (image.size().width / tempImage.size().width);
                    }
                }
            }

            return std::make_pair(chessboardFound, points2DVector);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_pair(false, std::vector<cv::Point2f>());
        }
    }

    Points2DOrigin getPoints2DOrigin(const cv::Size& gridInnerCorners,
                                     const std::vector<cv::Point2f>& points2DVector)
    {
        try
        {
            Points2DOrigin points2DOrigin;

            const auto outterCornerIndices = getOutterCornerIndices(points2DVector, gridInnerCorners);
            const std::vector<cv::Point2f> fourPointsVector{
                points2DVector.at(outterCornerIndices.at(0)),
                points2DVector.at(outterCornerIndices.at(1)),
                points2DVector.at(outterCornerIndices.at(2)),
                points2DVector.at(outterCornerIndices.at(3))
            };
            auto fourPointsVectorXSorted = fourPointsVector;
            std::sort(fourPointsVectorXSorted.begin(), fourPointsVectorXSorted.end(),
                      [](cv::Point2f const& l, cv::Point2f const& r) { return l.x < r.x; });

            // If point is in the left
            if (points2DVector.at(0) == fourPointsVectorXSorted.at(0)
                || points2DVector.at(0) == fourPointsVectorXSorted.at(1))
            {
                if (points2DVector.at(0).y >= fourPointsVectorXSorted.at(0).y
                    && points2DVector.at(0).y >= fourPointsVectorXSorted.at(1).y)
                    points2DOrigin = {Points2DOrigin::BottomLeft};
                else
                    points2DOrigin = {Points2DOrigin::TopLeft};
            }

            // If point is in the right
            else
            {
                if (points2DVector.at(0).y >= fourPointsVectorXSorted.at(2).y
                    && points2DVector.at(0).y >= fourPointsVectorXSorted.at(3).y)
                    points2DOrigin = {Points2DOrigin::BottomRight};
                else
                    points2DOrigin = {Points2DOrigin::TopRight};
            }

            return points2DOrigin;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Points2DOrigin::TopLeft;
        }
    }

    void invertXPositionsIndices(std::vector<cv::Point2f>& points2DVector, const cv::Size& gridInnerCorners)
    {
        try
        {
            // Invert points orientation within each row
            for (auto x = 0 ; x < gridInnerCorners.width / 2 ; x++)
            {
                for (auto y = 0 ; y < gridInnerCorners.height ; y++)
                {
                    const auto rowComponent = gridInnerCorners.width * y;
                    std::swap(points2DVector.at(rowComponent + x),
                              points2DVector.at(rowComponent + (gridInnerCorners.width - 1) - x));
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }



    // Public functions
    std::pair<bool, std::vector<cv::Point2f>> findAccurateGridCorners(const cv::Mat& image,
                                                                      const cv::Size& gridInnerCorners)
    {
        try
        {
            // Grayscale for speeding up
            cv::Mat imageGray;
            cv::cvtColor(image, imageGray, CV_BGR2GRAY);

            // Find chessboard corners
            auto foundGridCornersAndLocations = heavilyTryToFindGridCorners(imageGray, gridInnerCorners);

            // Increase accuracy
            if (foundGridCornersAndLocations.first)
                improveCornersPositionsAtSubPixelLevel(foundGridCornersAndLocations.second, imageGray);

            // return final result
            return foundGridCornersAndLocations;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_pair(false, std::vector<cv::Point2f>());
        }
    }

    std::vector<cv::Point3f> getObjects3DVector(const cv::Size& gridInnerCorners,
                                                const float gridSquareSizeMm)
    {
        try
        {
            std::vector<cv::Point3f> objects3DVector;

            for (auto x = 0 ; x < gridInnerCorners.height ; x++)
                for (auto y = 0 ; y < gridInnerCorners.width ; y++)
                    objects3DVector.emplace_back(cv::Point3f{x*gridSquareSizeMm,
                                                             y*gridSquareSizeMm,
                                                             0.f});
            // for (int i = 0; i < gridInnerCorners.height; i++)
            //     for (int j = 0; j < gridInnerCorners.width; j++)
            //         corners.emplace_back(cv::Point3f(float( j*squareSize ), float( i*squareSize ), 0));

            return objects3DVector;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    void drawGridCorners(cv::Mat& image, const cv::Size& gridInnerCorners,
                         const std::vector<cv::Point2f>& points2DVector)
    {
        try
        {
            // Draw every corner
            cv::drawChessboardCorners(image, gridInnerCorners, cv::Mat(points2DVector), true/*found*/);
            // Draw 4 outter corners
            const auto radiusAndThickness = std::max(5,
                                                     (int)std::round(std::sqrt(image.cols * image.rows) / 100));
            const auto outterCornerIndices = getOutterCornerIndices(points2DVector, gridInnerCorners);
            cv::circle(image, points2DVector.at(outterCornerIndices.at(0)), radiusAndThickness,
                       cv::Scalar{0, 0, 255, 1}, radiusAndThickness);
            cv::circle(image, points2DVector.at(outterCornerIndices.at(1)), radiusAndThickness,
                       cv::Scalar{0, 255, 0, 1}, radiusAndThickness);
            cv::circle(image, points2DVector.at(outterCornerIndices.at(2)), radiusAndThickness,
                       cv::Scalar{255, 0, 0, 1}, radiusAndThickness);
            cv::circle(image, points2DVector.at(outterCornerIndices.at(3)), radiusAndThickness,
                       cv::Scalar{0, 0, 0, 1}, radiusAndThickness);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::array<unsigned int, 4> getOutterCornerIndices(const std::vector<cv::Point2f>& points2DVector,
                                                       const cv::Size& gridInnerCorners)
    {
        try
        {
            std::array<unsigned int, 4> result{
                0u,
                (unsigned int)(gridInnerCorners.width - 1),
                (unsigned int)points2DVector.size() - gridInnerCorners.width,
                (unsigned int)points2DVector.size() - 1
            };

            for (const auto& value : result)
                if (value >= points2DVector.size())
                    error("Vector `points2DVector` is smaller than expected. Were all corners found?",
                          __LINE__, __FUNCTION__, __FILE__);

            return result;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::array<unsigned int, 4>();
        }
    }

    void reorderPoints(std::vector<cv::Point2f>& points2DVector,
                       const cv::Size& gridInnerCorners,
                       const Points2DOrigin points2DOriginDesired)
    {
        try
        {
            const auto points2DOriginCurrent = getPoints2DOrigin(gridInnerCorners, points2DVector);

            // No mirrored image
            if (points2DOriginDesired == Points2DOrigin::TopLeft)
            {
                if (points2DOriginCurrent == Points2DOrigin::TopRight)
                    invertXPositionsIndices(points2DVector, gridInnerCorners);
                else if (points2DOriginCurrent == Points2DOrigin::BottomLeft)
                {
                    std::reverse(points2DVector.begin(), points2DVector.end());
                    invertXPositionsIndices(points2DVector, gridInnerCorners);
                }
                else if (points2DOriginCurrent == Points2DOrigin::BottomRight)
                    std::reverse(points2DVector.begin(), points2DVector.end());
            }

            // Mirrored image
            else if (points2DOriginDesired == Points2DOrigin::TopRight)
            {
                if (points2DOriginCurrent == Points2DOrigin::TopLeft)
                    invertXPositionsIndices(points2DVector, gridInnerCorners);
                else if (points2DOriginCurrent == Points2DOrigin::BottomLeft)
                    std::reverse(points2DVector.begin(), points2DVector.end());
                else if (points2DOriginCurrent == Points2DOrigin::BottomRight)
                {
                    std::reverse(points2DVector.begin(), points2DVector.end());
                    invertXPositionsIndices(points2DVector, gridInnerCorners);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void plotGridCorners(const cv::Size& gridInnerCorners,
                         const std::vector<cv::Point2f>& points2DVector,
                         const bool gridIsMirrored,
                         const std::string& imagePath,
                         const cv::Mat& image)
    {
        try
        {
            cv::Mat imageToPlot = image.clone();

            // Draw corners
            drawGridCorners(imageToPlot, gridInnerCorners, points2DVector);

            // Plotting results
            const std::string windowName{
                getFileNameAndExtension(imagePath) + " - " + (gridIsMirrored ? "" : "no") + " mirrored grid"};
            cv::pyrDown(imageToPlot, imageToPlot);
            cv::imshow(windowName, imageToPlot);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
