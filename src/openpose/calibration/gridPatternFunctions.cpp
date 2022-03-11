#include <openpose_private/calibration/gridPatternFunctions.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp>

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
                cv::cornerSubPix(
                    image, points2DVector,
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

    std::pair<bool, std::vector<cv::Point2f>> tryToFindGridCorners(const cv::Mat& image,
                                                                   const cv::Size& gridInnerCorners)
    {
        try
        {
            std::vector<cv::Point2f> points2DVector;
            // CALIB_CB_FAST_CHECK -> faster but more false negatives
            const auto chessboardFound = cv::findChessboardCorners(
                image, gridInnerCorners, points2DVector,
                CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE | CV_CALIB_CB_FILTER_QUADS);

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
                std::tie(chessboardFound, points2DVector) = tryToFindGridCorners(image, gridInnerCorners);

                if (!chessboardFound)
                {
                    std::tie(chessboardFound, points2DVector) = tryToFindGridCorners(image, gridInnerCorners);
                    if (!chessboardFound)
                    {
                        // If not chessboardFound -> try sharpening the image
                        cv::Mat sharperedImage;
                        // hardcoded filter size, to be tested on 50 mm lens
                        cv::GaussianBlur(image, sharperedImage, cv::Size{0,0}, 105);
                        // hardcoded weight, to be tested.
                        cv::addWeighted(image, 1.8, sharperedImage, -0.8, 0, sharperedImage);
                        std::tie(chessboardFound, points2DVector) = tryToFindGridCorners(
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
                    auto counter = 0;
                    while (!chessboardFound && counter <= 2) // 3 pyrdown max
                    {
                        cv::pyrDown((!tempImage.empty() ? tempImage : image), tempImage);
                        std::tie(chessboardFound, points2DVector) = mediumlyTryToFindGridCorners(
                            tempImage, gridInnerCorners);
                        counter++;

                        // After next pyrDown if will be area > 5000 < 71 x 71 px image
                        if (tempImage.size().area() <= 20e3)
                            break;
                    }
                    if (chessboardFound && image.size().width != tempImage.size().width)
                    {
                        opLog("Chessboard found at lower resolution (" + std::to_string(tempImage.cols) + "x"
                            + std::to_string(tempImage.rows) + ").", Priority::High);
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
    std::pair<bool, std::vector<cv::Point2f>> findAccurateGridCorners(
        const cv::Mat& image, const cv::Size& gridInnerCorners)
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

    std::vector<cv::Point3f> getObjects3DVector(
        const cv::Size& gridInnerCorners, const float gridSquareSizeMm)
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
            // Draw 4 outer corners
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

    std::array<unsigned int, 4> getOutterCornerIndices(
        const std::vector<cv::Point2f>& points2DVector, const cv::Size& gridInnerCorners)
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

    void reorderPoints(std::vector<cv::Point2f>& points2DVector, const cv::Size& gridInnerCorners,
                       const cv::Mat& image, const bool showWarning)
    {
        try
        {
            const auto debugging = false;
            // gridInnerCorners is 2 even
            // In this case, there will be 2 black corners in diagonal, and 2 white ones in the opposite diagonal
            // No way to get absolute orientation.
            // OR
            // gridInnerCorners is 2 odd
            // In this case, there will be all black (or white) corners in diagonal, and 2 white ones in the opposite
            // diagonal
            // No way to get absolute orientation.
            if ((gridInnerCorners.width % 2 == 0 && gridInnerCorners.height % 2 == 0)
                || (gridInnerCorners.width % 2 == 1 && gridInnerCorners.height % 2 == 1))
            {
                // Warning
                if (showWarning)
                    opLog("For maximum multi-view accuracy: The number of corners of the chessboard should be even in"
                        " 1 dimension and odd in the other (e.g., 1x2, 2x1, 1x4, 3x8, 6x9, 9x6, etc.). Otherwise,"
                        " extrinsics calibration results might be affected.", Priority::High);
                // Old method
                const auto points2DOriginCurrent = getPoints2DOrigin(gridInnerCorners, points2DVector);
                // // No mirrored image
                // if (points2DOriginDesired == Points2DOrigin::TopLeft)
                // {
                    if (points2DOriginCurrent == Points2DOrigin::TopRight)
                        invertXPositionsIndices(points2DVector, gridInnerCorners);
                    else if (points2DOriginCurrent == Points2DOrigin::BottomLeft)
                    {
                        std::reverse(points2DVector.begin(), points2DVector.end());
                        invertXPositionsIndices(points2DVector, gridInnerCorners);
                    }
                    else if (points2DOriginCurrent == Points2DOrigin::BottomRight)
                        std::reverse(points2DVector.begin(), points2DVector.end());
                // }
                // // Mirrored image
                // else if (points2DOriginDesired == Points2DOrigin::TopRight)
                // {
                //     if (points2DOriginCurrent == Points2DOrigin::TopLeft)
                //         invertXPositionsIndices(points2DVector, gridInnerCorners);
                //     else if (points2DOriginCurrent == Points2DOrigin::BottomLeft)
                //         std::reverse(points2DVector.begin(), points2DVector.end());
                //     else if (points2DOriginCurrent == Points2DOrigin::BottomRight)
                //     {
                //         std::reverse(points2DVector.begin(), points2DVector.end());
                //         invertXPositionsIndices(points2DVector, gridInnerCorners);
                //     }
                // }
            }
            // gridInnerCorners is 1 even, 1 odd
            // In this case, there will be 2 consecutive white and 2 consecutive black corners.
            // Easy to get absolute orientation: We detect the 2 black corners. Then, we pick as main one the black
            // one whose left is white and right is black (only 1 satisfies this) by cross product properties.
            else
            {
                const auto outterCornerIndices = getOutterCornerIndices(points2DVector, gridInnerCorners);
                const std::vector<cv::Point2f> fourPointsVector{
                    points2DVector.at(outterCornerIndices.at(0)),
                    points2DVector.at(outterCornerIndices.at(1)),
                    points2DVector.at(outterCornerIndices.at(2)),
                    points2DVector.at(outterCornerIndices.at(3))
                };
                const auto point01 = fourPointsVector.at(0)-fourPointsVector.at(1);
                const auto point01Norm = cv::norm(point01);
                const auto point02 = fourPointsVector.at(0)-fourPointsVector.at(2);
                const auto point02Norm = cv::norm(point02);
                const auto point13 = fourPointsVector.at(1)-fourPointsVector.at(3);
                const auto point13Norm = cv::norm(point13);
                const auto averageSquareSizePx =
                    (point01Norm/gridInnerCorners.width
                    + point02Norm/gridInnerCorners.height
                    + point13Norm/gridInnerCorners.height
                    + cv::norm(fourPointsVector.at(3)-fourPointsVector.at(2))/gridInnerCorners.width)
                    / 4.;
                // Debugging
                if (debugging)
                    opLog("\naverageSquareSizePx: " + std::to_string(averageSquareSizePx));
                // How many pixels does the outer square has?
                // 0.67 is a threshold to be safe
                const auto diagonalLength = 0.67 * std::sqrt(2) * averageSquareSizePx;
                // Debugging
                if (debugging)
                    opLog("diagonalLength: " + std::to_string(diagonalLength));

                // In which direction do I have to look?
                // Normal vector between corners 0-1, 0-2, 1-3?
                const auto point01Direction = 1. / point01Norm * point01;
                const auto point02Direction = 1. / point02Norm * point02;
                const auto point13Direction = 1. / point13Norm * point13;
                // Debugging
                if (debugging)
                {
                    opLog("\npoint01Direction:");
                    opLog(point01Direction);
                    opLog("point02Direction:");
                    opLog(point02Direction);
                    opLog("point13Direction:");
                    opLog(point13Direction);
                    opLog(" ");
                }

                auto pointDirection = fourPointsVector; // Initialization
                pointDirection[0] = 1. / cv::norm(point01Direction + point02Direction)
                                  * (point01Direction + point02Direction);
                pointDirection[1] = pointDirection[0];
                pointDirection[1].y *= -1;
                pointDirection[2] = pointDirection[0];
                pointDirection[2].x *= -1;
                pointDirection[3] = -pointDirection[0];
                // Debugging
                if (debugging)
                {
                    for (auto i = 0u ; i < fourPointsVector.size() ; i++)
                    {
                        opLog("pointDirection[" + std::to_string(i) + "]:");
                        opLog(pointDirection[i]);
                    }
                    opLog(" ");
                }

                // Get line to check whether outer grid color is black
                auto pointLimit = fourPointsVector; // Initialization
                for (auto i = 0u ; i < fourPointsVector.size() ; i++)
                    pointLimit[i] = fourPointsVector[i] + diagonalLength * pointDirection[i];

                // Line search to see if white or black
                std::vector<double> meanPxValues(fourPointsVector.size());
                const auto numberPointsInLine = 25;
                const auto imageSize = image.size();
                for (auto i = 0u ; i < fourPointsVector.size() ; i++)
                {
                    auto sum = 0.;
                    auto count = 0u;
                    for (auto lm = 0; lm < numberPointsInLine; lm++)
                    {
                        const auto mX = fastMax(
                            0, fastMin(
                                imageSize.width-1, positiveIntRound(fourPointsVector[i].x + lm*pointDirection[i].x)));
                        const auto mY = fastMax(
                            0, fastMin(
                                imageSize.height-1, positiveIntRound(fourPointsVector[i].y + lm*pointDirection[i].y)));
                        const cv::Vec3b bgrValue = image.at<cv::Vec3b>(mY, mX);
                        sum += (bgrValue.val[0] + bgrValue.val[1] + bgrValue.val[2])/3;
                        count++;
                    }
                    meanPxValues[i] = sum/count;
                // Debugging
                if (debugging)
                    opLog("meanPxValues[" + std::to_string(i) + "]: " + std::to_string(meanPxValues[i]));
                }

                // Get black indexes
                const bool blackIs0 = meanPxValues[0] < meanPxValues[3];
                bool blackIs1 = meanPxValues[1] < meanPxValues[2];

                // Debugging
                if (debugging)
                {
                    // Plotting visually results
                    auto imageToPlot = image.clone();
                    for (auto i = 0u ; i < fourPointsVector.size() ; i++)
                    cv::line(imageToPlot, fourPointsVector[i], pointLimit[i], cv::Scalar{0,0,255}, 10);
                    // Black indexes
                    opLog(" ");
                    opLog("blackIs0: " + std::to_string(blackIs0));
                    opLog("blackIs1: " + std::to_string(blackIs1));
                    // Plotting results
                    // Chessboard before
                    drawGridCorners(imageToPlot, gridInnerCorners, points2DVector);
                    cv::pyrDown(imageToPlot, imageToPlot);
                    cv::imshow("image_before", imageToPlot);
                }

                // Apply transformations
                // For simplicity, we assume 0 is black
                if (!blackIs0)
                {
                    std::reverse(points2DVector.begin(), points2DVector.end());
                    blackIs1 = !blackIs1;
                    // Debugging
                    if (debugging)
                        opLog("Swapping 0 and 3 so 0 is black.");
                }
                // Lead is 0 or 1||2 (depending on blackIs1)?
                const auto outterCornerIndicesAfter = getOutterCornerIndices(points2DVector, gridInnerCorners);
                const auto middle = 0.25f*(
                    fourPointsVector[0] + fourPointsVector[1] + fourPointsVector[2] + fourPointsVector[3]);
                const std::vector<cv::Point2f> fourPointsVectorAfter{
                    points2DVector.at(outterCornerIndicesAfter.at(0)) - middle,
                    points2DVector.at(outterCornerIndicesAfter.at(1)) - middle,
                    points2DVector.at(outterCornerIndicesAfter.at(2)) - middle,
                    points2DVector.at(outterCornerIndicesAfter.at(3)) - middle
                };
                const auto crossProduct = fourPointsVectorAfter[0].cross(fourPointsVectorAfter[(blackIs1 ? 1 : 2)]);
                // Debugging
                if (debugging)
                    opLog("crossProduct: " + std::to_string(crossProduct));
                const auto leadIs0 = crossProduct < 0;
                // Second transformation
                if (!leadIs0)
                {
                    // Debugging
                    if (debugging)
                        opLog("Lead is not 0.");
                    // Second black is 1
                    if (blackIs1)
                    {
                        // Debugging
                        if (debugging)
                            opLog("Lead was 1.");
                        invertXPositionsIndices(points2DVector, gridInnerCorners); // 1->0
                    }
                    // Second black is 2
                    else
                    {
                        // Debugging
                        if (debugging)
                            opLog("Lead was 2.");
                        std::reverse(points2DVector.begin(), points2DVector.end()); // 2->3
                        invertXPositionsIndices(points2DVector, gridInnerCorners); // 3->0
                    }
                }

                // Debugging
                if (debugging)
                {
                    // Chessboard after
                    plotGridCorners(gridInnerCorners, points2DVector, "image_after.jpg", image);
                    cv::waitKey(0);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void plotGridCorners(
        const cv::Size& gridInnerCorners, const std::vector<cv::Point2f>& points2DVector,
        const std::string& imagePath, const cv::Mat& image)
    {
        try
        {
            cv::Mat imageToPlot = image.clone();
            // Draw corners
            drawGridCorners(imageToPlot, gridInnerCorners, points2DVector);
            // Plotting results
            const std::string windowName = getFileNameAndExtension(imagePath);
            cv::pyrDown(imageToPlot, imageToPlot);
            cv::imshow(windowName, imageToPlot);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
