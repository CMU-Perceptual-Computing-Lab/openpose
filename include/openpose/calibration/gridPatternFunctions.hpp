#ifndef OPENPOSE_CALIBRATION_GRID_PATTERN_FUNCTIONS_HPP
#define OPENPOSE_CALIBRATION_GRID_PATTERN_FUNCTIONS_HPP

#include <opencv2/opencv.hpp>
#include <openpose/core/common.hpp>

namespace op
{
    enum class Points2DOrigin
    {
        TopLeft,
        TopRight,
        BottomLeft,
        BottomRight
    };

    OP_API std::pair<bool, std::vector<cv::Point2f>> findAccurateGridCorners(const cv::Mat& image,
                                                                             const cv::Size& gridInnerCorners);

    OP_API std::vector<cv::Point3f> getObjects3DVector(const cv::Size& gridInnerCorners,
                                                       const float gridSquareSizeMm);

    OP_API void drawGridCorners(cv::Mat& image, const cv::Size& gridInnerCorners,
                                const std::vector<cv::Point2f>& points2DVector);

    OP_API std::array<unsigned int, 4> getOutterCornerIndices(const std::vector<cv::Point2f>& points2DVector,
                                                              const cv::Size& gridInnerCorners);

    OP_API void reorderPoints(std::vector<cv::Point2f>& points2DVector,
                              const cv::Size& gridInnerCorners,
                              const Points2DOrigin points2DOriginDesired);

    OP_API void plotGridCorners(const cv::Size& gridInnerCorners,
                                const std::vector<cv::Point2f>& points2DVector,
                                const bool gridIsMirrored,
                                const std::string& imagePath,
                                const cv::Mat& image);
}

#endif // OPENPOSE_CALIBRATION_GRID_PATTERN_FUNCTIONS_HPP
