#ifndef OPENPOSE_PRIVATE_CALIBRATION_GRID_PATTERN_FUNCTIONS_HPP
#define OPENPOSE_PRIVATE_CALIBRATION_GRID_PATTERN_FUNCTIONS_HPP

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

    std::pair<bool, std::vector<cv::Point2f>> findAccurateGridCorners(
        const cv::Mat& image, const cv::Size& gridInnerCorners);

    std::vector<cv::Point3f> getObjects3DVector(
        const cv::Size& gridInnerCorners, const float gridSquareSizeMm);

    void drawGridCorners(
        cv::Mat& image, const cv::Size& gridInnerCorners, const std::vector<cv::Point2f>& points2DVector);

    std::array<unsigned int, 4> getOutterCornerIndices(
        const std::vector<cv::Point2f>& points2DVector, const cv::Size& gridInnerCorners);

    void reorderPoints(
        std::vector<cv::Point2f>& points2DVector, const cv::Size& gridInnerCorners,
        const cv::Mat& image, const bool showWarning = true);

    void plotGridCorners(
        const cv::Size& gridInnerCorners, const std::vector<cv::Point2f>& points2DVector,
        const std::string& imagePath, const cv::Mat& image);
}

#endif // OPENPOSE_PRIVATE_CALIBRATION_GRID_PATTERN_FUNCTIONS_HPP
