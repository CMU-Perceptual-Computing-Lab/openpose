#ifndef OPENPOSE_PRIVATE_3D_POSE_TRIANGULATION_PRIVATE_HPP
#define OPENPOSE_PRIVATE_3D_POSE_TRIANGULATION_PRIVATE_HPP

#include <opencv2/core/core.hpp>
#include <openpose/core/common.hpp>

namespace op
{
    /**
     * 3D triangulation given known camera parameter matrices and based on linear DLT algorithm.
     * The returned cv::Mat is a 4x1 matrix, where the last coordinate is 1.
     */
    double triangulate(
        cv::Mat& reconstructedPoint, const std::vector<cv::Mat>& cameraMatrices,
        const std::vector<cv::Point2d>& pointsOnEachCamera);

    /**
     * 3D triangulation given known camera parameter matrices and based on linear DLT algorithm with additional LMA
     * non-linear refinement.
     * The returned cv::Mat is a 4x1 matrix, where the last coordinate is 1.
     * Note: If Ceres is not enabled, the LMA refinement is skipped and this function is equivalent to triangulate().
     */
    double triangulateWithOptimization(
        cv::Mat& reconstructedPoint, const std::vector<cv::Mat>& cameraMatrices,
        const std::vector<cv::Point2d>& pointsOnEachCamera, const double reprojectionMaxAcceptable);
}

#endif // OPENPOSE_PRIVATE_3D_POSE_TRIANGULATION_PRIVATE_HPP
