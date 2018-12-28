#ifndef OPENPOSE_3D_POSE_TRIANGULATION_HPP
#define OPENPOSE_3D_POSE_TRIANGULATION_HPP

#include <opencv2/core/core.hpp>
#include <openpose/core/common.hpp>

namespace op
{
    /**
     * 3D triangulation given known camera parameter matrices and based on linear DLT algorithm.
     * The returned cv::Mat is a 4x1 matrix, where the last coordinate is 1.
     */
    OP_API double triangulate(
        cv::Mat& reconstructedPoint, const std::vector<cv::Mat>& cameraMatrices,
        const std::vector<cv::Point2d>& pointsOnEachCamera);

    /**
     * 3D triangulation given known camera parameter matrices and based on linear DLT algorithm with additional LMA
     * non-linear refinement.
     * The returned cv::Mat is a 4x1 matrix, where the last coordinate is 1.
     * Note: If Ceres is not enabled, the LMA refinement is skipped and this function is equivalent to triangulate().
     */
    OP_API double triangulateWithOptimization(
        cv::Mat& reconstructedPoint, const std::vector<cv::Mat>& cameraMatrices,
        const std::vector<cv::Point2d>& pointsOnEachCamera, const double reprojectionMaxAcceptable);

    class OP_API PoseTriangulation
    {
    public:
        PoseTriangulation(const int minViews3d);

        virtual ~PoseTriangulation();

        void initializationOnThread();

        Array<float> reconstructArray(const std::vector<Array<float>>& keypointsVector,
                                      const std::vector<cv::Mat>& cameraMatrices,
                                      const std::vector<Point<int>>& imageSizes) const;

        std::vector<Array<float>> reconstructArray(const std::vector<std::vector<Array<float>>>& keypointsVector,
                                                   const std::vector<cv::Mat>& cameraMatrices,
                                                   const std::vector<Point<int>>& imageSizes) const;

    private:
        const int mMinViews3d;
    };
}

#endif // OPENPOSE_3D_POSE_TRIANGULATION_HPP
