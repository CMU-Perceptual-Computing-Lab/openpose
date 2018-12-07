#ifndef OPENPOSE_3D_POSE_TRIANGULATION_HPP
#define OPENPOSE_3D_POSE_TRIANGULATION_HPP

#include <opencv2/core/core.hpp>
#include <openpose/core/common.hpp>

namespace op
{
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

    double triangulateWithOptimization(cv::Mat& reconstructedPoint, const std::vector<cv::Mat>& cameraMatrices,
                                   const std::vector<cv::Point2d>& pointsOnEachCamera,
                                   const double reprojectionMaxAcceptable);
}

#endif // OPENPOSE_3D_POSE_TRIANGULATION_HPP
