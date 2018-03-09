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

        Array<float> reconstructArray(const std::vector<Array<float>>& keypointsVector,
                                      const std::vector<cv::Mat>& cameraMatrices) const;

    private:
        const int mMinViews3d;
    };
}

#endif // OPENPOSE_3D_POSE_TRIANGULATION_HPP
