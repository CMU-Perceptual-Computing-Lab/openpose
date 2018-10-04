#ifdef USE_3D_ADAM_MODEL
#ifndef OPENPOSE_3D_JOINT_ANGLE_ESTIMATION_HPP
#define OPENPOSE_3D_JOINT_ANGLE_ESTIMATION_HPP

#ifdef USE_EIGEN
    #include <Eigen/Core>
#endif
#ifdef USE_3D_ADAM_MODEL
    #include <adam/totalmodel.h>
#endif
#include <opencv2/core/core.hpp>
#include <openpose/core/common.hpp>

namespace op
{
    OP_API int mapOPToAdam(const int oPPart);

    class OP_API JointAngleEstimation
    {
    public:
        static const std::shared_ptr<const TotalModel> getTotalModel();

        JointAngleEstimation(const bool returnJacobian);

        virtual ~JointAngleEstimation();

        void initializationOnThread();

        void adamFastFit(Eigen::Matrix<double, 62, 3, Eigen::RowMajor>& adamPose,
                         Eigen::Vector3d& adamTranslation,
                         Eigen::Matrix<double, Eigen::Dynamic, 1>& vtVec,
                         Eigen::Matrix<double, Eigen::Dynamic, 1>& j0Vec,
                         Eigen::VectorXd& adamFacecoeffsExp,
                         const Array<float>& poseKeypoints3D,
                         const Array<float>& faceKeypoints3D,
                         const std::array<Array<float>, 2>& handKeypoints3D);

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplJointAngleEstimation;
        std::shared_ptr<ImplJointAngleEstimation> spImpl;

        // PIMP requires DELETE_COPY & destructor, or extra code
        // http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html
        DELETE_COPY(JointAngleEstimation);
    };
}

#endif // OPENPOSE_3D_JOINT_ANGLE_ESTIMATION_HPP
#endif
