#include <chrono>
#include <cstring>
#include <iostream>
#include <AdamFastCost.h>

// #define COMPARE_AUTOMATIC_VS_MANUAL

bool AdamFastCost::Evaluate(double const* const* parameters,
    double* residuals,
    double** jacobians) const
{
// const auto start = std::chrono::high_resolution_clock::now();
// const auto start1 = std::chrono::high_resolution_clock::now();
    using namespace Eigen;

    // 1st step: forward kinematics
    const int num_t = (TotalModel::NUM_JOINTS) * 3 * 5;  // transform 3 * 4 + joint location 3 * 1
    const double* const p_eulers = parameters[1];
// const auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start1).count();
// const auto start2 = std::chrono::high_resolution_clock::now();

    #ifdef COMPARE_AUTOMATIC_VS_MANUAL
        // uncomment these lines for comparison with the old code
        Matrix<double, Dynamic, 3 * TotalModel::NUM_JOINTS, RowMajor> old_dTrdP(num_t, 3 * TotalModel::NUM_JOINTS);
        Matrix<double, Dynamic, 3 * TotalModel::NUM_JOINTS, RowMajor> old_dTrdJ(num_t, 3 * TotalModel::NUM_JOINTS);
        old_dTrdP.setZero(); old_dTrdJ.setZero();
        VectorXd old_transforms_joint(3 * TotalModel::NUM_JOINTS * 4 + 3 * TotalModel::NUM_JOINTS); // the first part is transform, the second part is outJoint

        ceres::AutoDiffCostFunction<smpl::PoseToTransformsNoLR_Eulers_adamModel,
        (TotalModel::NUM_JOINTS) * 3 * 4 + 3 * TotalModel::NUM_JOINTS,
        (TotalModel::NUM_JOINTS) * 3,
        (TotalModel::NUM_JOINTS) * 3> old_p2t(new smpl::PoseToTransformsNoLR_Eulers_adamModel(m_adam));

        const double* old_p2t_parameters[2] = { p_eulers, m_J0.data() };
        double* old_p2t_residuals = old_transforms_joint.data();
        double* old_p2t_jacobians[2] = { old_dTrdP.data(), old_dTrdJ.data() };
        std::clock_t startComparison1 = std::clock();
        old_p2t.Evaluate(old_p2t_parameters, old_p2t_residuals, old_p2t_jacobians);
        std::clock_t endComparison1 = std::clock();
    #endif // COMPARE_AUTOMATIC_VS_MANUAL

// const auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start2).count();
// const auto start3 = std::chrono::high_resolution_clock::now();
    Matrix<double, Dynamic, 3 * TotalModel::NUM_JOINTS, RowMajor> dTrdP(num_t, 3 * TotalModel::NUM_JOINTS);
    VectorXd transforms_joint(3 * TotalModel::NUM_JOINTS * 4 + 3 * TotalModel::NUM_JOINTS); // the first part is transform, the second part is outJoint

    smpl::PoseToTransformsNoLR_Eulers_adamModel_withDiff p2t(m_adam, m_J0);
    const double* p2t_parameters[1] = { p_eulers };
    double* p2t_residuals = transforms_joint.data() + (TotalModel::NUM_JOINTS) * 3 * 4;
    #ifdef COMPARE_AUTOMATIC_VS_MANUAL
        std::clock_t startComparison2 = std::clock();
    #endif // COMPARE_AUTOMATIC_VS_MANUAL
    if (jacobians)
    {
        double* p2t_jacobians[1] = { dTrdP.data() + TotalModel::NUM_JOINTS * 3 * 4 * TotalModel::NUM_JOINTS * 3 };
        p2t.Evaluate(p2t_parameters, p2t_residuals, p2t_jacobians);
    }
    else
        p2t.Evaluate(p2t_parameters, p2t_residuals, nullptr);
    #ifdef COMPARE_AUTOMATIC_VS_MANUAL
        std::clock_t endComparison2 = std::clock();
    #endif // COMPARE_AUTOMATIC_VS_MANUAL
// const auto duration3 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start3).count();

    #ifdef COMPARE_AUTOMATIC_VS_MANUAL
        // uncomment these lines for comparison with the old code
        old_dTrdP.block(0, 0, 3 * TotalModel::NUM_JOINTS * 4, 3 * TotalModel::NUM_JOINTS).setZero();
        dTrdP.block(0, 0, 3 * TotalModel::NUM_JOINTS * 4, 3 * TotalModel::NUM_JOINTS).setZero();

        std::cout << "J" << std::endl;
        const auto maxCoeffJ = (old_transforms_joint - transforms_joint).block(3 * TotalModel::NUM_JOINTS * 4, 0, 3 * TotalModel::NUM_JOINTS, 1).maxCoeff();
        const auto minCoeffJ = (old_transforms_joint - transforms_joint).block(3 * TotalModel::NUM_JOINTS * 4, 0, 3 * TotalModel::NUM_JOINTS, 1).minCoeff();
        std::cout << "max diff: " << maxCoeffJ << std::endl;
        std::cout << "min diff: " << minCoeffJ << std::endl;
        assert(std::abs(maxCoeffJ) < 1e-12);
        assert(std::abs(minCoeffJ) < 1e-12);

        if (jacobians)
        {
            std::cout << "DJ" << std::endl;
            const auto maxCoeffDJ = (old_dTrdP - dTrdP).maxCoeff();
            const auto minCoeffDJ = (old_dTrdP - dTrdP).minCoeff();
            std::cout << "max diff: " << maxCoeffDJ << std::endl;
            std::cout << "min diff: " << minCoeffDJ << std::endl;
            assert(std::abs(maxCoeffDJ) < 1e-12);
            assert(std::abs(minCoeffDJ) < 1e-12);

            std::cout << "time 1: " << (endComparison1 - startComparison1) / (double)CLOCKS_PER_SEC << std::endl;
            std::cout << "time 2: " << (endComparison2 - startComparison2) / (double)CLOCKS_PER_SEC << std::endl;
        }
    #endif // COMPARE_AUTOMATIC_VS_MANUAL

// const auto start4 = std::chrono::high_resolution_clock::now();
    // Jacobian: d(Joint) / d(pose)
    VectorXd outJoint = transforms_joint.block<3 * TotalModel::NUM_JOINTS, 1>(3 * TotalModel::NUM_JOINTS * 4, 0);  // outJoint
// const auto duration4 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start4).count();
// const auto start5 = std::chrono::high_resolution_clock::now();

    // 2nd step: set residuals
    // Joint Constraints
    VectorXd tempJoints(3 * m_nCorrespond_adam2joints);  // predicted joint given current parameter
    const double* const t = parameters[0];
    const Map<const Vector3d> t_vec(t);
    for (int i = 0; i < m_adam.m_indices_jointConst_adamIdx.rows(); i++)
        tempJoints.block<3,1>(3 * i, 0) = outJoint.block<3,1>(3 * m_adam.m_indices_jointConst_adamIdx(i), 0) + t_vec;
// const auto duration5 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start5).count();
// const auto start6 = std::chrono::high_resolution_clock::now();
    int offset = m_adam.m_indices_jointConst_adamIdx.rows();
    for (int i = 0; i < m_adam.m_correspond_adam2lHand_adamIdx.rows(); i++)
        tempJoints.block<3,1>(3*(i + offset), 0) = outJoint.block<3,1>(3 * m_adam.m_correspond_adam2lHand_adamIdx(i), 0) + t_vec;
// const auto duration6 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start6).count();
// const auto start7 = std::chrono::high_resolution_clock::now();
    offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();
    for (int i = 0; i < m_adam.m_correspond_adam2rHand_adamIdx.rows(); i++)
        tempJoints.block<3,1>(3*(i + offset), 0) = outJoint.block<3,1>(3 * m_adam.m_correspond_adam2rHand_adamIdx(i), 0) + t_vec;
// const auto duration7 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start7).count();
// const auto start8 = std::chrono::high_resolution_clock::now();

    const auto* tempJointsPts = tempJoints.data();
    const auto* targetPts = m_targetPts.data();
    const auto* targetPtsWeight = m_targetPts_weight.data();
    std::fill(residuals, residuals + m_nResiduals, 0.0);
// const auto duration8 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start8).count();
// const auto start9 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < m_nCorrespond_adam2joints; i++)
    {
        if (!m_targetPts.block<3,1>(5 * i, 0).isZero(0))
        {
            for (int r = 0; r < 3 ; r++)
            {
                const int baseIndex = 3 * i + r;
                residuals[baseIndex] = (tempJointsPts[baseIndex] - targetPts[5 * i + r]) * targetPtsWeight[baseIndex];
            }
            // Vectorized (slower) equivalent
            // Map< VectorXd > res(residuals, m_nResiduals);
            // res.block<3,1>(3 * i, 0) = (tempJoints.block<3,1>(3 * i, 0) - m_targetPts.block<3,1>(5 * i, 0)).cwiseProduct(m_targetPts_weight.block<3,1>(3 * i, 0));
        }
    }
// const auto duration9 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start9).count();
// auto duration10a = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start9).count() * 0;
// auto duration10b = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start9).count() * 0;
// auto duration10d = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start9).count() * 0;
// auto duration10e = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start9).count() * 0;
// auto duration10f = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start9).count() * 0;
// auto duration10g = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start9).count() * 0;
// auto duration10h = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start9).count() * 0;
// const auto start10 = std::chrono::high_resolution_clock::now();
    // 3rd step: set residuals
    if (jacobians)
    {
        if (jacobians[0])  // jacobian w.r.t translation
        {
// const auto start10a = std::chrono::high_resolution_clock::now();
            std::fill(jacobians[0], jacobians[0] + m_nResiduals*3, 0.0);
// duration10a += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start10a).count();
// const auto start10b = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < m_nCorrespond_adam2joints; i++)
            {
                if (targetPts[5 * i] != 0 || targetPts[5 * i + 1] != 0 || targetPts[5 * i + 2] != 0)
                // if (!m_targetPts.block<3,1>(5 * i, 0).isZero(0))
                {
                    for (int r = 0; r < 3; r++)
                        jacobians[0][(3*i+r)*3+r] = targetPtsWeight[3*i+r];
                }
            }
// duration10b += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start10b).count();
            // // 2-for equivalent
            // // 1st loop
            // // Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdt(jacobians[0], m_nResiduals, 3);
            // Matrix<double, Dynamic, Dynamic, RowMajor> drdt(m_nResiduals, 3);
            // drdt.setZero();
            // for (int i = 0; i < m_nCorrespond_adam2joints; i++)
            //     if (!m_targetPts.block<3,1>(5 * i, 0).isZero(0))
            //         drdt.block<3,3>(3 * i, 0).setIdentity();
            // // 2nd loop
            // drdt = m_targetPts_weight.asDiagonal() * drdt;
            // // Slower for loop equivalent of 2nd loop
            // for (int j = 0; j < 3 * m_nCorrespond_adam2joints; ++j)
            //     drdt.row(j) *= m_targetPts_weight[j];
        }

        if (jacobians[1]) // jacobian w.r.t pose
        {
// const auto start10d = std::chrono::high_resolution_clock::now();
            // Matrix<double, Dynamic, Dynamic, RowMajor> dTJdP = dTrdP.block(3 * TotalModel::NUM_JOINTS * 4, 0, 3 * TotalModel::NUM_JOINTS, 3 * TotalModel::NUM_JOINTS);
            const auto& dTJdP = dTrdP.block<3 * TotalModel::NUM_JOINTS, 3 * TotalModel::NUM_JOINTS>(3 * TotalModel::NUM_JOINTS * 4, 0);
            Map< Matrix<double, Dynamic, Dynamic, RowMajor> > dr_dPose(jacobians[1], m_nResiduals, TotalModel::NUM_JOINTS * 3);
            std::fill(jacobians[1], jacobians[1] + m_nResiduals * TotalModel::NUM_JOINTS * 3, 0.0);
// duration10d += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start10d).count();
// const auto start10e = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < m_adam.m_indices_jointConst_adamIdx.rows(); i++)
                if (!m_targetPts.block<3,1>(5 * i, 0).isZero(0))
                    dr_dPose.block<3,TotalModel::NUM_POSE_PARAMETERS>(3 * i, 0) = dTJdP.block<3,TotalModel::NUM_POSE_PARAMETERS>(3 * m_adam.m_indices_jointConst_adamIdx(i), 0);
// duration10e += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start10e).count();
// const auto start10f = std::chrono::high_resolution_clock::now();
            int offset = m_adam.m_indices_jointConst_adamIdx.rows();
            for (int i = 0; i < m_adam.m_correspond_adam2lHand_adamIdx.rows(); i++)
                if (!m_targetPts.block<3,1>(5 * (i + offset), 0).isZero(0))
                    dr_dPose.block<3,TotalModel::NUM_POSE_PARAMETERS>(3*(i + offset), 0) = dTJdP.block<3,TotalModel::NUM_POSE_PARAMETERS>(3 * m_adam.m_correspond_adam2lHand_adamIdx(i), 0);
// duration10f += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start10f).count();
// const auto start10g = std::chrono::high_resolution_clock::now();
            offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();
            for (int i = 0; i < m_adam.m_correspond_adam2rHand_adamIdx.rows(); i++)
                if (!m_targetPts.block<3,1>(5 * (i + offset), 0).isZero(0))
                    dr_dPose.block<3,TotalModel::NUM_POSE_PARAMETERS>(3*(i + offset), 0) = dTJdP.block<3,TotalModel::NUM_POSE_PARAMETERS>(3 * m_adam.m_correspond_adam2rHand_adamIdx(i), 0);
// duration10g += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start10g).count();
// const auto start10h = std::chrono::high_resolution_clock::now();
            for (int j = 0; j < 3 * m_nCorrespond_adam2joints; ++j)
            {
                const auto numberRows = dr_dPose.rows();
                for (int r = 0; r < numberRows; ++r)
                    jacobians[1][j*numberRows+r] *= targetPtsWeight[j];
            }
            // // Slow for loop equivalent
            // for (int j = 0; j < 3 * m_nCorrespond_adam2joints; ++j)
            //     dr_dPose.row(j) *= m_targetPts_weight[j];
            // // Even slower option
            // dr_dPose = m_targetPts_weight.asDiagonal() * dr_dPose;
// duration10h += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start10h).count();
        }
    }
// const auto duration10 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start10).count();
// const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
// std::cout << __FILE__ << " " << duration1 * 1e-6 << " 1\n"
//           << __FILE__ << " " << duration2 * 1e-6 << " 2\n"
//           << __FILE__ << " " << duration3 * 1e-6 << " 3\n"
//           << __FILE__ << " " << duration4 * 1e-6 << " 4\n"
//           << __FILE__ << " " << duration5 * 1e-6 << " 5\n"
//           << __FILE__ << " " << duration6 * 1e-6 << " 6\n"
//           << __FILE__ << " " << duration7 * 1e-6 << " 7\n"
//           << __FILE__ << " " << duration8 * 1e-6 << " 8\n"
//           << __FILE__ << " " << duration9 * 1e-6 << " 9\n"
//           << __FILE__ << " " << duration10* 1e-6 << " 10\n"
//           << __FILE__ << " " << duration10a* 1e-6 << " 10a\n"
//           << __FILE__ << " " << duration10b* 1e-6 << " 10b\n"
//           << __FILE__ << " " << duration10d* 1e-6 << " 10d\n"
//           << __FILE__ << " " << duration10e* 1e-6 << " 10e\n"
//           << __FILE__ << " " << duration10f* 1e-6 << " 10f\n"
//           << __FILE__ << " " << duration10g* 1e-6 << " 10g\n"
//           << __FILE__ << " " << duration10h* 1e-6 << " 10h\n"
//           << __FILE__ << " " << duration  * 1e-6 << " T\n" << std::endl;

    return true;
}
