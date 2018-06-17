#include <chrono>
#include <cstring>
#include <iostream>
#include <AdamFastCost.h>
#include <omp.h>

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
    Matrix<double, Dynamic, 3, RowMajor> outV(total_vertex.size(), 3);
    Matrix<double, Dynamic, TotalModel::NUM_POSE_PARAMETERS, RowMajor> dVdP(total_vertex.size() * 3, TotalModel::NUM_POSE_PARAMETERS);

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

    // perform LBS based on the transformation already computed
    p2t.sparse_lbs(m_Vt, total_vertex, outV.data(), jacobians? dVdP.data() : nullptr);

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
    VectorXd tempJoints(3 * m_nCorrespond_adam2joints + 3 * m_nCorrespond_adam2pts);  // predicted joint given current parameter
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
    offset += m_adam.m_correspond_adam2rHand_adamIdx.rows();
    // updating the vertex data
    tempJoints.block<3, 1>(3 * (0 + offset), 0) = outV.block<1, 3>(0, 0).transpose() + t_vec; // nose
    tempJoints.block<3, 1>(3 * (1 + offset), 0) = outV.block<1, 3>(1, 0).transpose() + t_vec; // eye
    tempJoints.block<3, 1>(3 * (2 + offset), 0) = outV.block<1, 3>(2, 0).transpose() + t_vec; // ear
    tempJoints.block<3, 1>(3 * (3 + offset), 0) = outV.block<1, 3>(3, 0).transpose() + t_vec; // eye
    tempJoints.block<3, 1>(3 * (4 + offset), 0) = outV.block<1, 3>(4, 0).transpose() + t_vec; // ear
    tempJoints.block<3, 1>(3 * (5 + offset), 0) = outV.block<1, 3>(5, 0).transpose() + t_vec; // bigtoe
    tempJoints.block<3, 1>(3 * (6 + offset), 0) = outV.block<1, 3>(6, 0).transpose() + t_vec; // smalltoe
    tempJoints.block<3, 1>(3 * (7 + offset), 0) = 0.5 * (outV.block<1, 3>(7, 0) + outV.block<1, 3>(8, 0)).transpose() + t_vec;  // heel
    tempJoints.block<3, 1>(3 * (8 + offset), 0) = outV.block<1, 3>(9, 0).transpose() + t_vec; // bigtoe
    tempJoints.block<3, 1>(3 * (9 + offset), 0) = outV.block<1, 3>(10, 0).transpose() + t_vec; // smalltoe
    tempJoints.block<3, 1>(3 * (10 + offset), 0) = 0.5 * (outV.block<1, 3>(11, 0) + outV.block<1, 3>(12, 0)).transpose() + t_vec;  // heel

    const auto* tempJointsPts = tempJoints.data();
    const auto* targetPts = m_targetPts.data();
    const auto* targetPtsWeight = m_targetPts_weight.data();
    std::fill(residuals, residuals + m_nResiduals, 0.0);
// const auto duration8 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start8).count();
// const auto start9 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < m_nCorrespond_adam2joints + m_nCorrespond_adam2pts; i++)
    {
        if (!m_targetPts.block<3,1>(5 * i, 0).isZero(0))
        {
            for (int r = 0; r < 3 ; r++)
            {
                const int baseIndex = 3 * i + r;
                residuals[baseIndex] = (tempJointsPts[baseIndex] - targetPts[5 * i + r]) * targetPtsWeight[i];
            }
            // Vectorized (slower) equivalent
            // Map< VectorXd > res(residuals, m_nResiduals);
            // res.block<3,1>(3 * i, 0) = (tempJoints.block<3,1>(3 * i, 0) - m_targetPts.block<3,1>(5 * i, 0)).cwiseProduct(targetPtsWeight.block<3,1>(3 * i, 0));
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
            for (int i = 0; i < m_nCorrespond_adam2joints + m_nCorrespond_adam2pts; i++)
            {
                if (targetPts[5 * i] != 0 || targetPts[5 * i + 1] != 0 || targetPts[5 * i + 2] != 0)
                // if (!m_targetPts.block<3,1>(5 * i, 0).isZero(0))
                {
                    for (int r = 0; r < 3; r++)
                        jacobians[0][(3*i+r)*3+r] = targetPtsWeight[i];
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
            // drdt = targetPtsWeight.asDiagonal() * drdt;
            // // Slower for loop equivalent of 2nd loop
            // for (int j = 0; j < 3 * m_nCorrespond_adam2joints; ++j)
            //     drdt.row(j) *= targetPtsWeight[j];
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
                    dr_dPose.block<3,TotalModel::NUM_POSE_PARAMETERS>(3 * i, 0) = targetPtsWeight[i] * dTJdP.block<3,TotalModel::NUM_POSE_PARAMETERS>(3 * m_adam.m_indices_jointConst_adamIdx(i), 0);
// duration10e += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start10e).count();
// const auto start10f = std::chrono::high_resolution_clock::now();
            int offset = m_adam.m_indices_jointConst_adamIdx.rows();
            for (int i = 0; i < m_adam.m_correspond_adam2lHand_adamIdx.rows(); i++)
                if (!m_targetPts.block<3,1>(5 * (i + offset), 0).isZero(0))
                    dr_dPose.block<3,TotalModel::NUM_POSE_PARAMETERS>(3*(i + offset), 0) = targetPtsWeight[i + offset] * dTJdP.block<3,TotalModel::NUM_POSE_PARAMETERS>(3 * m_adam.m_correspond_adam2lHand_adamIdx(i), 0);
// duration10f += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start10f).count();
// const auto start10g = std::chrono::high_resolution_clock::now();
            offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();
            for (int i = 0; i < m_adam.m_correspond_adam2rHand_adamIdx.rows(); i++)
                if (!m_targetPts.block<3,1>(5 * (i + offset), 0).isZero(0))
                    dr_dPose.block<3,TotalModel::NUM_POSE_PARAMETERS>(3*(i + offset), 0) = targetPtsWeight[i + offset] * dTJdP.block<3,TotalModel::NUM_POSE_PARAMETERS>(3 * m_adam.m_correspond_adam2rHand_adamIdx(i), 0);
// duration10g += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start10g).count();
// const auto start10h = std::chrono::high_resolution_clock::now();
            offset += m_adam.m_correspond_adam2rHand_adamIdx.rows();
            if (!m_targetPts.block<3,1>(5 * (0 + offset), 0).isZero(0)) dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(3*(0 + offset), 0) = targetPtsWeight[0 + offset] * dVdP.block<3, TotalModel::NUM_POSE_PARAMETERS>(3 * 0, 0);
            else dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(3*(0 + offset), 0).setZero();
            if (!m_targetPts.block<3,1>(5 * (1 + offset), 0).isZero(0)) dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(3*(1 + offset), 0) = targetPtsWeight[1 + offset] * dVdP.block<3, TotalModel::NUM_POSE_PARAMETERS>(3 * 1, 0);
            else dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(3*(1 + offset), 0).setZero();
            if (!m_targetPts.block<3,1>(5 * (2 + offset), 0).isZero(0)) dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(3*(2 + offset), 0) = targetPtsWeight[2 + offset] * dVdP.block<3, TotalModel::NUM_POSE_PARAMETERS>(3 * 2, 0);
            else dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(3*(2 + offset), 0).setZero();
            if (!m_targetPts.block<3,1>(5 * (3 + offset), 0).isZero(0)) dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(3*(3 + offset), 0) = targetPtsWeight[3 + offset] * dVdP.block<3, TotalModel::NUM_POSE_PARAMETERS>(3 * 3, 0);
            else dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(3*(3 + offset), 0).setZero();
            if (!m_targetPts.block<3,1>(5 * (4 + offset), 0).isZero(0)) dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(3*(4 + offset), 0) = targetPtsWeight[4 + offset] * dVdP.block<3, TotalModel::NUM_POSE_PARAMETERS>(3 * 4, 0);
            else dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(3*(4 + offset), 0).setZero();
            if (!m_targetPts.block<3,1>(5 * (5 + offset), 0).isZero(0)) dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(3*(5 + offset), 0) = targetPtsWeight[5 + offset] * 0.5 * (dVdP.block<3, TotalModel::NUM_POSE_PARAMETERS>(3 * 5, 0) + dVdP.block<3, TotalModel::NUM_POSE_PARAMETERS>(3 * 6, 0));
            else dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(3*(5 + offset), 0).setZero();
            if (!m_targetPts.block<3,1>(5 * (6 + offset), 0).isZero(0)) dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(3*(6 + offset), 0) = targetPtsWeight[6 + offset] * dVdP.block<3, TotalModel::NUM_POSE_PARAMETERS>(3 * 7, 0);
            else dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(3*(6 + offset), 0).setZero();
            if (!m_targetPts.block<3,1>(5 * (7 + offset), 0).isZero(0)) dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(3*(7 + offset), 0) = targetPtsWeight[7 + offset] * dVdP.block<3, TotalModel::NUM_POSE_PARAMETERS>(3 * 8, 0);
            else dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(3*(7 + offset), 0).setZero();
            if (!m_targetPts.block<3,1>(5 * (8 + offset), 0).isZero(0)) dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(3*(8 + offset), 0) = targetPtsWeight[8 + offset] * 0.5 * (dVdP.block<3, TotalModel::NUM_POSE_PARAMETERS>(3 * 9, 0) + dVdP.block<3, TotalModel::NUM_POSE_PARAMETERS>(3 * 10, 0));
            else dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(3*(8 + offset), 0).setZero();
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

void ComputedTrdc(const double* dTrdJ_data, const double* dJdc_data, double* dTrdc_data,
                  const std::array<std::vector<int>, TotalModel::NUM_JOINTS>& parentIndexes)
{
    // const Eigen::Map<const Eigen::Matrix<double, 5 * 3 * TotalModel::NUM_JOINTS, 3 * TotalModel::NUM_JOINTS, Eigen::RowMajor>> dTrdJ(dTrdJ_data);
    // const Eigen::Map<const Eigen::Matrix<double, 3 * TotalModel::NUM_JOINTS, TotalModel::NUM_SHAPE_COEFFICIENTS, Eigen::RowMajor>> dJdc(dJdc_data);
    // Eigen::Map<Eigen::Matrix<double, 5 * 3 * TotalModel::NUM_JOINTS, TotalModel::NUM_SHAPE_COEFFICIENTS, Eigen::RowMajor>> dTrdc(dTrdc_data);
    const int ncol = 3 * TotalModel::NUM_JOINTS;
    const int ncol_out = TotalModel::NUM_SHAPE_COEFFICIENTS;
    std::fill(dTrdc_data, dTrdc_data + 5 * 3 * TotalModel::NUM_JOINTS * TotalModel::NUM_SHAPE_COEFFICIENTS, 0);
    for (int i = 0; i < TotalModel::NUM_JOINTS; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            // 12 rows to take care of
            // only row 12 * i + 4 * j + 3 is non-zero
            const auto* dTrdJ_row = dTrdJ_data + (12 * i + 4 * j + 3) * ncol;
            auto* dTrdc_row = dTrdc_data + (12 * i + 4 * j + 3) * ncol_out;
            // 3 rows to take care of
            // only row 12 * i + 4 * j + 3 is non-zero
            const auto* dTrdJ_row2 = dTrdJ_data + 4 * 3 * TotalModel::NUM_JOINTS * 3 * TotalModel::NUM_JOINTS + (3 * i + j) * ncol;
            auto* dTrdc_row2 = dTrdc_data + 4 * 3 * TotalModel::NUM_JOINTS * TotalModel::NUM_SHAPE_COEFFICIENTS + (3 * i + j) * ncol_out;
            for (auto& ipar: parentIndexes[i])
            {
                const auto ipar3 = 3 * ipar;
                for(int c = 0; c < ncol_out; c++)
                {
                    // 12 rows to take care of
                    dTrdc_row[c] += dTrdJ_row[ipar3] * dJdc_data[ipar3 * ncol_out + c]
                                  + dTrdJ_row[ipar3 + 1] * dJdc_data[(ipar3 + 1) * ncol_out + c]
                                  + dTrdJ_row[ipar3 + 2] * dJdc_data[(ipar3 + 2) * ncol_out + c];
                    // 3 rows to take care of
                    dTrdc_row2[c] += dTrdJ_row2[ipar3] * dJdc_data[ipar3 * ncol_out + c]
                                   + dTrdJ_row2[ipar3 + 1] * dJdc_data[(ipar3 + 1) * ncol_out + c]
                                   + dTrdJ_row2[ipar3 + 2] * dJdc_data[(ipar3 + 2) * ncol_out + c];
                }
            }
        }
    }
}

const int AdamFullCost::DEFINED_INNER_CONSTRAINTS;

bool AdamFullCost::Evaluate(double const* const* parameters,
        double* residuals,
        double** jacobians) const
{
// const auto start = std::chrono::high_resolution_clock::now();
// const auto start_iter = std::chrono::high_resolution_clock::now();
    using namespace Eigen;
    typedef double T;
    const T* t = parameters[0];
    const T* p_eulers = parameters[1];
    const T* c = parameters[2];
    const T* face_coeff = fit_face_exp? parameters[3]: nullptr;

    Map< const Vector3d > t_vec(t);
    Map< const Matrix<double, Dynamic, 1> > c_bodyshape(c, TotalModel::NUM_SHAPE_COEFFICIENTS);

    // 0st step: Compute all the current joints
    Matrix<double, TotalModel::NUM_JOINTS, 3, RowMajor> J;
    Map< Matrix<double, Dynamic, 1> > J_vec(J.data(), TotalModel::NUM_JOINTS * 3);
    // Vector form faster than below loop
    J_vec = fit_data_.adam.J_mu_ + fit_data_.adam.dJdc_ * c_bodyshape;
    // const auto* jMuPtr = fit_data_.adam.J_mu_.data();
    // const auto* dJdcPtr = fit_data_.adam.dJdc_.data();
    // const auto dJdcCols = fit_data_.adam.dJdc_.cols();
    // // Note: Eigen::Matrix<double, Eigen::Dynamic, NUM_SHAPE_COEFFICIENTS, Eigen::RowMajor> dJdc_;
    // for (auto i = 0u; i < TotalModel::NUM_JOINTS * 3; i++)
    // {
    //     J_vec[i] = jMuPtr[i];
    //     const auto rowIndex = i*dJdcCols;
    //     for (auto j = 0u; j < TotalModel::NUM_SHAPE_COEFFICIENTS; j++)
    //     {
    //         J_vec[i] += dJdcPtr[rowIndex + j] * c[j];
    //     }
    // }

    // 1st step: forward kinematics
    const int num_t = (TotalModel::NUM_JOINTS) * 3 * 5;  // transform 3 * 4 + joint location 3 * 1

    // Matrix<double, Dynamic, 3 * TotalModel::NUM_JOINTS, RowMajor> old_dTrdP(num_t, 3 * TotalModel::NUM_JOINTS);
    // Matrix<double, Dynamic, 3 * TotalModel::NUM_JOINTS, RowMajor> old_dTrdJ(num_t, 3 * TotalModel::NUM_JOINTS);
    // old_dTrdP.setZero(); old_dTrdJ.setZero();
    // VectorXd old_transforms_joint(3 * TotalModel::NUM_JOINTS * 4 + 3 * TotalModel::NUM_JOINTS); // the first part is transform, the second part is outJoint
    // ceres::AutoDiffCostFunction<smpl::PoseToTransformsNoLR_Eulers_adamModel,
    // (TotalModel::NUM_JOINTS) * 3 * 4 + 3 * TotalModel::NUM_JOINTS,
    // (TotalModel::NUM_JOINTS) * 3,
    // (TotalModel::NUM_JOINTS) * 3> old_p2t(new smpl::PoseToTransformsNoLR_Eulers_adamModel(fit_data_.adam));
    // const double* old_p2t_parameters[2] = { p_eulers, J.data() };
    // double* old_p2t_residuals = old_transforms_joint.data();
    // double* old_p2t_jacobians[2] = { old_dTrdP.data(), old_dTrdJ.data() };
    // old_p2t.Evaluate(old_p2t_parameters, old_p2t_residuals, old_p2t_jacobians);

    Matrix<double, Dynamic, 3 * TotalModel::NUM_JOINTS, RowMajor> dTrdP(num_t, 3 * TotalModel::NUM_JOINTS);
    Matrix<double, Dynamic, 3 * TotalModel::NUM_JOINTS, RowMajor> dTrdJ(num_t, 3 * TotalModel::NUM_JOINTS);

    VectorXd transforms_joint(3 * TotalModel::NUM_JOINTS * 4 + 3 * TotalModel::NUM_JOINTS);
    const double* p2t_parameters[2] = { p_eulers, J.data() };
    double* p2t_residuals = transforms_joint.data();
    double* p2t_jacobians[2] = { dTrdP.data(), jacobians && jacobians[2]? dTrdJ.data(): nullptr };

    smpl::PoseToTransform_AdamFull_withDiff p2t(fit_data_.adam, parentIndexes, rigid_body);
// const auto duration_iter = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_iter).count();
// const auto start_FK = std::chrono::high_resolution_clock::now();
    p2t.Evaluate(p2t_parameters, p2t_residuals, jacobians ? p2t_jacobians : nullptr );
// const auto duration_FK = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_FK).count();

    // std::cout << "J" << std::endl;
    // std::cout << "max diff: " << (old_transforms_joint - transforms_joint).maxCoeff() << std::endl;
    // std::cout << "min diff: " << (old_transforms_joint - transforms_joint).minCoeff() << std::endl;
    // std::cout << "dJdP" << std::endl;
    // std::cout << "max diff: " << (old_dTrdP - dTrdP).maxCoeff() << std::endl;
    // std::cout << "min diff: " << (old_dTrdP - dTrdP).minCoeff() << std::endl;
    // std::cout << "dJdJ" << std::endl;
    // std::cout << "max diff: " << (old_dTrdJ - dTrdJ).maxCoeff() << std::endl;
    // std::cout << "min diff: " << (old_dTrdJ - dTrdJ).minCoeff() << std::endl;

// const auto start_transJ = std::chrono::high_resolution_clock::now();
    // MatrixXdr dTJdP = dTrdP.block(3 * TotalModel::NUM_JOINTS * 4, 0, 3 * TotalModel::NUM_JOINTS, 3 * TotalModel::NUM_JOINTS);
    Map<MatrixXdr> dTJdP(dTrdP.data() + 3 * TotalModel::NUM_JOINTS * 4 * 3 * TotalModel::NUM_JOINTS, 3 * TotalModel::NUM_JOINTS, 3 * TotalModel::NUM_JOINTS);
    // The following lines are equiv to MatrixXdr dTrdc = dTrdJ * fit_data_.adam.dJdc_;
    MatrixXdr dTrdc(num_t, TotalModel::NUM_SHAPE_COEFFICIENTS);
    if (jacobians && jacobians[1]) ComputedTrdc(dTrdJ.data(), fit_data_.adam.dJdc_.data(), dTrdc.data(), parentIndexes);
    // MatrixXdr dTJdc = dTrdc.block(3 * TotalModel::NUM_JOINTS * 4, 0, 3 * TotalModel::NUM_JOINTS, TotalModel::NUM_SHAPE_COEFFICIENTS);
    Map<MatrixXdr> dTJdc(dTrdc.data() + 3 * TotalModel::NUM_JOINTS * 4 * TotalModel::NUM_SHAPE_COEFFICIENTS, 3 * TotalModel::NUM_JOINTS, TotalModel::NUM_SHAPE_COEFFICIENTS);
    VectorXd outJoint = transforms_joint.block<3 * TotalModel::NUM_JOINTS, 1>(3 * TotalModel::NUM_JOINTS * 4, 0);  // outJoint
    auto* outJointPtr = outJoint.data();
    for (auto i = 0u; i < TotalModel::NUM_JOINTS; i++) outJoint.block<3,1>(3 * i, 0) += t_vec;

    MatrixXdr outVert(total_vertex.size(), 3);
    auto* outVertPtr = outVert.data();
    Map<MatrixXdr> dVdP(dVdP_data, 3 * total_vertex.size(), TotalModel::NUM_POSE_PARAMETERS);
    Map<MatrixXdr> dVdc(dVdc_data, 3 * total_vertex.size(), TotalModel::NUM_SHAPE_COEFFICIENTS);
    Map<MatrixXdr> dVdfc(dVdfc_data, 3 * total_vertex.size(), TotalModel::NUM_EXP_BASIS_COEFFICIENTS);
// const auto duration_transJ = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_transJ).count();
// const auto start_LBS = std::chrono::high_resolution_clock::now();
    if (jacobians) select_lbs(c, transforms_joint, dTrdP, dTrdc, outVert, dVdP_data, dVdc_data, face_coeff, dVdfc_data);
    else select_lbs(c, transforms_joint, outVert, face_coeff);
// const auto duration_LBS = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_LBS).count();
// const auto start_target = std::chrono::high_resolution_clock::now();
    // Next for loop is a much faster equivalent of the following vector operation
    // outVert.rowwise() += t_vec.transpose();
    for (auto i = 0u; i < total_vertex.size(); i++)
    {
        // Remember that MatrixXdr outVert(total_vertex.size(), 3);
        const auto baseIndex = 3*i;
        outVertPtr[baseIndex  ] += t[0];
        outVertPtr[baseIndex+1] += t[1];
        outVertPtr[baseIndex+2] += t[2];
    }
    std::array<double*, 3> out_data{{ outJointPtr, outVertPtr, nullptr }};
    std::array<Map<MatrixXdr>*, 3> dodP = {{ &dTJdP, &dVdP, nullptr }};  // array of reference is not allowed, only array of pointer
    std::array<Map<MatrixXdr>*, 3> dodc = {{ &dTJdc, &dVdc, nullptr }};
    std::array<Map<MatrixXdr>*, 3> dodfc = {{ nullptr, &dVdfc, nullptr }};

    // 2nd step: compute the target joints (copy from FK)
    // Arrange the Output Joints & Vertex to the order of constraints
    VectorXd tempJoints(3 * (m_nCorrespond_adam2joints + m_nCorrespond_adam2pts));
    auto* tempJointsPtr = tempJoints.data();
    Map<MatrixXdr> dOdP(dOdP_data, 3 * (m_nCorrespond_adam2joints + m_nCorrespond_adam2pts), TotalModel::NUM_POSE_PARAMETERS);
    Map<MatrixXdr> dOdc(dOdc_data, 3 * (m_nCorrespond_adam2joints + m_nCorrespond_adam2pts), TotalModel::NUM_SHAPE_COEFFICIENTS);
    Map<MatrixXdr> dOdfc(dOdfc_data, 3 * (m_nCorrespond_adam2joints + m_nCorrespond_adam2pts), TotalModel::NUM_EXP_BASIS_COEFFICIENTS);
    const auto* indicesJointConstAdamIdxPtr = fit_data_.adam.m_indices_jointConst_adamIdx.data();
    const auto* correspondAdam2lHandAdamIdx = fit_data_.adam.m_correspond_adam2lHand_adamIdx.data();
    const auto* correspondAdam2rHandAdamIdx = fit_data_.adam.m_correspond_adam2rHand_adamIdx.data();
    if (regressor_type == 0)
    {
        for (int i = 0; i < fit_data_.adam.m_indices_jointConst_adamIdx.rows(); i++)
        {
            const auto baseIndex = 3*i;
            tempJointsPtr[baseIndex  ] = outJointPtr[3*indicesJointConstAdamIdxPtr[i]  ];
            tempJointsPtr[baseIndex+1] = outJointPtr[3*indicesJointConstAdamIdxPtr[i]+1];
            tempJointsPtr[baseIndex+2] = outJointPtr[3*indicesJointConstAdamIdxPtr[i]+2];
            // // Much slower vector equivalent
            // tempJoints.block<3,1>(baseIndex, 0) = outJoint.block<3,1>(3 * fit_data_.adam.m_indices_jointConst_adamIdx(i), 0);
        }
        int offset = fit_data_.adam.m_indices_jointConst_adamIdx.rows();
        for (int i = 0; i < fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows(); i++)
        {
            const auto baseIndex = 3*(i + offset);
            tempJointsPtr[baseIndex  ] = outJointPtr[3*correspondAdam2lHandAdamIdx[i]  ];
            tempJointsPtr[baseIndex+1] = outJointPtr[3*correspondAdam2lHandAdamIdx[i]+1];
            tempJointsPtr[baseIndex+2] = outJointPtr[3*correspondAdam2lHandAdamIdx[i]+2];
            // // Much slower vector equivalent
            // tempJoints.block<3,1>(3*(i + offset), 0) = outJoint.block<3,1>(3 * fit_data_.adam.m_correspond_adam2lHand_adamIdx(i), 0);
        }
        offset += fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows();
        for (int i = 0; i < fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows(); i++)
        {
            const auto baseIndex = 3*(i + offset);
            tempJointsPtr[baseIndex  ] = outJointPtr[3*correspondAdam2rHandAdamIdx[i]  ];
            tempJointsPtr[baseIndex+1] = outJointPtr[3*correspondAdam2rHandAdamIdx[i]+1];
            tempJointsPtr[baseIndex+2] = outJointPtr[3*correspondAdam2rHandAdamIdx[i]+2];
            // // Much slower vector equivalent
            // tempJoints.block<3,1>(3*(i + offset), 0) = outJoint.block<3,1>(3 * fit_data_.adam.m_correspond_adam2rHand_adamIdx(i), 0);
        }
        offset += fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows();
        for (auto i = 0u; i < corres_vertex2targetpt.size(); i++)
        {
            // Remember that MatrixXdr outVert(total_vertex.size(), 3);
            const auto baseIndex = 3*(i + offset);
            tempJointsPtr[baseIndex  ] = outVertPtr[3*i  ];
            tempJointsPtr[baseIndex+1] = outVertPtr[3*i+1];
            tempJointsPtr[baseIndex+2] = outVertPtr[3*i+2];
            // // Much slower vector equivalent
            // tempJoints.block<3,1>(3*(i + offset), 0) = outVert.row(i).transpose();
        }

        if (jacobians)
        {
            const auto* dTJdPPtr = dTJdP.data();
            const auto* dTJdcPtr = dTJdc.data();
            if (rigid_body)
            {
                int offset = 0;
                for (int i = 0; i < fit_data_.adam.m_indices_jointConst_adamIdx.rows(); i++)
                {
                    dOdP.block<3, 3>(3 * (i + offset), 0) = dTJdP.block<3, 3>(3 * indicesJointConstAdamIdxPtr[i], 0);
                }
                offset += fit_data_.adam.m_indices_jointConst_adamIdx.rows();
                for (int i = 0; i < fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows(); i++)
                {
                    dOdP.block<3, 3>(3 * (i + offset), 0) = dTJdP.block<3, 3>(3 * correspondAdam2lHandAdamIdx[i], 0);
                }
                offset += fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows();
                for (int i = 0; i < fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows(); i++)
                {
                    dOdP.block<3, 3>(3 * (i + offset), 0) = dTJdP.block<3, 3>(3 * correspondAdam2rHandAdamIdx[i], 0);
                }
                offset += fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows();
                dOdP.block(3 * offset, 0, 3 * corres_vertex2targetpt.size(), 3) = dVdP.block(0, 0, 3 * corres_vertex2targetpt.size(), 3);
            }
            else
            {
                int offset = 0;
                for (int i = 0; i < fit_data_.adam.m_indices_jointConst_adamIdx.rows(); i++)
                {
                    std::copy(dTJdPPtr + 3 * indicesJointConstAdamIdxPtr[i] * TotalModel::NUM_POSE_PARAMETERS,
                              dTJdPPtr + 3 * (indicesJointConstAdamIdxPtr[i] + 1) * TotalModel::NUM_POSE_PARAMETERS,
                              dOdP.data() + 3 * (i + offset) * TotalModel::NUM_POSE_PARAMETERS);
                }
                offset += fit_data_.adam.m_indices_jointConst_adamIdx.rows();
                for (int i = 0; i < fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows(); i++)
                {
                    std::copy(dTJdPPtr + 3 * correspondAdam2lHandAdamIdx[i] * TotalModel::NUM_POSE_PARAMETERS,
                              dTJdPPtr + 3 * (correspondAdam2lHandAdamIdx[i] + 1) * TotalModel::NUM_POSE_PARAMETERS,
                              dOdP.data() + 3 * (i + offset) * TotalModel::NUM_POSE_PARAMETERS);
                }
                offset += fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows();
                for (int i = 0; i < fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows(); i++)
                {
                    std::copy(dTJdPPtr + 3 * correspondAdam2rHandAdamIdx[i] * TotalModel::NUM_POSE_PARAMETERS,
                              dTJdPPtr + 3 * (correspondAdam2rHandAdamIdx[i] + 1) * TotalModel::NUM_POSE_PARAMETERS,
                              dOdP.data() + 3 * (i + offset) * TotalModel::NUM_POSE_PARAMETERS);
                }
                offset += fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows();
                std::copy(dVdP_data, dVdP_data + 3 * corres_vertex2targetpt.size() * TotalModel::NUM_POSE_PARAMETERS,
                          dOdP.data() + 3 * offset * TotalModel::NUM_POSE_PARAMETERS);
            }
            if (jacobians[2])
            {
                int offset = 0;
                for (int i = 0; i < fit_data_.adam.m_indices_jointConst_adamIdx.rows(); i++)
                {
                    std::copy(dTJdcPtr + 3 * indicesJointConstAdamIdxPtr[i] * TotalModel::NUM_SHAPE_COEFFICIENTS,
                              dTJdcPtr + 3 * (indicesJointConstAdamIdxPtr[i] + 1) * TotalModel::NUM_SHAPE_COEFFICIENTS,
                              dOdc.data() + 3 * (i + offset) * TotalModel::NUM_SHAPE_COEFFICIENTS);
                }
                offset += fit_data_.adam.m_indices_jointConst_adamIdx.rows();
                for (int i = 0; i < fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows(); i++)
                {
                    std::copy(dTJdcPtr + 3 * correspondAdam2lHandAdamIdx[i] * TotalModel::NUM_SHAPE_COEFFICIENTS,
                              dTJdcPtr + 3 * (correspondAdam2lHandAdamIdx[i] + 1) * TotalModel::NUM_SHAPE_COEFFICIENTS,
                          dOdc.data() + 3 * (i + offset) * TotalModel::NUM_SHAPE_COEFFICIENTS);
                }
                offset += fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows();
                for (int i = 0; i < fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows(); i++)
                {
                    std::copy(dTJdcPtr + 3 * correspondAdam2rHandAdamIdx[i] * TotalModel::NUM_SHAPE_COEFFICIENTS,
                              dTJdcPtr + 3 * (correspondAdam2rHandAdamIdx[i] + 1) * TotalModel::NUM_SHAPE_COEFFICIENTS,
                              dOdc.data() + 3 * (i + offset) * TotalModel::NUM_SHAPE_COEFFICIENTS);
                }
                offset += fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows();
                std::copy(dVdc_data, dVdc_data + 3 * corres_vertex2targetpt.size() * TotalModel::NUM_SHAPE_COEFFICIENTS,
                          dOdc.data() + 3 * offset * TotalModel::NUM_SHAPE_COEFFICIENTS);
            }

            if (fit_face_exp)
            {
                std::fill(dOdfc_data, dOdfc_data + 3 * m_nCorrespond_adam2joints * TotalModel::NUM_EXP_BASIS_COEFFICIENTS, 0.0);
                std::copy(dVdfc_data, dVdfc_data + 3 * m_nCorrespond_adam2pts * TotalModel::NUM_EXP_BASIS_COEFFICIENTS,
                          dOdfc_data + 3 * m_nCorrespond_adam2joints * TotalModel::NUM_EXP_BASIS_COEFFICIENTS);
            }
        }
    }
    else if (regressor_type == 1) // use Human 3.6M regressor
    {
        if(jacobians) SparseRegress(fit_data_.adam.m_cocoplus_reg, outVertPtr, dVdP_data, dVdc_data, tempJointsPtr, dOdP.data(), dOdc.data());
        else SparseRegress(fit_data_.adam.m_cocoplus_reg, outVertPtr, nullptr, nullptr, tempJointsPtr, nullptr, nullptr);
        out_data[2] = tempJointsPtr;
        if (jacobians)
        {
            dodP[2] = &dOdP;
            dodc[2] = &dOdc;
        }
    }
    else
    {
        assert(regressor_type == 2); // use COCO regressor
        if(jacobians) SparseRegress(fit_data_.adam.m_small_coco_reg, outVertPtr, dVdP_data, dVdc_data, tempJointsPtr, dOdP.data(), dOdc.data());
        else SparseRegress(fit_data_.adam.m_small_coco_reg, outVertPtr, nullptr, nullptr, tempJointsPtr, nullptr, nullptr);
        // SparseRegressor only set the data for body & face, we need to copy finger data from FK output
        std::copy(outJointPtr + 3 * 22, outJointPtr + 3 * 62,  tempJointsPtr + 3 * fit_data_.adam.h36m_jointConst_smcIdx.size()); // 22-42 are left hand, 42 - 62 are right hand
        // copy foot & face vertex
        for (auto i = 0u; i < corres_vertex2targetpt.size(); i++)
        {
            tempJoints[(m_nCorrespond_adam2joints + i) * 3 + 0] = outVert(corres_vertex2targetpt[i].first, 0);
            tempJoints[(m_nCorrespond_adam2joints + i) * 3 + 1] = outVert(corres_vertex2targetpt[i].first, 1);
            tempJoints[(m_nCorrespond_adam2joints + i) * 3 + 2] = outVert(corres_vertex2targetpt[i].first, 2);
        }
        out_data[2] = tempJointsPtr;
        if (jacobians)
        {
            std::copy(dTJdP.data() + 3 * 22 * TotalModel::NUM_POSE_PARAMETERS, dTJdP.data() + 3 * 62 * TotalModel::NUM_POSE_PARAMETERS,
                      dOdP_data + 3 * fit_data_.adam.h36m_jointConst_smcIdx.size() * TotalModel::NUM_POSE_PARAMETERS);
            std::copy(dTJdc.data() + 3 * 22 * TotalModel::NUM_SHAPE_COEFFICIENTS, dTJdc.data() + 3 * 62 * TotalModel::NUM_SHAPE_COEFFICIENTS,
                      dOdc_data + 3 * fit_data_.adam.h36m_jointConst_smcIdx.size() * TotalModel::NUM_SHAPE_COEFFICIENTS);
            for (auto i = 0u; i < corres_vertex2targetpt.size(); i++)
            {
                std::copy(dVdP_data + (corres_vertex2targetpt[i].first) * 3 * TotalModel::NUM_POSE_PARAMETERS,
                          dVdP_data + (corres_vertex2targetpt[i].first + 1) * 3 * TotalModel::NUM_POSE_PARAMETERS,
                          dOdP_data + (m_nCorrespond_adam2joints + i) * 3 * TotalModel::NUM_POSE_PARAMETERS);
                std::copy(dVdc_data + (corres_vertex2targetpt[i].first) * 3 * TotalModel::NUM_SHAPE_COEFFICIENTS,
                          dVdc_data + (corres_vertex2targetpt[i].first + 1) * 3 * TotalModel::NUM_SHAPE_COEFFICIENTS,
                          dOdc_data + (m_nCorrespond_adam2joints + i) * 3 * TotalModel::NUM_SHAPE_COEFFICIENTS);
            }
            dodP[2] = &dOdP;
            dodc[2] = &dOdc;
            if (fit_face_exp)
            {
                std::fill(dOdfc_data, dOdfc_data + 3 * m_nCorrespond_adam2joints * TotalModel::NUM_EXP_BASIS_COEFFICIENTS, 0.0);
                std::copy(dVdfc_data, dVdfc_data + 3 * m_nCorrespond_adam2pts * TotalModel::NUM_EXP_BASIS_COEFFICIENTS,
                          dOdfc_data + 3 * m_nCorrespond_adam2joints * TotalModel::NUM_EXP_BASIS_COEFFICIENTS);
            }
        }
    }
// const auto duration_target = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_target).count();

    // 3rd step: set residuals
// const auto start_res = std::chrono::high_resolution_clock::now();
    Map< VectorXd > res(residuals, m_nResiduals);
    const auto* targetPts = m_targetPts.data();
    const auto* targetPtsWeight = m_targetPts_weight.data();
    if (fit_data_.fit3D)  // put constrains on 3D
    {
        for(int i = 0; i < m_nCorrespond_adam2joints + m_nCorrespond_adam2pts; i++)
        {
            const int residualIndex = res_dim * i;
            const int index3 = 3*i;
            const int index5 = 5*i;
            if (targetPts[index5] == 0 && targetPts[index5 + 1] == 0 && targetPts[index5 + 2] == 0)
            {
                residuals[residualIndex  ] = 0;
                residuals[residualIndex+1] = 0;
                residuals[residualIndex+2] = 0;
                // // Vectorized (slower) equivalent
                // res.block<3,1>(res_dim * i, 0).setZero();
            }
            else
            {
                residuals[residualIndex  ] = targetPtsWeight[i] * (tempJointsPtr[index3  ] - targetPts[index5]);
                residuals[residualIndex+1] = targetPtsWeight[i] * (tempJointsPtr[index3+1] - targetPts[index5+1]);
                residuals[residualIndex+2] = targetPtsWeight[i] * (tempJointsPtr[index3+2] - targetPts[index5+2]);
                // // Vectorized (slower) equivalent
                // res.block<3,1>(res_dim * i, 0) = targetPtsWeight[i] * (tempJoints.block<3,1>(3 * i, 0) - m_targetPts.block<3,1>(5 * i, 0));
            }
        }
    }

    if (fit_data_.fit2D)
    {
        Eigen::Map< Matrix<double, Dynamic, 3, RowMajor> > jointArray(tempJointsPtr, m_nCorrespond_adam2joints + m_nCorrespond_adam2pts, 3);
        // const Eigen::Map< const Matrix<double, 3, 3, RowMajor> > K(fit_data_.K);
        const Eigen::Map< const Matrix<double, 3, 3> > K(fit_data_.K);
        const MatrixXdr jointProjection = jointArray * K;
        const auto* JP = jointProjection.data();
        for(int i = 0; i < m_nCorrespond_adam2joints + m_nCorrespond_adam2pts; i++)
        {
            if (targetPts[5 * i + 3] == 0 && targetPts[5 * i + 4] == 0)
                res.block<2,1>(res_dim * i + start_2d_dim, 0).setZero();
            else
            {
                // the following two lines are equivalent to
                // residuals[res_dim * i + start_2d_dim + 0] = (jointProjection(i, 0) / jointProjection(i, 2) - m_targetPts(5 * i + 3)) * targetPtsWeight[res_dim * i + start_2d_dim + 0];
                // residuals[res_dim * i + start_2d_dim + 1] = (jointProjection(i, 1) / jointProjection(i, 2) - m_targetPts(5 * i + 4)) * targetPtsWeight[res_dim * i + start_2d_dim + 1];
                const auto residualIndex = res_dim * i + start_2d_dim;
                const auto baseIndex = 3*i;
                residuals[residualIndex    ] = (JP[baseIndex    ] / JP[baseIndex + 2] - targetPts[5 * i + 3]) * targetPtsWeight[i];
                residuals[residualIndex + 1] = (JP[baseIndex + 1] / JP[baseIndex + 2] - targetPts[5 * i + 4]) * targetPtsWeight[i];
            }
        }
    }

    if (fit_data_.fitPAF)
    {
        const int offset = start_PAF;
        for (auto i = 0; i < num_PAF_constraint; i++)
        {
            const auto residualIndex = offset + 3 * i;
            if (fit_data_.PAF.col(i).isZero(0))
            {
                residuals[residualIndex] = residuals[residualIndex + 1] = residuals[residualIndex + 2] = 0.0;
            }
            else
            {
                const auto& pafConnectionI = PAF_connection[i];
                const std::array<double, 3> AB{{
                    out_data[pafConnectionI[2]][3 * pafConnectionI[3] + 0] - out_data[pafConnectionI[0]][3 * pafConnectionI[1] + 0],
                    out_data[pafConnectionI[2]][3 * pafConnectionI[3] + 1] - out_data[pafConnectionI[0]][3 * pafConnectionI[1] + 1],
                    out_data[pafConnectionI[2]][3 * pafConnectionI[3] + 2] - out_data[pafConnectionI[0]][3 * pafConnectionI[1] + 2],
                }};
                const auto length = sqrt(AB[0] * AB[0] + AB[1] * AB[1] + AB[2] * AB[2]);
                residuals[residualIndex    ] = (AB[0] / length - fit_data_.PAF(0, i)) * PAF_weight[i];
                residuals[residualIndex + 1] = (AB[1] / length - fit_data_.PAF(1, i)) * PAF_weight[i];
                residuals[residualIndex + 2] = (AB[2] / length - fit_data_.PAF(2, i)) * PAF_weight[i];
            }
        }
    }

    if (fit_data_.inner_weight[0] > 0)
    {
        // the first defined inner constraints: should not bend (adam joint 6 should be close to the middle of central hip and neck)
        residuals[start_inner    ] = (outJointPtr[3 * 6    ] - 0.5 * (outJointPtr[0] + outJointPtr[3 * 12 + 0])) * fit_data_.inner_weight[0];
        residuals[start_inner + 1] = (outJointPtr[3 * 6 + 1] - 0.5 * (outJointPtr[1] + outJointPtr[3 * 12 + 1])) * fit_data_.inner_weight[0];
        residuals[start_inner + 2] = (outJointPtr[3 * 6 + 2] - 0.5 * (outJointPtr[2] + outJointPtr[3 * 12 + 2])) * fit_data_.inner_weight[0];
    }
    else
    {
        residuals[start_inner] = residuals[start_inner + 1] = residuals[start_inner + 2] = 0.0;
    }
// const auto duration_res = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_res).count();

    // 4th step: set jacobians
// const auto start_jacob = std::chrono::high_resolution_clock::now();
// auto start_jacob0 = start_jacob;
// auto start_jacob1 = start_jacob;
// auto start_jacob2 = start_jacob;
// auto start_jacob3 = start_jacob;
// auto duration_jacob0 = 0.;
// auto duration_jacob1 = 0.;
// auto duration_jacob2 = 0.;
// auto duration_jacob3 = 0.;
    if (jacobians)
    {
// start_jacob0 = std::chrono::high_resolution_clock::now();
        if (jacobians[0])
        {
            auto* jac0 = jacobians[0];
            const auto jac0Cols = 3;
            // Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdt(jac0, m_nResiduals, jac0Cols);

            if (fit_data_.fit3D)
            {
                const auto maximumIndex = m_nCorrespond_adam2joints + m_nCorrespond_adam2pts;
                // Set to 0
                std::fill(jac0,
                          jac0 + res_dim * maximumIndex * jac0Cols,
                          0.0);
                for (int i = 0; i < maximumIndex; i++)
                {
                    // Set identity if required
                    const int index5 = 5*i;
                    if (targetPts[index5] != 0 || targetPts[index5 + 1] != 0 || targetPts[index5 + 2] != 0)
                    {
                        const auto baseIndex = res_dim * i;
                        for (auto col = 0 ; col < jac0Cols ; col++)
                            jac0[baseIndex*jac0Cols + jac0Cols*col + col] = targetPtsWeight[i];
                    }

                }
                // // Vectorized (slower) equivalent of above code
                // for (int i = 0; i < m_nCorrespond_adam2joints + m_nCorrespond_adam2pts; i++)
                // {
                //     if (targetPts[5 * i] == 0 && targetPts[5 * i + 1] == 0 && targetPts[5 * i + 2] == 0)
                //         drdt.block<3,3>(res_dim * i, 0).setZero();
                //     else
                //         drdt.block<3,3>(res_dim * i, 0) = m_targetPts_weight[i] * Matrix<double, 3, 3>::Identity();
                // }
            }

            if (fit_data_.fit2D)
            {
                Eigen::Map< Matrix<double, Dynamic, 3, RowMajor> > jointArray(tempJointsPtr, m_nCorrespond_adam2joints + m_nCorrespond_adam2pts, 3);
                Matrix<double, Dynamic, Dynamic, RowMajor> dJdt(3, 3);
                dJdt.setIdentity();
                for (int i = 0; i < m_nCorrespond_adam2joints + m_nCorrespond_adam2pts; i++)
                {
                    const int index5 = 5*i;
                    if (targetPts[index5 + 3] == 0 && targetPts[index5 + 4] == 0)
                    {
                        // drdt.block<2,3>(res_dim * i + start_2d_dim, 0).setZero();
                        const auto jacIndex = jac0Cols * (res_dim * i + start_2d_dim);
                        std::fill(jac0 + jacIndex,
                                  jac0 + jacIndex + inner_dim[0]*jac0Cols, 0.0);
                    }
                    else
                    {
                        projection_Derivative(jac0, dJdt.data(), jac0Cols, (double*)(jointArray.data() + 3 * i),
                                              fit_data_.K, res_dim * i + start_2d_dim, 0, targetPtsWeight[i]);
                    }
                }
            }

            if (fit_data_.fitPAF)
            {
                // drdt.block<3 * num_PAF_constraint, 3>(start_PAF, 0).setZero();
                std::fill(jac0 + jac0Cols * start_PAF, jac0 + jac0Cols * start_PAF + 9 * num_PAF_constraint, 0.0);
                // Note by Gines: The 2 above lines are not equivalent (at least generically) to each other. Is this a bug?
            }

            // inner constraint 1
            // drdt.block(start_inner, 0, inner_dim[0], 3).setZero();
            std::fill(jac0 + jac0Cols * start_inner,
                      jac0 + jac0Cols * start_inner + inner_dim[0]*jac0Cols, 0.0);
        }

// duration_jacob0 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_jacob0).count();
// start_jacob2 = std::chrono::high_resolution_clock::now();
        if (jacobians[1])
        {
            auto* jac1 = jacobians[1];
            const auto jac1Cols = TotalModel::NUM_JOINTS * 3;
            Map< Matrix<double, Dynamic, Dynamic, RowMajor> > dr_dPose(jac1, m_nResiduals, jac1Cols);
            if (fit_data_.fit3D)
            {
                for (int i = 0; i < m_nCorrespond_adam2joints + m_nCorrespond_adam2pts; i++)
                {
                    if (targetPts[5 * i] == 0 && targetPts[5 * i + 1] == 0 && targetPts[5 * i + 2] == 0)
                    {
                        std::fill(jac1 + res_dim * i * TotalModel::NUM_POSE_PARAMETERS,
                                  jac1 + (3 + res_dim * i) * TotalModel::NUM_POSE_PARAMETERS, 0);
                        // dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(res_dim * (i + offset), 0).setZero();
                    }
                    else
                    {
                        if (rigid_body)
                            dr_dPose.block<3, 3>(res_dim * i, 0) = targetPtsWeight[i] *
                                dOdP.block<3, 3>(3 * i, 0);
                        else
                            dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(res_dim * i, 0) = targetPtsWeight[i] *
                                dOdP.block<3, TotalModel::NUM_POSE_PARAMETERS>(3 * i, 0);
                    }
                }
            }

            if (fit_data_.fit2D)
            {
                Eigen::Map< Matrix<double, Dynamic, 3, RowMajor> > jointArray(tempJointsPtr, m_nCorrespond_adam2joints + m_nCorrespond_adam2pts, 3);
                for (int i = 0; i < m_nCorrespond_adam2joints + m_nCorrespond_adam2pts; i++)
                {
                    if (targetPts[5 * i + 3] == 0 && targetPts[5 * i + 4] == 0)
                    {
                        std::fill(jac1 + (start_2d_dim + res_dim * i) * TotalModel::NUM_POSE_PARAMETERS,
                                  jac1 + (2 + start_2d_dim + res_dim * i) * TotalModel::NUM_POSE_PARAMETERS, 0);
                        // dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(res_dim * (i + offset), 0).setZero();
                    }
                    else
                        projection_Derivative(jac1, dOdP.data(), jac1Cols, (double*)(jointArray.data() + 3 * i), fit_data_.K,
                                              res_dim * i + start_2d_dim, 3 * i, targetPtsWeight[i]);
                }
            }

            if (fit_data_.fitPAF)
            {
                const int offset = start_PAF;
                for (auto i = 0; i < num_PAF_constraint; i++)
                {
                    if (fit_data_.PAF.col(i).isZero(0))
                    {
                        // dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(offset + 3 * i, 0).setZero();
                        std::fill(jac1 + (offset + 3 * i) * TotalModel::NUM_POSE_PARAMETERS,
                                  jac1 + (offset + 3 * i + 3) * TotalModel::NUM_POSE_PARAMETERS, 0);
                        continue;
                    }
                    const std::array<double, 3> AB{{
                        out_data[PAF_connection[i][2]][3 * PAF_connection[i][3] + 0] - out_data[PAF_connection[i][0]][3 * PAF_connection[i][1] + 0],
                        out_data[PAF_connection[i][2]][3 * PAF_connection[i][3] + 1] - out_data[PAF_connection[i][0]][3 * PAF_connection[i][1] + 1],
                        out_data[PAF_connection[i][2]][3 * PAF_connection[i][3] + 2] - out_data[PAF_connection[i][0]][3 * PAF_connection[i][1] + 2],
                    }};
                    const auto length2 = AB[0] * AB[0] + AB[1] * AB[1] + AB[2] * AB[2];
                    const auto length = sqrt(length2);
                    const Eigen::Map< const Matrix<double, 3, 1> > AB_vec(AB.data());
                    const Eigen::Matrix<double, 3, 3, RowMajor> dudJ = Eigen::Matrix<double, 3, 3>::Identity() / length - AB_vec * AB_vec.transpose() / length2 / length;
                    if (PAF_connection[i][0] == 0 && PAF_connection[i][2] == 0)
                    {
                        std::fill(jac1 + (offset + 3 * i) * TotalModel::NUM_POSE_PARAMETERS, jac1 + (offset + 3 * i + 3) * TotalModel::NUM_POSE_PARAMETERS, 0);
                        const double* dudJ_data = dudJ.data();
                        double* drdp_row0 = jac1 + (offset + 3 * i) * TotalModel::NUM_POSE_PARAMETERS;
                        double* drdp_row1 = jac1 + (offset + 3 * i + 1) * TotalModel::NUM_POSE_PARAMETERS;
                        double* drdp_row2 = jac1 + (offset + 3 * i + 2) * TotalModel::NUM_POSE_PARAMETERS;
                        {
                            const double* dJdP_row0 = dTJdP.data() + (3 * PAF_connection[i][3]) * TotalModel::NUM_POSE_PARAMETERS;
                            const double* dJdP_row1 = dTJdP.data() + (3 * PAF_connection[i][3] + 1) * TotalModel::NUM_POSE_PARAMETERS;
                            const double* dJdP_row2 = dTJdP.data() + (3 * PAF_connection[i][3] + 2) * TotalModel::NUM_POSE_PARAMETERS;
                            for(auto& ipar: parentIndexes[PAF_connection[i][3]])
                            {
                                drdp_row0[3 * ipar + 0] += PAF_weight[i] * (dudJ_data[0] * dJdP_row0[3 * ipar + 0] + dudJ_data[1] * dJdP_row1[3 * ipar + 0] + dudJ_data[2] * dJdP_row2[3 * ipar + 0]);
                                drdp_row0[3 * ipar + 1] += PAF_weight[i] * (dudJ_data[0] * dJdP_row0[3 * ipar + 1] + dudJ_data[1] * dJdP_row1[3 * ipar + 1] + dudJ_data[2] * dJdP_row2[3 * ipar + 1]);
                                drdp_row0[3 * ipar + 2] += PAF_weight[i] * (dudJ_data[0] * dJdP_row0[3 * ipar + 2] + dudJ_data[1] * dJdP_row1[3 * ipar + 2] + dudJ_data[2] * dJdP_row2[3 * ipar + 2]);
                                drdp_row1[3 * ipar + 0] += PAF_weight[i] * (dudJ_data[3] * dJdP_row0[3 * ipar + 0] + dudJ_data[4] * dJdP_row1[3 * ipar + 0] + dudJ_data[5] * dJdP_row2[3 * ipar + 0]);
                                drdp_row1[3 * ipar + 1] += PAF_weight[i] * (dudJ_data[3] * dJdP_row0[3 * ipar + 1] + dudJ_data[4] * dJdP_row1[3 * ipar + 1] + dudJ_data[5] * dJdP_row2[3 * ipar + 1]);
                                drdp_row1[3 * ipar + 2] += PAF_weight[i] * (dudJ_data[3] * dJdP_row0[3 * ipar + 2] + dudJ_data[4] * dJdP_row1[3 * ipar + 2] + dudJ_data[5] * dJdP_row2[3 * ipar + 2]);
                                drdp_row2[3 * ipar + 0] += PAF_weight[i] * (dudJ_data[6] * dJdP_row0[3 * ipar + 0] + dudJ_data[7] * dJdP_row1[3 * ipar + 0] + dudJ_data[8] * dJdP_row2[3 * ipar + 0]);
                                drdp_row2[3 * ipar + 1] += PAF_weight[i] * (dudJ_data[6] * dJdP_row0[3 * ipar + 1] + dudJ_data[7] * dJdP_row1[3 * ipar + 1] + dudJ_data[8] * dJdP_row2[3 * ipar + 1]);
                                drdp_row2[3 * ipar + 2] += PAF_weight[i] * (dudJ_data[6] * dJdP_row0[3 * ipar + 2] + dudJ_data[7] * dJdP_row1[3 * ipar + 2] + dudJ_data[8] * dJdP_row2[3 * ipar + 2]);
                            }
                        }
                        {
                            const double* dJdP_row0 = dTJdP.data() + (3 * PAF_connection[i][1]) * TotalModel::NUM_POSE_PARAMETERS;
                            const double* dJdP_row1 = dTJdP.data() + (3 * PAF_connection[i][1] + 1) * TotalModel::NUM_POSE_PARAMETERS;
                            const double* dJdP_row2 = dTJdP.data() + (3 * PAF_connection[i][1] + 2) * TotalModel::NUM_POSE_PARAMETERS;
                            for(auto& ipar: parentIndexes[PAF_connection[i][1]])
                            {
                                drdp_row0[3 * ipar + 0] -= PAF_weight[i] * (dudJ_data[0] * dJdP_row0[3 * ipar + 0] + dudJ_data[1] * dJdP_row1[3 * ipar + 0] + dudJ_data[2] * dJdP_row2[3 * ipar + 0]);
                                drdp_row0[3 * ipar + 1] -= PAF_weight[i] * (dudJ_data[0] * dJdP_row0[3 * ipar + 1] + dudJ_data[1] * dJdP_row1[3 * ipar + 1] + dudJ_data[2] * dJdP_row2[3 * ipar + 1]);
                                drdp_row0[3 * ipar + 2] -= PAF_weight[i] * (dudJ_data[0] * dJdP_row0[3 * ipar + 2] + dudJ_data[1] * dJdP_row1[3 * ipar + 2] + dudJ_data[2] * dJdP_row2[3 * ipar + 2]);
                                drdp_row1[3 * ipar + 0] -= PAF_weight[i] * (dudJ_data[3] * dJdP_row0[3 * ipar + 0] + dudJ_data[4] * dJdP_row1[3 * ipar + 0] + dudJ_data[5] * dJdP_row2[3 * ipar + 0]);
                                drdp_row1[3 * ipar + 1] -= PAF_weight[i] * (dudJ_data[3] * dJdP_row0[3 * ipar + 1] + dudJ_data[4] * dJdP_row1[3 * ipar + 1] + dudJ_data[5] * dJdP_row2[3 * ipar + 1]);
                                drdp_row1[3 * ipar + 2] -= PAF_weight[i] * (dudJ_data[3] * dJdP_row0[3 * ipar + 2] + dudJ_data[4] * dJdP_row1[3 * ipar + 2] + dudJ_data[5] * dJdP_row2[3 * ipar + 2]);
                                drdp_row2[3 * ipar + 0] -= PAF_weight[i] * (dudJ_data[6] * dJdP_row0[3 * ipar + 0] + dudJ_data[7] * dJdP_row1[3 * ipar + 0] + dudJ_data[8] * dJdP_row2[3 * ipar + 0]);
                                drdp_row2[3 * ipar + 1] -= PAF_weight[i] * (dudJ_data[6] * dJdP_row0[3 * ipar + 1] + dudJ_data[7] * dJdP_row1[3 * ipar + 1] + dudJ_data[8] * dJdP_row2[3 * ipar + 1]);
                                drdp_row2[3 * ipar + 2] -= PAF_weight[i] * (dudJ_data[6] * dJdP_row0[3 * ipar + 2] + dudJ_data[7] * dJdP_row1[3 * ipar + 2] + dudJ_data[8] * dJdP_row2[3 * ipar + 2]);
                            }
                        }
                        // for(auto& ipar: parentIndexes[PAF_connection[i][3]])
                        //     dr_dPose.block(offset + 3 * i, 3 * ipar, 3, 3) += PAF_weight[i] * dudJ * dodP[PAF_connection[i][2]]->block(3 * PAF_connection[i][3], 3 * ipar, 3, 3);
                        // for(auto& ipar: parentIndexes[PAF_connection[i][1]])
                        //     dr_dPose.block(offset + 3 * i, 3 * ipar, 3, 3) -= PAF_weight[i] * dudJ * dodP[PAF_connection[i][0]]->block(3 * PAF_connection[i][1], 3 * ipar, 3, 3);
                    }
                    else
                    {
                        // slow
                        dr_dPose.block<3, TotalModel::NUM_POSE_PARAMETERS>(offset + 3 * i, 0) = PAF_weight[i] * dudJ *
                            ( dodP[PAF_connection[i][2]]->block(3 * PAF_connection[i][3], 0, 3, TotalModel::NUM_POSE_PARAMETERS) -
                            dodP[PAF_connection[i][0]]->block(3 * PAF_connection[i][1], 0, 3, TotalModel::NUM_POSE_PARAMETERS) );
                    }
                }
            }

            if (fit_data_.inner_weight[0] > 0)
            {
                // the first defined inner constraints: should not bend (adam joint 6 should be close to the middle of central hip and neck)
                dr_dPose.block(start_inner, 0, inner_dim[0], TotalModel::NUM_POSE_PARAMETERS) =
                    fit_data_.inner_weight[0] * (dTJdP.block<3, TotalModel::NUM_POSE_PARAMETERS>(6 * 3, 0) -
                    0.5 * (dTJdP.block<3, TotalModel::NUM_POSE_PARAMETERS>(0 * 3, 0) + dTJdP.block<3, TotalModel::NUM_POSE_PARAMETERS>(12 * 3, 0)));
            }
            else
            {
                for (auto j = (unsigned int)start_inner ; j < start_inner+inner_dim[0] ; j++)
                {
                    std::fill(jac1 + j*jac1Cols,
                              jac1 + j*jac1Cols + TotalModel::NUM_POSE_PARAMETERS, 0.0);
                }
                // // Vectorized (slower) equivalent of above code
                // dr_dPose.block(start_inner, 0, inner_dim[0], TotalModel::NUM_POSE_PARAMETERS).setZero();
            }

            if (rigid_body)
            {
                for (auto j = 0 ; j < m_nResiduals ; j++)
                {
                    std::fill(jac1 + j*jac1Cols + 3,
                              jac1 + j*jac1Cols + TotalModel::NUM_POSE_PARAMETERS, 0.0);
                }
                // // Vectorized (slower) equivalent of above code
                // dr_dPose.block(0, 3, m_nResiduals, TotalModel::NUM_POSE_PARAMETERS - 3).setZero();
            }

            if (freeze_missing)
            {
                // used for the demo, when a joint target (smc) is missing, freeze the parent joint angle.
                for (int ic = 0; ic < fit_data_.adam.m_indices_jointConst_adamIdx.rows(); ic++)
                {
                    const int smcjoint = fit_data_.adam.m_indices_jointConst_smcIdx(ic);
                    if (smcjoint == 4 || smcjoint == 5 || smcjoint == 10 || smcjoint == 11)
                    {
                        const int adam_index = fit_data_.adam.m_parent[indicesJointConstAdamIdxPtr[ic]];
                        if (fit_data_.bodyJoints.col(smcjoint).isZero(0))
                        {
                            for (auto j = 0 ; j < m_nResiduals ; j++)
                            {
                                std::fill(jac1 + j*jac1Cols + 3 * adam_index,
                                          jac1 + j*jac1Cols + 3 * adam_index + 3, 0.0);
                            }
                            // // Vectorized (slower) equivalent of above code
                            // dr_dPose.block(0, 3 * adam_index, m_nResiduals, 3).setZero();
                        }
                    }
                }

                for (int ic = 0; ic < fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows(); ic++)
                {
                    const int smcjoint = fit_data_.adam.m_correspond_adam2lHand_lHandIdx(ic);
                    const int adam_index = fit_data_.adam.m_parent[correspondAdam2lHandAdamIdx[ic]];
                    if (fit_data_.lHandJoints.col(smcjoint).isZero(0))
                    {
                        for (auto j = 0 ; j < m_nResiduals ; j++)
                        {
                            std::fill(jac1 + j*jac1Cols + 3 * adam_index,
                                      jac1 + j*jac1Cols + 3 * adam_index + 3, 0.0);
                        }
                        // // Vectorized (slower) equivalent of above code
                        // dr_dPose.block(0, 3 * adam_index, m_nResiduals, 3).setZero();
                    }
                }

                for (int ic = 0; ic < fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows(); ic++)
                {
                    const int smcjoint = fit_data_.adam.m_correspond_adam2rHand_rHandIdx(ic);
                    const int adam_index = fit_data_.adam.m_parent[correspondAdam2rHandAdamIdx[ic]];
                    if (fit_data_.rHandJoints.col(smcjoint).isZero(0))
                    {
                        for (auto j = 0 ; j < m_nResiduals ; j++)
                        {
                            std::fill(jac1 + j*jac1Cols + 3 * adam_index,
                                      jac1 + j*jac1Cols + 3 * adam_index + 3, 0.0);
                        }
                        // // Vectorized (slower) equivalent of above code
                        // dr_dPose.block(0, 3 * adam_index, m_nResiduals, 3).setZero();
                    }
                }
            }
        }

        if (rigid_body)
        {
            if (jacobians[2])
                std::fill(jacobians[2], jacobians[2] + m_nResiduals * TotalModel::NUM_SHAPE_COEFFICIENTS, 0);
            if (fit_face_exp && jacobians[3])
                std::fill(jacobians[3], jacobians[3] + m_nResiduals * TotalModel::NUM_EXP_BASIS_COEFFICIENTS, 0);
            return true;
        }

// duration_jacob1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_jacob1).count();
// start_jacob2 = std::chrono::high_resolution_clock::now();
        if (jacobians[2])
        {
            Map< Matrix<double, Dynamic, Dynamic, RowMajor> > dr_dCoeff(jacobians[2], m_nResiduals, TotalModel::NUM_SHAPE_COEFFICIENTS);

            if (fit_data_.fit3D)
            {
                for (int i = 0; i < m_nCorrespond_adam2joints + m_nCorrespond_adam2pts; i++)
                {
                    if (targetPts[5 * i] == 0 && targetPts[5 * i + 1] == 0 && targetPts[5 * i + 2] == 0)
                    {
                        std::fill(dr_dCoeff.data() + res_dim * i * TotalModel::NUM_SHAPE_COEFFICIENTS,
                                  dr_dCoeff.data() + (3 + res_dim * i) * TotalModel::NUM_SHAPE_COEFFICIENTS, 0);
                    }
                    else
                        dr_dCoeff.block<3, TotalModel::NUM_SHAPE_COEFFICIENTS>(res_dim * i, 0) = targetPtsWeight[i] *
                            dOdc.block<3, TotalModel::NUM_SHAPE_COEFFICIENTS>(3 * i, 0);
                }
            }

            if (fit_data_.fit2D)
            {
                Eigen::Map< Matrix<double, Dynamic, 3, RowMajor> > jointArray(tempJointsPtr, m_nCorrespond_adam2joints + m_nCorrespond_adam2pts, 3);
                for (int i = 0; i < m_nCorrespond_adam2joints + m_nCorrespond_adam2pts; i++)
                {
                    if (targetPts[5 * i + 3] == 0 && targetPts[5 * i + 4] == 0)
                    {
                        std::fill(dr_dCoeff.data() + (start_2d_dim + res_dim * i) * TotalModel::NUM_SHAPE_COEFFICIENTS,
                                  dr_dCoeff.data() + (2 + start_2d_dim + res_dim * i) * TotalModel::NUM_SHAPE_COEFFICIENTS, 0);
                    }
                    else projection_Derivative(dr_dCoeff.data(), dOdc.data(), dr_dCoeff.cols(), (double*)(jointArray.data() + 3 * i), fit_data_.K,
                                               res_dim * i + start_2d_dim, 3 * i, targetPtsWeight[i]);
                }
            }

            if (fit_data_.fitPAF)
            {
                const int offset = start_PAF;
                for (auto i = 0; i < num_PAF_constraint; i++)
                {
                    if (fit_data_.PAF.col(i).isZero(0))
                    {
                        std::fill(dr_dCoeff.data() + (offset + 3 * i) * TotalModel::NUM_SHAPE_COEFFICIENTS,
                                  dr_dCoeff.data() + (offset + 3 * i + 3) * TotalModel::NUM_SHAPE_COEFFICIENTS, 0);
                        // dr_dCoeff.block(offset + 3 * i, 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS).setZero();
                        continue;
                    }
                    const std::array<double, 3> AB{{
                        out_data[PAF_connection[i][2]][3 * PAF_connection[i][3] + 0] - out_data[PAF_connection[i][0]][3 * PAF_connection[i][1] + 0],
                        out_data[PAF_connection[i][2]][3 * PAF_connection[i][3] + 1] - out_data[PAF_connection[i][0]][3 * PAF_connection[i][1] + 1],
                        out_data[PAF_connection[i][2]][3 * PAF_connection[i][3] + 2] - out_data[PAF_connection[i][0]][3 * PAF_connection[i][1] + 2],
                    }};
                    const auto length2 = AB[0] * AB[0] + AB[1] * AB[1] + AB[2] * AB[2];
                    const auto length = sqrt(length2);
                    const Eigen::Map< const Matrix<double, 3, 1> > AB_vec(AB.data());
                    const Eigen::Matrix<double, 3, 3, RowMajor> dudJ = Eigen::Matrix<double, 3, 3>::Identity() / length - AB_vec * AB_vec.transpose() / length2 / length;
                    dr_dCoeff.block<3, TotalModel::NUM_SHAPE_COEFFICIENTS>(offset + 3 * i, 0) = PAF_weight[i] * dudJ *
                        ( dodc[PAF_connection[i][2]]->block(3 * PAF_connection[i][3], 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS) -
                        dodc[PAF_connection[i][0]]->block(3 * PAF_connection[i][1], 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS) );
                }
            }

            if (fit_data_.inner_weight[0] > 0)
            {
                // the first defined inner constraints: should not bend (adam joint 6 should be close to the middle of central hip and neck)
                dr_dCoeff.block(start_inner, 0, inner_dim[0], TotalModel::NUM_SHAPE_COEFFICIENTS) =
                    fit_data_.inner_weight[0] * (dTJdc.block<3, TotalModel::NUM_SHAPE_COEFFICIENTS>(6 * 3, 0) -
                    0.5 * (dTJdc.block<3, TotalModel::NUM_SHAPE_COEFFICIENTS>(0 * 3, 0) + dTJdc.block<3, TotalModel::NUM_SHAPE_COEFFICIENTS>(12 * 3, 0)));
            }
            else
                dr_dCoeff.block(start_inner, 0, inner_dim[0], TotalModel::NUM_SHAPE_COEFFICIENTS).setZero();
        }
// duration_jacob2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_jacob2).count();
// start_jacob3 = std::chrono::high_resolution_clock::now();

        if (fit_face_exp && jacobians[3])
        {
            auto* jac3 = jacobians[3];
            const auto jac3Cols = TotalModel::NUM_EXP_BASIS_COEFFICIENTS;
            Map< Matrix<double, Dynamic, Dynamic, RowMajor> > dr_dfc(jac3, m_nResiduals, jac3Cols);

            if (fit_data_.fit3D)
            {
                for (int i = 0; i < m_nCorrespond_adam2joints + m_nCorrespond_adam2pts; i++)
                {
                    if (targetPts[5 * i] == 0 && targetPts[5 * i + 1] == 0 && targetPts[5 * i + 2] == 0)
                    {
                        std::fill(jac3 + res_dim * i * TotalModel::NUM_EXP_BASIS_COEFFICIENTS,
                                  jac3 + (3 + res_dim * i) * TotalModel::NUM_EXP_BASIS_COEFFICIENTS, 0);
                    }
                    else dr_dfc.block<3, TotalModel::NUM_EXP_BASIS_COEFFICIENTS>(res_dim * i, 0) = targetPtsWeight[i] *
                        dOdfc.block<3, TotalModel::NUM_EXP_BASIS_COEFFICIENTS>(3 * i, 0);
                }
            }

            if (fit_data_.fit2D)
            {
                Eigen::Map< Matrix<double, Dynamic, 3, RowMajor> > jointArray(tempJointsPtr, m_nCorrespond_adam2joints + m_nCorrespond_adam2pts, 3);
                for (int i = 0; i < m_nCorrespond_adam2joints + m_nCorrespond_adam2pts; i++)
                {
                    if (targetPts[5 * i + 3] == 0 && targetPts[5 * i + 4] == 0)
                    {
                        std::fill(jac3 + (start_2d_dim + res_dim * i) * TotalModel::NUM_EXP_BASIS_COEFFICIENTS,
                                  jac3 + (2 + start_2d_dim + res_dim * i) * TotalModel::NUM_EXP_BASIS_COEFFICIENTS, 0);
                    }
                    else projection_Derivative(jac3, dOdfc.data(), jac3Cols, (double*)(jointArray.data() + 3 * i), fit_data_.K,
                                               res_dim * i + start_2d_dim, 3 * i, targetPtsWeight[i]);
                }
            }

            if (fit_data_.fitPAF)
            {
                const int offset = start_PAF;
                for (auto i = 0; i < num_PAF_constraint; i++)
                {
                    if (fit_data_.PAF.col(i).isZero(0))
                    {
                        std::fill(jac3 + (offset + 3 * i) * TotalModel::NUM_EXP_BASIS_COEFFICIENTS,
                                  jac3 + (offset + 3 * i + 3) * TotalModel::NUM_EXP_BASIS_COEFFICIENTS, 0);
                        // dr_dCoeff.block(offset + 3 * i, 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS).setZero();
                        continue;
                    }
                    if (out_data[PAF_connection[i][0]] == nullptr || out_data[PAF_connection[i][2]] == nullptr)
                        continue;
                    const auto& pafConnectionI = PAF_connection[i];
                    const std::array<double, 3> AB{{
                        out_data[pafConnectionI[2]][3 * pafConnectionI[3]    ] - out_data[pafConnectionI[0]][3 * pafConnectionI[1]],
                        out_data[pafConnectionI[2]][3 * pafConnectionI[3] + 1] - out_data[pafConnectionI[0]][3 * pafConnectionI[1] + 1],
                        out_data[pafConnectionI[2]][3 * pafConnectionI[3] + 2] - out_data[pafConnectionI[0]][3 * pafConnectionI[1] + 2],
                    }};
                    const auto length2 = AB[0] * AB[0] + AB[1] * AB[1] + AB[2] * AB[2];
                    const auto length = sqrt(length2);
                    const Eigen::Map< const Matrix<double, 3, 1> > AB_vec(AB.data());
                    const Eigen::Matrix<double, 3, 3, RowMajor> dudJ = Eigen::Matrix<double, 3, 3>::Identity() / length - AB_vec * AB_vec.transpose() / length2 / length;
                    dr_dfc.block<3, TotalModel::NUM_EXP_BASIS_COEFFICIENTS>(offset + 3 * i, 0) = PAF_weight[i] * dudJ *
                        ( dodfc[PAF_connection[i][2]]->block(3 * PAF_connection[i][3], 0, 3, TotalModel::NUM_EXP_BASIS_COEFFICIENTS) -
                        dodfc[PAF_connection[i][0]]->block(3 * PAF_connection[i][1], 0, 3, TotalModel::NUM_EXP_BASIS_COEFFICIENTS) );
                }
            }

            std::fill(jac3 + start_inner*jac3Cols,
                      jac3 + (start_inner+inner_dim[0])*jac3Cols, 0.0);
            // // Vectorized (slower) equivalent of above code
            // dr_dfc.block(start_inner, 0, inner_dim[0], jac3Cols).setZero();
        }
// duration_jacob3 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_jacob3).count();
    }
// const auto duration_jacob = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_jacob).count();
// const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
// std::cout << "iter     " << duration_iter * 1e-6 << "\n"
//           << "FK       " << duration_FK * 1e-6 << "\n"
//           << "transJ   " << duration_transJ * 1e-6 << "\n"
//           << "LBS      " << duration_LBS * 1e-6 << "\n"
//           << "target   " << duration_target * 1e-6 << "\n"
//           << "Residual " << duration_res * 1e-6 << "\n"
//           << "Jacobian " << duration_jacob * 1e-6 << "\n"
//           << "    Jac0 " << duration_jacob0 * 1e-6 << "\n"
//           << "    Jac1 " << duration_jacob1 * 1e-6 << "\n"
//           << "    Jac2 " << duration_jacob2 * 1e-6 << "\n"
//           << "    Jac3 " << duration_jacob3 * 1e-6 << "\n"
//           << "Total    " << duration * 1e-6 << std::endl;
// std::cout << "------------------------------------------\n";
    return true;
}

// LBS with Jacobian
void AdamFullCost::select_lbs(
    const double* c,
    const Eigen::VectorXd& T,  // transformation
    const MatrixXdr &dTdP,
    const MatrixXdr &dTdc,
    MatrixXdr &outVert,
    double* dVdP_data,    //output
    double* dVdc_data,
    const double* face_coeff,
    double* dVdfc_data
) const
{
    // read adam model and corres_vertex2targetpt from the class member
    using namespace Eigen;
    assert((unsigned int)outVert.rows() == total_vertex.size());
    std::fill(dVdc_data, dVdc_data + 3 * total_vertex.size() * TotalModel::NUM_SHAPE_COEFFICIENTS, 0); // dVdc.setZero();
    std::fill(dVdP_data, dVdP_data + 3 * total_vertex.size() * TotalModel::NUM_POSE_PARAMETERS, 0); // dVdP.setZero();
    const double* dTdc_data = dTdc.data();
    const double* dTdP_data = dTdP.data();
    const double* dV0dc_data = fit_data_.adam.m_shapespace_u.data();
    const double* meanshape_data = fit_data_.adam.m_meanshape.data();
    const double* face_basis_data = fit_data_.adam.m_dVdFaceEx.data();
    if (fit_face_exp) assert(face_coeff != nullptr && dVdfc_data != nullptr);

    for (auto i = 0u; i < total_vertex.size(); i++)
    {
        const int idv = total_vertex[i];
        // compute the default vertex, v0 is a column vector
        // The following lines are equivalent to
        // MatrixXd v0 = fit_data_.adam.m_meanshape.block(3 * idv, 0, 3, 1) + fit_data_.adam.m_shapespace_u.block(3 * idv, 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS) * c_bodyshape;
        MatrixXd v0(3, 1);
        auto* v0_data = v0.data();
        v0_data[0] = meanshape_data[3 * idv + 0];
        v0_data[1] = meanshape_data[3 * idv + 1];
        v0_data[2] = meanshape_data[3 * idv + 2];
        const int nrow = fit_data_.adam.m_shapespace_u.rows();
        for(int ic = 0; ic < TotalModel::NUM_SHAPE_COEFFICIENTS; ic++)
        {
            v0_data[0] += dV0dc_data[ic * nrow + 3 * idv + 0] * c[ic];
            v0_data[1] += dV0dc_data[ic * nrow + 3 * idv + 1] * c[ic];
            v0_data[2] += dV0dc_data[ic * nrow + 3 * idv + 2] * c[ic];
        }
        if (fit_face_exp)
        {
            const int nrow = fit_data_.adam.m_dVdFaceEx.rows();
            for(int ic = 0; ic < TotalModel::NUM_EXP_BASIS_COEFFICIENTS; ic++)
            {
                v0_data[0] += face_basis_data[ic * nrow + 3 * idv + 0] * face_coeff[ic];
                v0_data[1] += face_basis_data[ic * nrow + 3 * idv + 1] * face_coeff[ic];
                v0_data[2] += face_basis_data[ic * nrow + 3 * idv + 2] * face_coeff[ic];
            }
        }

        auto* outVrow_data = outVert.data() + 3 * i;
        outVrow_data[0] = outVrow_data[1] = outVrow_data[2] = 0;
        for (int idj = 0; idj < TotalModel::NUM_JOINTS; idj++)
        {
            const double w = fit_data_.adam.m_blendW(idv, idj);
            if (w)
            {
                const auto* const Trow_data = T.data() + 12 * idj;
                outVrow_data[0] += w * (Trow_data[0] * v0_data[0] + Trow_data[1] * v0_data[1] + Trow_data[2] * v0_data[2] + Trow_data[3]);
                outVrow_data[1] += w * (Trow_data[4] * v0_data[0] + Trow_data[5] * v0_data[1] + Trow_data[6] * v0_data[2] + Trow_data[7]);
                outVrow_data[2] += w * (Trow_data[8] * v0_data[0] + Trow_data[9] * v0_data[1] + Trow_data[10] * v0_data[2] + Trow_data[11]);

                const int ncol = TotalModel::NUM_POSE_PARAMETERS;
                double* dVdP_row0 = dVdP_data + (i * 3) * TotalModel::NUM_POSE_PARAMETERS;
                double* dVdP_row1 = dVdP_data + (i * 3 + 1) * TotalModel::NUM_POSE_PARAMETERS;
                double* dVdP_row2 = dVdP_data + (i * 3 + 2) * TotalModel::NUM_POSE_PARAMETERS;
                const double* dTdP_base = dTdP_data + idj * 12 * TotalModel::NUM_POSE_PARAMETERS;
                for (auto j = 0u; j < parentIndexes[idj].size(); j++)
                {
                    const int idp = parentIndexes[idj][j];
                    // The following lines are equiv to
                    // dVdP(i * 3 + 0, 3 * idp + 0) +=
                    //     w * (v0_data[0] * dTdP(idj * 3 * 4 + 0 * 4 + 0, 3 * idp + 0) + v0_data[1] * dTdP(idj * 3 * 4 + 0 * 4 + 1, 3 * idp + 0) + v0_data[2] * dTdP(idj * 3 * 4 + 0 * 4 + 2, 3 * idp + 0) + dTdP(idj * 12 + 0 * 4 + 3, 3 * idp + 0));
                    // dVdP(i * 3 + 1, 3 * idp + 0) +=
                    //     w * (v0_data[0] * dTdP(idj * 3 * 4 + 1 * 4 + 0, 3 * idp + 0) + v0_data[1] * dTdP(idj * 3 * 4 + 1 * 4 + 1, 3 * idp + 0) + v0_data[2] * dTdP(idj * 3 * 4 + 1 * 4 + 2, 3 * idp + 0) + dTdP(idj * 12 + 1 * 4 + 3, 3 * idp + 0));
                    // dVdP(i * 3 + 2, 3 * idp + 0) +=
                    //     w * (v0_data[0] * dTdP(idj * 3 * 4 + 2 * 4 + 0, 3 * idp + 0) + v0_data[1] * dTdP(idj * 3 * 4 + 2 * 4 + 1, 3 * idp + 0) + v0_data[2] * dTdP(idj * 3 * 4 + 2 * 4 + 2, 3 * idp + 0) + dTdP(idj * 12 + 2 * 4 + 3, 3 * idp + 0));
                    // dVdP(i * 3 + 0, 3 * idp + 1) +=
                    //     w * (v0_data[0] * dTdP(idj * 3 * 4 + 0 * 4 + 0, 3 * idp + 1) + v0_data[1] * dTdP(idj * 3 * 4 + 0 * 4 + 1, 3 * idp + 1) + v0_data[2] * dTdP(idj * 3 * 4 + 0 * 4 + 2, 3 * idp + 1) + dTdP(idj * 12 + 0 * 4 + 3, 3 * idp + 1));
                    // dVdP(i * 3 + 1, 3 * idp + 1) +=
                    //     w * (v0_data[0] * dTdP(idj * 3 * 4 + 1 * 4 + 0, 3 * idp + 1) + v0_data[1] * dTdP(idj * 3 * 4 + 1 * 4 + 1, 3 * idp + 1) + v0_data[2] * dTdP(idj * 3 * 4 + 1 * 4 + 2, 3 * idp + 1) + dTdP(idj * 12 + 1 * 4 + 3, 3 * idp + 1));
                    // dVdP(i * 3 + 2, 3 * idp + 1) +=
                    //     w * (v0_data[0] * dTdP(idj * 3 * 4 + 2 * 4 + 0, 3 * idp + 1) + v0_data[1] * dTdP(idj * 3 * 4 + 2 * 4 + 1, 3 * idp + 1) + v0_data[2] * dTdP(idj * 3 * 4 + 2 * 4 + 2, 3 * idp + 1) + dTdP(idj * 12 + 2 * 4 + 3, 3 * idp + 1));
                    // dVdP(i * 3 + 0, 3 * idp + 2) +=
                    //     w * (v0_data[0] * dTdP(idj * 3 * 4 + 0 * 4 + 0, 3 * idp + 2) + v0_data[1] * dTdP(idj * 3 * 4 + 0 * 4 + 1, 3 * idp + 2) + v0_data[2] * dTdP(idj * 3 * 4 + 0 * 4 + 2, 3 * idp + 2) + dTdP(idj * 12 + 0 * 4 + 3, 3 * idp + 2));
                    // dVdP(i * 3 + 1, 3 * idp + 2) +=
                    //     w * (v0_data[0] * dTdP(idj * 3 * 4 + 1 * 4 + 0, 3 * idp + 2) + v0_data[1] * dTdP(idj * 3 * 4 + 1 * 4 + 1, 3 * idp + 2) + v0_data[2] * dTdP(idj * 3 * 4 + 1 * 4 + 2, 3 * idp + 2) + dTdP(idj * 12 + 1 * 4 + 3, 3 * idp + 2));
                    // dVdP(i * 3 + 2, 3 * idp + 2) +=
                    //     w * (v0_data[0] * dTdP(idj * 3 * 4 + 2 * 4 + 0, 3 * idp + 2) + v0_data[1] * dTdP(idj * 3 * 4 + 2 * 4 + 1, 3 * idp + 2) + v0_data[2] * dTdP(idj * 3 * 4 + 2 * 4 + 2, 3 * idp + 2) + dTdP(idj * 12 + 2 * 4 + 3, 3 * idp + 2));
                    dVdP_row0[3 * idp + 0] += w * (v0_data[0] * dTdP_base[(0 * 4 + 0) * ncol + 3 * idp + 0] + v0_data[1] * dTdP_base[(0 * 4 + 1) * ncol + 3 * idp + 0] + v0_data[2] * dTdP_base[(0 * 4 + 2) * ncol + 3 * idp + 0] + dTdP_base[(0 * 4 + 3) * ncol + 3 * idp + 0]);
                    dVdP_row1[3 * idp + 0] += w * (v0_data[0] * dTdP_base[(1 * 4 + 0) * ncol + 3 * idp + 0] + v0_data[1] * dTdP_base[(1 * 4 + 1) * ncol + 3 * idp + 0] + v0_data[2] * dTdP_base[(1 * 4 + 2) * ncol + 3 * idp + 0] + dTdP_base[(1 * 4 + 3) * ncol + 3 * idp + 0]);
                    dVdP_row2[3 * idp + 0] += w * (v0_data[0] * dTdP_base[(2 * 4 + 0) * ncol + 3 * idp + 0] + v0_data[1] * dTdP_base[(2 * 4 + 1) * ncol + 3 * idp + 0] + v0_data[2] * dTdP_base[(2 * 4 + 2) * ncol + 3 * idp + 0] + dTdP_base[(2 * 4 + 3) * ncol + 3 * idp + 0]);
                    dVdP_row0[3 * idp + 1] += w * (v0_data[0] * dTdP_base[(0 * 4 + 0) * ncol + 3 * idp + 1] + v0_data[1] * dTdP_base[(0 * 4 + 1) * ncol + 3 * idp + 1] + v0_data[2] * dTdP_base[(0 * 4 + 2) * ncol + 3 * idp + 1] + dTdP_base[(0 * 4 + 3) * ncol + 3 * idp + 1]);
                    dVdP_row1[3 * idp + 1] += w * (v0_data[0] * dTdP_base[(1 * 4 + 0) * ncol + 3 * idp + 1] + v0_data[1] * dTdP_base[(1 * 4 + 1) * ncol + 3 * idp + 1] + v0_data[2] * dTdP_base[(1 * 4 + 2) * ncol + 3 * idp + 1] + dTdP_base[(1 * 4 + 3) * ncol + 3 * idp + 1]);
                    dVdP_row2[3 * idp + 1] += w * (v0_data[0] * dTdP_base[(2 * 4 + 0) * ncol + 3 * idp + 1] + v0_data[1] * dTdP_base[(2 * 4 + 1) * ncol + 3 * idp + 1] + v0_data[2] * dTdP_base[(2 * 4 + 2) * ncol + 3 * idp + 1] + dTdP_base[(2 * 4 + 3) * ncol + 3 * idp + 1]);
                    dVdP_row0[3 * idp + 2] += w * (v0_data[0] * dTdP_base[(0 * 4 + 0) * ncol + 3 * idp + 2] + v0_data[1] * dTdP_base[(0 * 4 + 1) * ncol + 3 * idp + 2] + v0_data[2] * dTdP_base[(0 * 4 + 2) * ncol + 3 * idp + 2] + dTdP_base[(0 * 4 + 3) * ncol + 3 * idp + 2]);
                    dVdP_row1[3 * idp + 2] += w * (v0_data[0] * dTdP_base[(1 * 4 + 0) * ncol + 3 * idp + 2] + v0_data[1] * dTdP_base[(1 * 4 + 1) * ncol + 3 * idp + 2] + v0_data[2] * dTdP_base[(1 * 4 + 2) * ncol + 3 * idp + 2] + dTdP_base[(1 * 4 + 3) * ncol + 3 * idp + 2]);
                    dVdP_row2[3 * idp + 2] += w * (v0_data[0] * dTdP_base[(2 * 4 + 0) * ncol + 3 * idp + 2] + v0_data[1] * dTdP_base[(2 * 4 + 1) * ncol + 3 * idp + 2] + v0_data[2] * dTdP_base[(2 * 4 + 2) * ncol + 3 * idp + 2] + dTdP_base[(2 * 4 + 3) * ncol + 3 * idp + 2]);
                }

                // Note that dV0dc is column major
                const int ncolc = TotalModel::NUM_SHAPE_COEFFICIENTS;
                double* dVdc_row0 = dVdc_data + (i * 3 + 0) * ncolc;
                double* dVdc_row1 = dVdc_data + (i * 3 + 1) * ncolc;
                double* dVdc_row2 = dVdc_data + (i * 3 + 2) * ncolc;
                const double* dTdc_row0 = dTdc_data + (idj * 12 + 0 * 4 + 3) * ncolc;
                const double* dTdc_row1 = dTdc_data + (idj * 12 + 1 * 4 + 3) * ncolc;
                const double* dTdc_row2 = dTdc_data + (idj * 12 + 2 * 4 + 3) * ncolc;
                for (int idc = 0; idc < TotalModel::NUM_SHAPE_COEFFICIENTS; idc++) {
                    // The following lines are equiv to
                    // dVdc(i * 3 + 0, idc) +=
                    //     w * (dV0dc(idv * 3 + 0, idc) * Trow_data[0 * 4 + 0] + dV0dc(idv * 3 + 1, idc) * Trow_data[0 * 4 + 1] + dV0dc(idv * 3 + 2, idc) * Trow_data[0 * 4 + 2] + dTdc(idj * 12 + 0 * 4 + 3, idc));
                    // dVdc(i * 3 + 1, idc) +=
                    //     w * (dV0dc(idv * 3 + 0, idc) * Trow_data[1 * 4 + 0] + dV0dc(idv * 3 + 1, idc) * Trow_data[1 * 4 + 1] + dV0dc(idv * 3 + 2, idc) * Trow_data[1 * 4 + 2] + dTdc(idj * 12 + 1 * 4 + 3, idc));
                    // dVdc(i * 3 + 2, idc) +=
                    //     w * (dV0dc(idv * 3 + 0, idc) * Trow_data[2 * 4 + 0] + dV0dc(idv * 3 + 1, idc) * Trow_data[2 * 4 + 1] + dV0dc(idv * 3 + 2, idc) * Trow_data[2 * 4 + 2] + dTdc(idj * 12 + 2 * 4 + 3, idc));
                    dVdc_row0[idc] += w * (dV0dc_data[idc * nrow + idv * 3 + 0] * Trow_data[0 * 4 + 0] + dV0dc_data[idc * nrow + idv * 3 + 1] * Trow_data[0 * 4 + 1] + dV0dc_data[idc * nrow + idv * 3 + 2] * Trow_data[0 * 4 + 2] + dTdc_row0[idc]);
                    dVdc_row1[idc] += w * (dV0dc_data[idc * nrow + idv * 3 + 0] * Trow_data[1 * 4 + 0] + dV0dc_data[idc * nrow + idv * 3 + 1] * Trow_data[1 * 4 + 1] + dV0dc_data[idc * nrow + idv * 3 + 2] * Trow_data[1 * 4 + 2] + dTdc_row1[idc]);
                    dVdc_row2[idc] += w * (dV0dc_data[idc * nrow + idv * 3 + 0] * Trow_data[2 * 4 + 0] + dV0dc_data[idc * nrow + idv * 3 + 1] * Trow_data[2 * 4 + 1] + dV0dc_data[idc * nrow + idv * 3 + 2] * Trow_data[2 * 4 + 2] + dTdc_row2[idc]);
                }
            }
        }

        if (fit_face_exp)
        {
            const int idj = 15;
            const double w = fit_data_.adam.m_blendW(idv, idj);
            if (w)
            {
                const auto* const Trow_data = T.data() + 12 * idj;
                const int ncolc = TotalModel::NUM_EXP_BASIS_COEFFICIENTS;
                double* dVdfc_row0 = dVdfc_data + (i * 3 + 0) * ncolc;
                double* dVdfc_row1 = dVdfc_data + (i * 3 + 1) * ncolc;
                double* dVdfc_row2 = dVdfc_data + (i * 3 + 2) * ncolc;
                for (int idc = 0; idc < TotalModel::NUM_EXP_BASIS_COEFFICIENTS; idc++) {
                    dVdfc_row0[idc] = w * (face_basis_data[idc * nrow + idv * 3 + 0] * Trow_data[0 * 4 + 0] + face_basis_data[idc * nrow + idv * 3 + 1] * Trow_data[0 * 4 + 1] + face_basis_data[idc * nrow + idv * 3 + 2] * Trow_data[0 * 4 + 2]);
                    dVdfc_row1[idc] = w * (face_basis_data[idc * nrow + idv * 3 + 0] * Trow_data[1 * 4 + 0] + face_basis_data[idc * nrow + idv * 3 + 1] * Trow_data[1 * 4 + 1] + face_basis_data[idc * nrow + idv * 3 + 2] * Trow_data[1 * 4 + 2]);
                    dVdfc_row2[idc] = w * (face_basis_data[idc * nrow + idv * 3 + 0] * Trow_data[2 * 4 + 0] + face_basis_data[idc * nrow + idv * 3 + 1] * Trow_data[2 * 4 + 1] + face_basis_data[idc * nrow + idv * 3 + 2] * Trow_data[2 * 4 + 2]);
                }
            }
            else std::fill(dVdfc_data + 3 * i * TotalModel::NUM_EXP_BASIS_COEFFICIENTS,
                           dVdfc_data + 3 * (i + 1) * TotalModel::NUM_EXP_BASIS_COEFFICIENTS,
                           0.0);
        }
    }
}

// LBS w/o jacobian
void AdamFullCost::select_lbs(
    const double* c,
    const Eigen::VectorXd& T,  // transformation
    MatrixXdr &outVert,
    const double* face_coeff
) const
{
    // read adam model and total_vertex from the class member
    using namespace Eigen;
    // Map< const Matrix<double, Dynamic, 1> > c_bodyshape(c, TotalModel::NUM_SHAPE_COEFFICIENTS);
    assert((unsigned int)outVert.rows() == total_vertex.size());
    // const Eigen::MatrixXd& dV0dc = fit_data_.adam.m_shapespace_u;
    const double* dV0dc_data = fit_data_.adam.m_shapespace_u.data();
    const double* meanshape_data = fit_data_.adam.m_meanshape.data();
    const double* face_basis_data = fit_data_.adam.m_dVdFaceEx.data();
    if (fit_face_exp) assert(face_coeff != nullptr);

    for (auto i = 0u; i < total_vertex.size(); i++)
    {
        const int idv = total_vertex[i];
        // compute the default vertex, v0 is a column vector
        // The following lines are equivalent to
        // MatrixXd v0 = fit_data_.adam.m_meanshape.block(3 * idv, 0, 3, 1) + fit_data_.adam.m_shapespace_u.block(3 * idv, 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS) * c_bodyshape;
        MatrixXd v0(3, 1);
        auto* v0_data = v0.data();
        v0_data[0] = meanshape_data[3 * idv + 0];
        v0_data[1] = meanshape_data[3 * idv + 1];
        v0_data[2] = meanshape_data[3 * idv + 2];
        const int nrow = fit_data_.adam.m_shapespace_u.rows();
        for(int ic = 0; ic < TotalModel::NUM_SHAPE_COEFFICIENTS; ic++)
        {
            v0_data[0] += dV0dc_data[ic * nrow + 3 * idv + 0] * c[ic];
            v0_data[1] += dV0dc_data[ic * nrow + 3 * idv + 1] * c[ic];
            v0_data[2] += dV0dc_data[ic * nrow + 3 * idv + 2] * c[ic];
        }
        if (fit_face_exp)
        {
            const int nrow = fit_data_.adam.m_dVdFaceEx.rows();
            for(int ic = 0; ic < TotalModel::NUM_EXP_BASIS_COEFFICIENTS; ic++)
            {
                v0_data[0] += face_basis_data[ic * nrow + 3 * idv + 0] * face_coeff[ic];
                v0_data[1] += face_basis_data[ic * nrow + 3 * idv + 1] * face_coeff[ic];
                v0_data[2] += face_basis_data[ic * nrow + 3 * idv + 2] * face_coeff[ic];
            }
        }

        auto* outVrow_data = outVert.data() + 3 * i;
        outVrow_data[0] = outVrow_data[1] = outVrow_data[2] = 0;
        for (int idj = 0; idj < TotalModel::NUM_JOINTS; idj++)
        {
            const double w = fit_data_.adam.m_blendW(idv, idj);
            if (w)
            {
                const auto* const Trow_data = T.data() + 12 * idj;
                outVrow_data[0] += w * (Trow_data[0] * v0_data[0] + Trow_data[1] * v0_data[1] + Trow_data[2] * v0_data[2] + Trow_data[3]);
                outVrow_data[1] += w * (Trow_data[4] * v0_data[0] + Trow_data[5] * v0_data[1] + Trow_data[6] * v0_data[2] + Trow_data[7]);
                outVrow_data[2] += w * (Trow_data[8] * v0_data[0] + Trow_data[9] * v0_data[1] + Trow_data[10] * v0_data[2] + Trow_data[11]);
            }
        }
    }
}

void AdamFullCost::SparseRegress(const Eigen::SparseMatrix<double>& reg, const double* V_data, const double* dVdP_data, const double* dVdc_data,
                                 double* J_data, double* dJdP_data, double* dJdc_data) const
{
    const int num_J = m_nCorrespond_adam2joints;
    std::fill(J_data, J_data + 3 * num_J, 0);
    for (auto ic = 0u; ic < total_vertex.size(); ic++)
    {
        const int c = total_vertex[ic];
        for (Eigen::SparseMatrix<double>::InnerIterator it(reg, c); it; ++it)
        {
            const int r = it.row();
            auto search = map_regressor_to_constraint.find(r);
            if (search == map_regressor_to_constraint.end()) continue;  // This joint is not used for constraint
            const int ind_constraint = search->second;
            const double value = it.value();
            J_data[3 * ind_constraint + 0] += value * V_data[3 * ic + 0];
            J_data[3 * ind_constraint + 1] += value * V_data[3 * ic + 1];
            J_data[3 * ind_constraint + 2] += value * V_data[3 * ic + 2];
        }
    }

    if (dVdP_data != nullptr)  // need to pass back the correct Jacobian
    {
        assert(dVdc_data != nullptr && dJdP_data != nullptr && dJdc_data != nullptr);
        std::fill(dJdP_data, dJdP_data + 3 * num_J * TotalModel::NUM_POSE_PARAMETERS, 0.0);
        std::fill(dJdc_data, dJdc_data + 3 * num_J * TotalModel::NUM_SHAPE_COEFFICIENTS, 0.0);
        for (auto ic = 0u; ic < total_vertex.size(); ic++)
        {
            const int c = total_vertex[ic];
            for (Eigen::SparseMatrix<double>::InnerIterator it(reg, c); it; ++it)
            {
                const int r = it.row();
                auto search = map_regressor_to_constraint.find(r);
                if (search == map_regressor_to_constraint.end()) continue;  // This joint is not used for constraint
                const int ind_constraint = search->second;
                const double value = it.value();
                for (int i = 0; i < TotalModel::NUM_POSE_PARAMETERS; i++)
                {
                    dJdP_data[(3 * ind_constraint + 0) * TotalModel::NUM_POSE_PARAMETERS + i] += value * dVdP_data[(3 * ic + 0) * TotalModel::NUM_POSE_PARAMETERS + i];
                    dJdP_data[(3 * ind_constraint + 1) * TotalModel::NUM_POSE_PARAMETERS + i] += value * dVdP_data[(3 * ic + 1) * TotalModel::NUM_POSE_PARAMETERS + i];
                    dJdP_data[(3 * ind_constraint + 2) * TotalModel::NUM_POSE_PARAMETERS + i] += value * dVdP_data[(3 * ic + 2) * TotalModel::NUM_POSE_PARAMETERS + i];
                }
                for (int i = 0; i < TotalModel::NUM_SHAPE_COEFFICIENTS; i++)
                {
                    dJdc_data[(3 * ind_constraint + 0) * TotalModel::NUM_SHAPE_COEFFICIENTS + i] += value * dVdc_data[(3 * ic + 0) * TotalModel::NUM_SHAPE_COEFFICIENTS + i];
                    dJdc_data[(3 * ind_constraint + 1) * TotalModel::NUM_SHAPE_COEFFICIENTS + i] += value * dVdc_data[(3 * ic + 1) * TotalModel::NUM_SHAPE_COEFFICIENTS + i];
                    dJdc_data[(3 * ind_constraint + 2) * TotalModel::NUM_SHAPE_COEFFICIENTS + i] += value * dVdc_data[(3 * ic + 2) * TotalModel::NUM_SHAPE_COEFFICIENTS + i];
                }
            }
        }
    }
}