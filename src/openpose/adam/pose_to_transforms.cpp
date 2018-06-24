#include <chrono>
#include <iostream>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "pose_to_transforms.h"

namespace smpl
{
    bool PoseToTransformsNoLR_Eulers_adamModel_withDiff::Evaluate(double const* const* parameters,
        double* residuals,
        double** jacobians)
    {
// // t_original = [54,67] // over 100
// // t_optimized = [3.9,4.2] // over 100
// const auto start = std::chrono::high_resolution_clock::now();
// const auto reps = 100;
// for(auto asdf = 0 ; asdf < reps ; asdf++)
// {
// const auto start1 = std::chrono::high_resolution_clock::now();
        using namespace Eigen;
        const double* pose = parameters[0];
        ceres::AngleAxisToRotationMatrix(pose, (double*)MR[0].data());
        Eigen::Map< Eigen::Matrix<double, TotalModel::NUM_JOINTS, 3, Eigen::RowMajor> > outJoint(residuals);
        outJoint.row(0) = J0_.row(0);
        Matrix<double, 3, 3, RowMajor> R; // Interface with ceres
// const auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start1).count();
// const auto start2 = std::chrono::high_resolution_clock::now();
        Map< Matrix<double, TotalModel::NUM_JOINTS * 3, TotalModel::NUM_JOINTS * 3, RowMajor> > dJdP((jacobians ? jacobians[0] : nullptr));
        // Matrix<double, 9, 3 * TotalModel::NUM_JOINTS, RowMajor> dRdP((jacobians ? 9 : 1), (jacobians ? 3 * TotalModel::NUM_JOINTS : 1));
        Matrix<double, 9, 3 * TotalModel::NUM_JOINTS, RowMajor> dRdP(9, 3 * TotalModel::NUM_JOINTS);
        if (jacobians)
        {
            dJdP.block<3, TotalModel::NUM_JOINTS * 3>(0, 0).setZero();
            AngleAxisToRotationMatrix_Derivative(pose, (double*)dMRdP.at(0).data(), 0);
        }
// const auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start2).count();
// auto duration7 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
// auto duration8 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
// auto duration9 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
// auto duration10 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
// auto duration11 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
// auto duration12 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
// auto duration13 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
// auto duration14 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
// auto duration15 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
// duration7 = 0;
// duration8 = 0;
// duration9 = 0;
// duration10 = 0;
// duration11 = 0;
// duration12 = 0;
// duration13 = 0;
// duration14 = 0;
// duration15 = 0;
// const auto start5 = std::chrono::high_resolution_clock::now();
        for (int idj = 1; idj < mod_.NUM_JOINTS; idj++)
        {
// const auto start7 = std::chrono::high_resolution_clock::now();
            const int ipar = mod_.m_parent[idj];
            //std::cout << idj << " " << ipar << "\n\n";
            // ceres::AngleAxisToRotationMatrix(pose + idj * 3, R.data());

            const auto baseIndex = idj * 3;
            double angles[3] = {pose[baseIndex], pose[baseIndex+1], pose[baseIndex+2]};

            //Freezing joints here  //////////////////////////////////////////////////////
            if (idj == 10 || idj == 11) //foot ends
            {
                //R.setIdentity();
                angles[0] = 0.0;
                angles[1] = 0.0;
                angles[2] = 0.0;
            }
            if (idj == 7 || idj == 8)   //foot ankle. Restrict side movement
            {
                angles[2] = 0.0;
            }
            if (idj == 24 || idj == 27 || idj == 28 || idj == 31 || idj == 32 || idj == 35 || idj == 36 || idj == 39 || idj == 40
                || idj == 44 || idj == 47 || idj == 48 || idj == 51 || idj == 52 || idj == 55 || idj == 56 || idj == 59 || idj == 60)   //all hands
            {
                angles[0] = 0.0;
                angles[1] = 0.0;
            }
// duration7 += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start7).count();
// const auto start8 = std::chrono::high_resolution_clock::now();
            ceres::EulerAnglesToRotationMatrix(angles, 3, R.data());
// duration8 += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start8).count();
// const auto start9 = std::chrono::high_resolution_clock::now();
            MR.at(idj) = MR.at(ipar) * R;
// duration9 += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start9).count();
// const auto start10 = std::chrono::high_resolution_clock::now();
            const Matrix<double, 3, 1> offset = (J0_.row(idj) - J0_.row(ipar)).transpose();
            outJoint.row(idj) = outJoint.row(ipar) + (MR.at(ipar) * offset).transpose();
            if (jacobians)
            {
// duration10 += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start10).count();
// const auto start11 = std::chrono::high_resolution_clock::now();
                EulerAnglesToRotationMatrix_Derivative(angles, dRdP.data(), idj);

                if (idj == 10 || idj == 11) //foot ends
                    dRdP.block<9, 3>(0, 3 * idj).setZero();
                if (idj == 7 || idj == 8)   //foot ankle. Restrict side movement
                    dRdP.block<9, 1>(0, 3 * idj + 2).setZero();
                if (idj == 24 || idj == 27 || idj == 28 || idj == 31 || idj == 32 || idj == 35 || idj == 36 || idj == 39 || idj == 40
                    || idj == 44 || idj == 47 || idj == 48 || idj == 51 || idj == 52 || idj == 55 || idj == 56 || idj == 59 || idj == 60)   //all hands
                    dRdP.block<9, 2>(0, 3 * idj).setZero();
// duration11 += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start11).count();
// const auto start12 = std::chrono::high_resolution_clock::now();
                if (idj == 10 || idj == 11) //foot ends
                    dMRdP.at(idj) = dMRdP.at(ipar);
                else
                {
                    // Sparse derivative
                    SparseProductDerivative(MR.at(ipar).data(), dMRdP.at(ipar).data(), R.data(), dRdP.data(), idj, mParentIndexes[idj], dMRdP.at(idj).data());
                    // // Slower but equivalent - Dense derivative
                    // Product_Derivative(MR.at(ipar).data(), dMRdP.at(ipar).data(), R.data(), dRdP.data(), dMRdP.at(idj).data()); // Compute the product of matrix multiplication
                }

// duration12 += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start12).count();
// const auto start13 = std::chrono::high_resolution_clock::now();
                auto dMtdPIdj = dJdP.block<3, TotalModel::NUM_JOINTS * 3>(3 * idj, 0);
// duration13 += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start13).count();
// const auto start14 = std::chrono::high_resolution_clock::now();
                // Product_Derivative(NULL, dMRdP.at(ipar).data(), offset.data(), NULL, dMtdPIdj.data(), 1); // dB_data is NULL since offset is a constant
                SparseProductDerivative(dMRdP.at(ipar).data(), offset.data(), mParentIndexes[idj], dMtdPIdj.data()); // dB_data is NULL since offset is a constant
// duration14 += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start14).count();
// const auto start15 = std::chrono::high_resolution_clock::now();
                // SparseAdd(dJdP.block<3, TotalModel::NUM_JOINTS * 3>(3 * ipar, 0).data(), mParentIndexes[idj], dMtdPIdj.data());
                SparseAdd(&dJdP.data()[3 * ipar * TotalModel::NUM_JOINTS * 3], mParentIndexes[idj], dMtdPIdj.data()); // Equivalent: dJdP.block<3, TotalModel::NUM_JOINTS * 3>(3 * ipar, 0).data()
            }
// duration15 += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start15).count();
        }
// const auto duration5 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start5).count();

// std::cout << __FILE__ << " " << duration1 * 1e-6 << "\n"
//           << __FILE__ << " " << duration2 * 1e-6 << "\n"
//           << __FILE__ << " " << duration5 * 1e-6 << " " << mod_.NUM_JOINTS << "\n"
//           << __FILE__ << " \t" << duration7 * 1e-6 << " 7" << "\n"
//           << __FILE__ << " \t" << duration8 * 1e-6 << " 8" << "\n"
//           << __FILE__ << " \t" << duration9 * 1e-6 << " 9" << "\n"
//           << __FILE__ << " \t" << duration10* 1e-6 << " 10" << "\n"
//           << __FILE__ << " \t" << duration11* 1e-6 << " 11" << "\n"
//           << __FILE__ << " \t" << duration12* 1e-6 << " 12" << "\n"
//           << __FILE__ << " \t" << duration13* 1e-6 << " 13" << "\n"
//           << __FILE__ << " \t" << duration14* 1e-6 << " 14" << "\n"
//           << __FILE__ << " \t" << duration15* 1e-6 << " 15\n" << std::endl;
//           // << __FILE__ << " T: " << duration * 1e-6 << "\n" << std::endl;
// }
// const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
// std::cout << __FILE__ << " " << duration * 1e-6 / reps << "\n" << std::endl;

        for(int idj = 0; idj < mod_.NUM_JOINTS; idj++)
        {
            Eigen::Matrix<double, 3, TotalModel::NUM_POSE_PARAMETERS, Eigen::RowMajor> dtdP;
            Mt[idj] = outJoint.row(idj).transpose() - MR[idj] * J0_.row(idj).transpose();  // compute Mt for LBS
            if (jacobians)
            {
                SparseProductDerivative(dMRdP.at(idj).data(), J0_.row(idj).data(), mParentIndexes[idj], dtdP.data());
                std::copy(dJdP.data() + 3 * idj * TotalModel::NUM_POSE_PARAMETERS, dJdP.data() + 3 * (idj + 1) * TotalModel::NUM_POSE_PARAMETERS, dMtdP[idj].data());
                SparseSubtract(dtdP.data(), mParentIndexes[idj], dMtdP[idj].data());
            }
        }

        return true;
    }

    void PoseToTransformsNoLR_Eulers_adamModel_withDiff::sparse_lbs(
        const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& Vt,
        const std::vector<int>& total_vertex,
        double* outV,
        double* dVdP)
    {
        const int num_vertex = total_vertex.size();
        if (dVdP) std::fill(dVdP, dVdP + 3 * num_vertex * TotalModel::NUM_POSE_PARAMETERS, 0);
        for (auto i = 0; i < num_vertex; i++)
        {
            const int idv = total_vertex[i];
            const auto* v0_data = Vt.data() + idv * 3;
            auto* outVrow_data = outV + 3 * i;
            outVrow_data[0] = outVrow_data[1] = outVrow_data[2] = 0;
            for (int idj = 0; idj < TotalModel::NUM_JOINTS; idj++)
            {
                const double w = mod_.m_blendW(idv, idj);
                if (w)
                {
                    outVrow_data[0] += w * (MR[idj].data()[0] * v0_data[0] + MR[idj].data()[1] * v0_data[1] + MR[idj].data()[2] * v0_data[2] + Mt[idj].data()[0]);
                    outVrow_data[1] += w * (MR[idj].data()[3] * v0_data[0] + MR[idj].data()[4] * v0_data[1] + MR[idj].data()[5] * v0_data[2] + Mt[idj].data()[1]);
                    outVrow_data[2] += w * (MR[idj].data()[6] * v0_data[0] + MR[idj].data()[7] * v0_data[1] + MR[idj].data()[8] * v0_data[2] + Mt[idj].data()[2]);
                    int ncol = TotalModel::NUM_POSE_PARAMETERS;
                    double* dVdP_row0 = dVdP + (i * 3 + 0) * TotalModel::NUM_POSE_PARAMETERS;
                    double* dVdP_row1 = dVdP + (i * 3 + 1) * TotalModel::NUM_POSE_PARAMETERS;
                    double* dVdP_row2 = dVdP + (i * 3 + 2) * TotalModel::NUM_POSE_PARAMETERS;
                    if (dVdP)
                    {
                        for (auto j = 0u; j < mParentIndexes[idj].size(); j++)
                        {
                            const int idp = mParentIndexes[idj][j];
                            dVdP_row0[3 * idp + 0] += w * (v0_data[0] * dMRdP[idj].data()[(0 * 3 + 0) * ncol + 3 * idp + 0] + v0_data[1] * dMRdP[idj].data()[(0 * 3 + 1) * ncol + 3 * idp + 0] + v0_data[2] * dMRdP[idj].data()[(0 * 3 + 2) * ncol + 3 * idp + 0] + dMtdP[idj].data()[0 * ncol + 3 * idp + 0]);
                            dVdP_row1[3 * idp + 0] += w * (v0_data[0] * dMRdP[idj].data()[(1 * 3 + 0) * ncol + 3 * idp + 0] + v0_data[1] * dMRdP[idj].data()[(1 * 3 + 1) * ncol + 3 * idp + 0] + v0_data[2] * dMRdP[idj].data()[(1 * 3 + 2) * ncol + 3 * idp + 0] + dMtdP[idj].data()[1 * ncol + 3 * idp + 0]);
                            dVdP_row2[3 * idp + 0] += w * (v0_data[0] * dMRdP[idj].data()[(2 * 3 + 0) * ncol + 3 * idp + 0] + v0_data[1] * dMRdP[idj].data()[(2 * 3 + 1) * ncol + 3 * idp + 0] + v0_data[2] * dMRdP[idj].data()[(2 * 3 + 2) * ncol + 3 * idp + 0] + dMtdP[idj].data()[2 * ncol + 3 * idp + 0]);
                            dVdP_row0[3 * idp + 1] += w * (v0_data[0] * dMRdP[idj].data()[(0 * 3 + 0) * ncol + 3 * idp + 1] + v0_data[1] * dMRdP[idj].data()[(0 * 3 + 1) * ncol + 3 * idp + 1] + v0_data[2] * dMRdP[idj].data()[(0 * 3 + 2) * ncol + 3 * idp + 1] + dMtdP[idj].data()[0 * ncol + 3 * idp + 1]);
                            dVdP_row1[3 * idp + 1] += w * (v0_data[0] * dMRdP[idj].data()[(1 * 3 + 0) * ncol + 3 * idp + 1] + v0_data[1] * dMRdP[idj].data()[(1 * 3 + 1) * ncol + 3 * idp + 1] + v0_data[2] * dMRdP[idj].data()[(1 * 3 + 2) * ncol + 3 * idp + 1] + dMtdP[idj].data()[1 * ncol + 3 * idp + 1]);
                            dVdP_row2[3 * idp + 1] += w * (v0_data[0] * dMRdP[idj].data()[(2 * 3 + 0) * ncol + 3 * idp + 1] + v0_data[1] * dMRdP[idj].data()[(2 * 3 + 1) * ncol + 3 * idp + 1] + v0_data[2] * dMRdP[idj].data()[(2 * 3 + 2) * ncol + 3 * idp + 1] + dMtdP[idj].data()[2 * ncol + 3 * idp + 1]);
                            dVdP_row0[3 * idp + 2] += w * (v0_data[0] * dMRdP[idj].data()[(0 * 3 + 0) * ncol + 3 * idp + 2] + v0_data[1] * dMRdP[idj].data()[(0 * 3 + 1) * ncol + 3 * idp + 2] + v0_data[2] * dMRdP[idj].data()[(0 * 3 + 2) * ncol + 3 * idp + 2] + dMtdP[idj].data()[0 * ncol + 3 * idp + 2]);
                            dVdP_row1[3 * idp + 2] += w * (v0_data[0] * dMRdP[idj].data()[(1 * 3 + 0) * ncol + 3 * idp + 2] + v0_data[1] * dMRdP[idj].data()[(1 * 3 + 1) * ncol + 3 * idp + 2] + v0_data[2] * dMRdP[idj].data()[(1 * 3 + 2) * ncol + 3 * idp + 2] + dMtdP[idj].data()[1 * ncol + 3 * idp + 2]);
                            dVdP_row2[3 * idp + 2] += w * (v0_data[0] * dMRdP[idj].data()[(2 * 3 + 0) * ncol + 3 * idp + 2] + v0_data[1] * dMRdP[idj].data()[(2 * 3 + 1) * ncol + 3 * idp + 2] + v0_data[2] * dMRdP[idj].data()[(2 * 3 + 2) * ncol + 3 * idp + 2] + dMtdP[idj].data()[2 * ncol + 3 * idp + 2]);
                        }
                    }
                }
            }
        }
    }

    bool PoseToTransform_AdamFull_withDiff::Evaluate(double const* const* parameters,
        double* residuals,
        double** jacobians) const
    {
// const auto start = std::chrono::high_resolution_clock::now();
// const auto start1 = std::chrono::high_resolution_clock::now();
        using namespace Eigen;
        const double* pose = parameters[0];
        const double* joints = parameters[1];
        Eigen::Map< const Eigen::Matrix<double, TotalModel::NUM_JOINTS, 3, Eigen::RowMajor> > P(pose);
        Eigen::Map< const Matrix<double, TotalModel::NUM_JOINTS, 3, RowMajor> > J0(joints);
        Eigen::Map< Matrix<double, 3 * TotalModel::NUM_JOINTS, 4, RowMajor> > outT(residuals);
        Eigen::Map< Matrix<double, TotalModel::NUM_JOINTS, 3, RowMajor> > outJoint(residuals + 3 * TotalModel::NUM_JOINTS * 4);

        Map< Matrix<double, 4 * TotalModel::NUM_JOINTS * 3, TotalModel::NUM_JOINTS * 3, RowMajor> > dTrdP(jacobians? jacobians[0] : nullptr);
        Map< Matrix<double, TotalModel::NUM_JOINTS * 3, TotalModel::NUM_JOINTS * 3, RowMajor> > dJdP(jacobians? jacobians[0] + TotalModel::NUM_JOINTS * TotalModel::NUM_JOINTS * 36 : nullptr);

        Map< Matrix<double, 4 * TotalModel::NUM_JOINTS * 3, TotalModel::NUM_JOINTS * 3, RowMajor> > dTrdJ(jacobians? jacobians[1] : nullptr);
        Map< Matrix<double, TotalModel::NUM_JOINTS * 3, TotalModel::NUM_JOINTS * 3, RowMajor> > dJdJ(jacobians? jacobians[1] + TotalModel::NUM_JOINTS * TotalModel::NUM_JOINTS * 36 : nullptr);
        // fill in dTrdJ first, because it is sparse, only dMtdJ is none-0.
        if (jacobians && jacobians[1])
            std::fill(jacobians[1], jacobians[1] + 36 * TotalModel::NUM_JOINTS * TotalModel::NUM_JOINTS, 0.0);

// const auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start1).count();
// const auto start2 = std::chrono::high_resolution_clock::now();
        Matrix<double, 3, 3, RowMajor> R; // Interface with ceres
        Matrix<double, 9, 3 * TotalModel::NUM_JOINTS, RowMajor> dRdP(9, 3 * TotalModel::NUM_JOINTS);
        Matrix<double, 3, 3 * TotalModel::NUM_JOINTS, Eigen::RowMajor> dtdJ(3, 3 * TotalModel::NUM_JOINTS); // a buffer for the derivative
        Matrix<double, 3, 3 * TotalModel::NUM_JOINTS, Eigen::RowMajor> dtdJ2(3, 3 * TotalModel::NUM_JOINTS); // a buffer for the derivative

        std::vector<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> MR(TotalModel::NUM_JOINTS, Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(3, 3));

        std::vector<Eigen::Matrix<double, 9, 3 * TotalModel::NUM_JOINTS, Eigen::RowMajor>> dMRdP(TotalModel::NUM_JOINTS, Eigen::Matrix<double, 9, 3 * TotalModel::NUM_JOINTS, Eigen::RowMajor>(9, 3 * TotalModel::NUM_JOINTS));
        std::vector<Eigen::Matrix<double, 3, 3 * TotalModel::NUM_JOINTS, Eigen::RowMajor>> dMtdP(TotalModel::NUM_JOINTS, Eigen::Matrix<double, 3, 3 * TotalModel::NUM_JOINTS, Eigen::RowMajor>(3, 3 * TotalModel::NUM_JOINTS));
        std::vector<Eigen::Matrix<double, 3, 3 * TotalModel::NUM_JOINTS, Eigen::RowMajor>> dMtdJ(TotalModel::NUM_JOINTS, Eigen::Matrix<double, 3, 3 * TotalModel::NUM_JOINTS, Eigen::RowMajor>(3, 3 * TotalModel::NUM_JOINTS));

// const auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start2).count();
// const auto start3 = std::chrono::high_resolution_clock::now();
        if (m_FK_joint_list[0])
        {
            ceres::AngleAxisToRotationMatrix(pose, R.data());
            outJoint.row(0) = J0.row(0);
            MR.at(0) = R;
            outT.block<3,3>(0, 0) = MR[0];
            outT.block<3,1>(0, 3) = J0.row(0).transpose();

    // const auto duration3 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start3).count();
    // const auto start4 = std::chrono::high_resolution_clock::now();
            if (jacobians)
            {
                AngleAxisToRotationMatrix_Derivative(pose, dMRdP.at(0).data(), 0);  
                if (m_rigid_body)
                {
                    dMtdP[0].block<3, 3>(0, 0).setZero();
                    dJdP.block<3, 3>(0, 0).setZero();
                }
                else
                {
                    std::fill(dMtdP[0].data(), dMtdP[0].data() + 9 * TotalModel::NUM_JOINTS, 0.0); // dMtdP.at(0).setZero();
                    std::copy(dMtdP[0].data(), dMtdP[0].data() + 9 * TotalModel::NUM_JOINTS, dJdP.data()); // dJdP.block(0, 0, 3, TotalModel::NUM_JOINTS * 3) = dMtdP[0];
                }
                if (jacobians[1])
                {
                    std::fill(dMtdJ[0].data(), dMtdJ[0].data() + 9 * TotalModel::NUM_JOINTS, 0.0); // dMtdJ.at(0).setZero();
                    dMtdJ.at(0).block<3,3>(0, 0).setIdentity();
                    std::copy(dMtdJ[0].data(), dMtdJ[0].data() + 9 * TotalModel::NUM_JOINTS, dJdJ.data()); // dJdJ.block(0, 0, 3, TotalModel::NUM_JOINTS * 3) = dMtdJ[0];
                }
            }
        }
        else
        {
            std::fill(outJoint.data(), outJoint.data() + 3, 0.0);
            std::fill(outT.data(), outT.data() + 12, 0.0);
            if (jacobians)
            {
                std::fill(dTrdP.data(), dTrdP.data() + 12 * TotalModel::NUM_POSE_PARAMETERS, 0.0);
                std::fill(dJdP.data(), dJdP.data() + 3 * TotalModel::NUM_POSE_PARAMETERS, 0.0);
                if (jacobians[1])
                {
                    std::fill(dTrdJ.data(), dTrdJ.data() + 12 * TotalModel::NUM_POSE_PARAMETERS, 0.0);
                    std::fill(dJdJ.data(), dJdJ.data() + 3 * TotalModel::NUM_POSE_PARAMETERS, 0.0);
                }
            }
        }

// const auto duration4 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start4).count();
// const auto start5 = std::chrono::high_resolution_clock::now();
        auto* dtdJPtr = dtdJ.data();
        for (int idj = 1; idj < mod_.NUM_JOINTS; idj++)
        {
            if (!m_FK_joint_list[idj])
            {
                std::fill(outJoint.data() + 3 * idj, outJoint.data() + 3 * idj + 3, 0.0);
                std::fill(outT.data() + 12 * idj, outT.data() + 12 * idj + 12, 0.0);
                if (jacobians)
                {
                    std::fill(dTrdP.data() + 12 * idj * TotalModel::NUM_POSE_PARAMETERS, dTrdP.data() + 12 * (idj + 1) * TotalModel::NUM_POSE_PARAMETERS, 0.0);
                    std::fill(dJdP.data() + 3 * idj * TotalModel::NUM_POSE_PARAMETERS, dJdP.data() + 3 * (idj + 1) * TotalModel::NUM_POSE_PARAMETERS, 0.0);
                    if (jacobians[1])
                    {
                        std::fill(dTrdJ.data() + 12 * idj * TotalModel::NUM_POSE_PARAMETERS, dTrdJ.data() + 12 * (idj + 1) * TotalModel::NUM_POSE_PARAMETERS, 0.0);
                        std::fill(dJdJ.data() + 3 * idj * TotalModel::NUM_POSE_PARAMETERS, dJdJ.data() + 3 * (idj + 1) * TotalModel::NUM_POSE_PARAMETERS, 0.0);
                    }
                }
                continue;
            }
            const int ipar = mod_.m_parent[idj];
            const auto baseIndex = idj * 3;
            double angles[3] = {pose[baseIndex], pose[baseIndex+1], pose[baseIndex+2]};

            //Freezing joints here  //////////////////////////////////////////////////////
            if (idj == 10 || idj == 11) //foot ends
            {
                angles[0] = 0.0;
                angles[1] = 0.0;
                angles[2] = 0.0;
            }
            else if (idj == 7 || idj == 8)   //foot ankle. Restrict side movement
            {
                angles[1] = 0.0;
                angles[2] = 0.0;
            }
            else if (idj == 24 || idj == 27 || idj == 28 || idj == 31 || idj == 32 || idj == 35 || idj == 36 || idj == 39 || idj == 40
                || idj == 44 || idj == 47 || idj == 48 || idj == 51 || idj == 52 || idj == 55 || idj == 56 || idj == 59 || idj == 60)  //all hands
            {
                angles[0] = 0.0;
                angles[1] = 0.0;
            }
            // else if (idj == 23 || idj == 26 || idj == 30 || idj == 34 || idj == 38 || idj == 43 || idj == 46 || idj == 50 || idj == 54 || idj == 58)
            //     angles[0] = 0.0;

            ceres::EulerAnglesToRotationMatrix(angles, 3, R.data());
            MR.at(idj) = MR.at(ipar) * R;
            const Matrix<double, 3, 1> offset = (J0.row(idj) - J0.row(ipar)).transpose();
            outJoint.row(idj) = outJoint.row(ipar) + (MR.at(ipar) * offset).transpose();

            if (jacobians)
            {
                EulerAnglesToRotationMatrix_Derivative(angles, dRdP.data(), idj);

                if (idj == 10 || idj == 11) //foot ends
                    dRdP.block<9,3>(0, 3 * idj).setZero();
                if (idj == 7 || idj == 8)   //foot ankle. Restrict side movement
                    dRdP.block<9,2>(0, 3 * idj + 1).setZero();
                if (idj == 24 || idj == 27 || idj == 28 || idj == 31 || idj == 32 || idj == 35 || idj == 36 || idj == 39 || idj == 40)  //all hands
                    dRdP.block<9,2>(0, 3 * idj).setZero();
                if (idj == 44 || idj == 47 || idj == 48 || idj == 51 || idj == 52 || idj == 55 || idj == 56 || idj == 59 || idj == 60)  //all hands
                    dRdP.block<9,2>(0, 3 * idj).setZero();
                // if (idj == 23 || idj == 26 || idj == 30 || idj == 34 || idj == 38 || idj == 43 || idj == 46 || idj == 50 || idj == 54 || idj == 58)
                // {
                //     dRdP.block<9,1>(0, 3 * idj).setZero();
                // }

                if (idj == 10 || idj == 11) //foot ends
                    dMRdP.at(idj) = dMRdP.at(ipar);
                else
                {
                    // Sparse derivative
                    SparseProductDerivative(MR.at(ipar).data(), dMRdP.at(ipar).data(), R.data(), dRdP.data(), idj, m_rigid_body? std::vector<int>(1, 0) :mParentIndexes.at(idj), dMRdP.at(idj).data());
                    // // Slower but equivalent - Dense derivative
                    // Product_Derivative(MR.at(ipar).data(), dMRdP.at(ipar).data(), R.data(), dRdP.data(), dMRdP.at(idj).data()); // Compute the product of matrix multiplication
                }
                SparseProductDerivative(dMRdP.at(ipar).data(), offset.data(), m_rigid_body? std::vector<int>(1, 0) :mParentIndexes.at(ipar), dMtdP.at(idj).data());
                // the following line is equivalent to dMtdP.at(idj) = dMtdP.at(idj) + dMtdP.at(ipar);
                SparseAdd(dMtdP.at(ipar).data(), m_rigid_body? std::vector<int>(1, 0) :mParentIndexes.at(ipar), dMtdP.at(idj).data());

                if (jacobians[1])
                {
                    std::fill(dtdJPtr, dtdJPtr + 9 * TotalModel::NUM_JOINTS, 0.0); // dtdJ.setZero();
                    // the following two lines are equiv to: dtdJ.block<3,3>(0, 3 * idj).setIdentity(); dtdJ.block<3,3>(0, 3 * ipar) -= Matrix<double, 3, 3>::Identity(); // derivative of offset wrt J
                    dtdJPtr[3 * idj] = 1;
                    dtdJPtr[3 * idj + 3 * TotalModel::NUM_JOINTS + 1] = 1;
                    dtdJPtr[3 * idj + 6 * TotalModel::NUM_JOINTS + 2] = 1;
                    dtdJPtr[3 * ipar] = -1;
                    dtdJPtr[3 * ipar + 3 * TotalModel::NUM_JOINTS + 1] = -1;
                    dtdJPtr[3 * ipar + 6 * TotalModel::NUM_JOINTS + 2] = -1;
                    // the following line is equivalent to Product_Derivative(MR.at(ipar).data(), NULL, offset.data(), dtdJPtr, dMtdJ.at(idj).data(), 1); // dA_data is NULL since rotation is not related to joint
                    SparseProductDerivativeConstA(MR.at(ipar).data(), dtdJPtr, mParentIndexes.at(idj), dMtdJ.at(idj).data());
                    // the following line is equivalent to dMtdJ.at(idj) = dMtdJ.at(idj) + dMtdJ.at(ipar);
                    SparseAdd(dMtdJ.at(ipar).data(), mParentIndexes.at(idj), dMtdJ.at(idj).data());
                    std::copy(dMtdJ[idj].data(), dMtdJ[idj].data() + 9 * TotalModel::NUM_JOINTS, dJdJ.data() + 9 * idj * TotalModel::NUM_JOINTS); // dJdJ.block(3 * idj, 0, 3, TotalModel::NUM_JOINTS * 3) = dMtdJ[idj];
                }
                std::copy(dMtdP[idj].data(), dMtdP[idj].data() + 9 * TotalModel::NUM_JOINTS, dJdP.data() + 9 * idj * TotalModel::NUM_JOINTS); // dJdP.block(3 * idj, 0, 3, TotalModel::NUM_JOINTS * 3) = dMtdP[idj];
            }
        }

// const auto duration5 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start5).count();
// const auto start6 = std::chrono::high_resolution_clock::now();
        for (int idj = 0; idj < mod_.NUM_JOINTS; idj++)
        {
            if (!m_FK_joint_list[idj]) continue;
            const Matrix<double, 3, 1> offset = J0.row(idj).transpose();

            outT.block<3,3>(3 * idj, 0) = MR.at(idj);
            outT.block<3,1>(3 * idj, 3) = outJoint.row(idj).transpose() - MR.at(idj) * offset;

            if (jacobians)
            {
                Matrix<double, 3, 3 * TotalModel::NUM_JOINTS, Eigen::RowMajor> dtdP(3, 3 * TotalModel::NUM_JOINTS); // a buffer for the derivative
                // The following line is equivalent to Product_Derivative(MR.at(idj).data(), dMRdP.at(idj).data(), offset.data(), NULL, dtdP.data(), 1);
                SparseProductDerivative(dMRdP.at(idj).data(), offset.data(), m_rigid_body? std::vector<int>(1, 0) :mParentIndexes.at(idj), dtdP.data());
                // The following line is equivalent to dMtdP.at(idj) -= dtdP;
                SparseSubtract(dtdP.data(), m_rigid_body? std::vector<int>(1, 0) :mParentIndexes.at(idj), dMtdP.at(idj).data());

                if (jacobians[1])
                {
                    std::fill(dtdJPtr, dtdJPtr + 9 * TotalModel::NUM_JOINTS, 0.0); // dtdJ.setZero();
                    // The follwing line is equivalent to dtdJ.block<3,3>(0, 3 * idj).setIdentity();
                    dtdJPtr[3 * idj] = 1; dtdJPtr[3 * idj + 3 * TotalModel::NUM_JOINTS + 1] = 1; dtdJPtr[3 * idj + 6 * TotalModel::NUM_JOINTS + 2] = 1;
                    // The following line is equivalent to Product_Derivative(MR.at(idj).data(), NULL, offset.data(), dtdJPtr, dtdJ2.data(), 1);
                    SparseProductDerivativeConstA(MR.at(idj).data(), dtdJPtr, mParentIndexes.at(idj), dtdJ2.data());
                    // The following line is equivalent to dMtdJ.at(idj) -= dtdJ2;
                    SparseSubtract(dtdJ2.data(), mParentIndexes.at(idj), dMtdJ.at(idj).data());
                }

                // The following lines are copying jacobian from dMRdP and dMtdP to dTrdP, equivalent to
                // dTrdP.block(12 * idj + 0, 0, 3, TotalModel::NUM_JOINTS * 3) = dMRdP.at(idj).block(0, 0, 3, TotalModel::NUM_JOINTS * 3);
                // dTrdP.block(12 * idj + 4, 0, 3, TotalModel::NUM_JOINTS * 3) = dMRdP.at(idj).block(3, 0, 3, TotalModel::NUM_JOINTS * 3);
                // dTrdP.block(12 * idj + 8, 0, 3, TotalModel::NUM_JOINTS * 3) = dMRdP.at(idj).block(6, 0, 3, TotalModel::NUM_JOINTS * 3);
                // dTrdP.block(12 * idj + 3, 0, 1, TotalModel::NUM_JOINTS * 3) = dMtdP.at(idj).block(0, 0, 1, TotalModel::NUM_JOINTS * 3);
                // dTrdP.block(12 * idj + 7, 0, 1, TotalModel::NUM_JOINTS * 3) = dMtdP.at(idj).block(1, 0, 1, TotalModel::NUM_JOINTS * 3);
                // dTrdP.block(12 * idj + 11, 0, 1, TotalModel::NUM_JOINTS * 3) = dMtdP.at(idj).block(2, 0, 1, TotalModel::NUM_JOINTS * 3);
                if (m_rigid_body)
                {
                    dTrdP.block(12 * idj + 0, 0, 3, 3) = dMRdP.at(idj).block(0, 0, 3, 3);
                    dTrdP.block(12 * idj + 4, 0, 3, 3) = dMRdP.at(idj).block(3, 0, 3, 3);
                    dTrdP.block(12 * idj + 8, 0, 3, 3) = dMRdP.at(idj).block(6, 0, 3, 3);
                    dTrdP.block(12 * idj + 3, 0, 1, 3) = dMtdP.at(idj).block(0, 0, 1, 3);
                    dTrdP.block(12 * idj + 7, 0, 1, 3) = dMtdP.at(idj).block(1, 0, 1, 3);
                    dTrdP.block(12 * idj + 11, 0, 1, 3) = dMtdP.at(idj).block(2, 0, 1, 3);
                }
                else
                {
                    std::copy(dMRdP.at(idj).data(), dMRdP.at(idj).data() + 9 * TotalModel::NUM_JOINTS, dTrdP.data() + 12 * idj * TotalModel::NUM_JOINTS * 3);
                    std::copy(dMtdP.at(idj).data(), dMtdP.at(idj).data() + 3 * TotalModel::NUM_JOINTS, dTrdP.data() + (12 * idj + 3) * TotalModel::NUM_JOINTS * 3);
                    std::copy(dMRdP.at(idj).data() + 9 * TotalModel::NUM_JOINTS, dMRdP.at(idj).data() + 18 * TotalModel::NUM_JOINTS,
                        dTrdP.data() + (12 * idj + 4)* TotalModel::NUM_JOINTS * 3);
                    std::copy(dMtdP.at(idj).data() + 3 * TotalModel::NUM_JOINTS, dMtdP.at(idj).data() + 6 * TotalModel::NUM_JOINTS,
                        dTrdP.data() + (12 * idj + 7) * TotalModel::NUM_JOINTS * 3);
                    std::copy(dMRdP.at(idj).data() + 18 * TotalModel::NUM_JOINTS, dMRdP.at(idj).data() + 27 * TotalModel::NUM_JOINTS,
                        dTrdP.data() + (12 * idj + 8)* TotalModel::NUM_JOINTS * 3);
                    std::copy(dMtdP.at(idj).data() + 6 * TotalModel::NUM_JOINTS, dMtdP.at(idj).data() + 9 * TotalModel::NUM_JOINTS,
                        dTrdP.data() + (12 * idj + 11) * TotalModel::NUM_JOINTS * 3);
                }

                if (jacobians[1])
                {
                    // The following lines are copying jacobian from and dMtdJ to dTrdJ, equivalent to
                    // dTrdJ.block(12 * idj + 3, 0, 1, TotalModel::NUM_JOINTS * 3) = dMtdJ.at(idj).block(0, 0, 1, TotalModel::NUM_JOINTS * 3);
                    // dTrdJ.block(12 * idj + 7, 0, 1, TotalModel::NUM_JOINTS * 3) = dMtdJ.at(idj).block(1, 0, 1, TotalModel::NUM_JOINTS * 3);
                    // dTrdJ.block(12 * idj + 11, 0, 1, TotalModel::NUM_JOINTS * 3) = dMtdJ.at(idj).block(2, 0, 1, TotalModel::NUM_JOINTS * 3);
                    std::copy(dMtdJ.at(idj).data(), dMtdJ.at(idj).data() + 3 * TotalModel::NUM_JOINTS, dTrdJ.data() + (12 * idj + 3) * TotalModel::NUM_JOINTS * 3);
                    std::copy(dMtdJ.at(idj).data() + 3 * TotalModel::NUM_JOINTS, dMtdJ.at(idj).data() + 6 * TotalModel::NUM_JOINTS,
                        dTrdJ.data() + (12 * idj + 7) * TotalModel::NUM_JOINTS * 3);
                    std::copy(dMtdJ.at(idj).data() + 6 * TotalModel::NUM_JOINTS, dMtdJ.at(idj).data() + 9 * TotalModel::NUM_JOINTS,
                        dTrdJ.data() + (12 * idj + 11) * TotalModel::NUM_JOINTS * 3);
                }
            }
        }
// const auto duration6 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start6).count();
// const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
// std::cout << __FILE__ << " 1:" << duration1 * 1e-6 << "\n"
//           << __FILE__ << " 2:" << duration2 * 1e-6 << "\n"
//           << __FILE__ << " 3:" << duration3 * 1e-6 << "\n"
//           << __FILE__ << " 4:" << duration4 * 1e-6 << "\n"
//           << __FILE__ << " 5:" << duration5 * 1e-6 << "\n"
//           << __FILE__ << " 6:" << duration6 * 1e-6 << "\n"
//           << __FILE__ << " T:" << duration * 1e-6 << std::endl;

        return true;
    }
}
