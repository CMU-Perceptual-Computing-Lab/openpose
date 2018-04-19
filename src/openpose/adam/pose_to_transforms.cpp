#include <chrono>
#include <iostream>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "pose_to_transforms.h"

namespace smpl
{
    bool PoseToTransformsNoLR_Eulers_adamModel_withDiff::Evaluate(double const* const* parameters,
        double* residuals,
        double** jacobians) const
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
        std::vector<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> MR(TotalModel::NUM_JOINTS, Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(3, 3));
        ceres::AngleAxisToRotationMatrix(pose, MR.at(0).data());
        std::vector<Eigen::Matrix<double, 9, 3 * TotalModel::NUM_JOINTS, Eigen::RowMajor>> dMRdP(TotalModel::NUM_JOINTS, Eigen::Matrix<double, 9, 3 * TotalModel::NUM_JOINTS, Eigen::RowMajor>(9, 3 * TotalModel::NUM_JOINTS));
        Map< Matrix<double, TotalModel::NUM_JOINTS * 3, TotalModel::NUM_JOINTS * 3, RowMajor> > dJdP(jacobians[0]);
// const auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start1).count();
// const auto start2 = std::chrono::high_resolution_clock::now();
        dJdP.block<3, TotalModel::NUM_JOINTS * 3>(0, 0).setZero();
        AngleAxisToRotationMatrix_Derivative(pose, dMRdP.at(0).data(), 0);
        Matrix<double, 9, 3 * TotalModel::NUM_JOINTS, RowMajor> dRdP(9, 3 * TotalModel::NUM_JOINTS);
        Eigen::Map< Eigen::Matrix<double, TotalModel::NUM_JOINTS, 3, Eigen::RowMajor> > outJoint(residuals);
        outJoint.row(0) = J0_.row(0);
        Matrix<double, 3, 3, RowMajor> R; // Interface with ceres
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
            if (idj == 24 || idj == 27 || idj == 28 || idj == 31 || idj == 32 || idj == 35 || idj == 26 || idj == 39 || idj == 40
                || idj == 44 || idj == 47 || idj == 48 || idj == 51 || idj == 52 || idj == 55 || idj == 56 || idj == 59 || idj == 60)   //all hands
            {
                angles[0] = 0.0;
                angles[1] = 0.0;
            }
            ceres::EulerAnglesToRotationMatrix(angles, 3, R.data());
// duration7 += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start7).count();
// const auto start8 = std::chrono::high_resolution_clock::now();
            EulerAnglesToRotationMatrix_Derivative(angles, dRdP.data(), idj);
// duration8 += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start8).count();
// const auto start9 = std::chrono::high_resolution_clock::now();

            if (idj == 10 || idj == 11) //foot ends
                dRdP.block<9, 3>(0, 3 * idj).setZero();
            if (idj == 7 || idj == 8)   //foot ankle. Restrict side movement
                dRdP.block<9, 1>(0, 3 * idj + 2).setZero();
            if (idj == 24 || idj == 27 || idj == 28 || idj == 31 || idj == 32 || idj == 35 || idj == 26 || idj == 39 || idj == 40
                || idj == 44 || idj == 47 || idj == 48 || idj == 51 || idj == 52 || idj == 55 || idj == 56 || idj == 59 || idj == 60)   //all hands
                dRdP.block<9, 2>(0, 3 * idj).setZero();
// duration9 += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start9).count();
// const auto start10 = std::chrono::high_resolution_clock::now();

            MR.at(idj) = MR.at(ipar) * R;
            if (idj == 10 || idj == 11) //foot ends
                dMRdP.at(idj) = dMRdP.at(ipar);
            else
            {
                // Sparse derivative
                SparseProductDerivative(MR.at(ipar).data(), dMRdP.at(ipar).data(), R.data(), dRdP.data(), idj, mParentIndexes[idj], dMRdP.at(idj).data());
                // // Slower but equivalent - Dense derivative
                // Product_Derivative(MR.at(ipar).data(), dMRdP.at(ipar).data(), R.data(), dRdP.data(), dMRdP.at(idj).data()); // Compute the product of matrix multiplication
            }
// duration10 += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start10).count();
// const auto start11 = std::chrono::high_resolution_clock::now();
// duration11 += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start11).count();
// const auto start12 = std::chrono::high_resolution_clock::now();

            const Matrix<double, 3, 1> offset = (J0_.row(idj) - J0_.row(ipar)).transpose();
            outJoint.row(idj) = outJoint.row(ipar) + (MR.at(ipar) * offset).transpose();
// duration12 += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start12).count();
// const auto start13 = std::chrono::high_resolution_clock::now();
            auto dMtdPIdj = dJdP.block<3, TotalModel::NUM_JOINTS * 3>(3 * idj, 0);
// duration13 += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start13).count();
// const auto start14 = std::chrono::high_resolution_clock::now();
            // Product_Derivative(NULL, dMRdP.at(ipar).data(), offset.data(), NULL, dMtdPIdj.data(), 1); // dB_data is NULL since offset is a constant
            SparseProductDerivative(dMRdP.at(ipar).data(), offset.data(), mParentIndexes[idj], dMtdPIdj.data()); // dB_data is NULL since offset is a constant
// duration14 += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start14).count();
// const auto start15 = std::chrono::high_resolution_clock::now();
            SparseAdd(dJdP.block<3, TotalModel::NUM_JOINTS * 3>(3 * ipar, 0).data(), mParentIndexes[idj], dMtdPIdj.data());
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

        return true;
    }
}
