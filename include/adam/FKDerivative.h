#pragma once
#include <Eigen/Dense>
#include <ceres/rotation.h>

void AngleAxisToRotationMatrix_Derivative(const double* pose, double* dR_data, const int idj);
void EulerAnglesToRotationMatrix_Derivative(const double* pose, double* dR_data, const int idj);
void Product_Derivative(const double* const A_data, const double* const dA_data, const double* const B_data,
                        const double* const dB_data, double* dAB_data, const int B_col=3);
void SparseProductDerivative(const double* const A_data, const double* const dA_data, const double* const B_data,
                             const double* const dB_data, const int colIndex,
                             const std::vector<int>& parentIndexes, double* dAB_data);
void SparseProductDerivative(const double* const dA_data, const double* const B_data,
                             const std::vector<int>& parentIndexes, double* dAB_data);
void SparseAdd(const double* const B_data, const std::vector<int>& parentIndexes, double* A_data);

struct CallAngleAxis
{
    CallAngleAxis() {}

    template <typename T>
    bool operator()(const T* const pose,
        T* residual
    ) const
    {
        using namespace Eigen;
        Matrix<T, 3, 3, RowMajor> A(3, 3);
        T angles[3] = {pose[3], pose[4], pose[5]};
        ceres::AngleAxisToRotationMatrix(angles, A.data());
        Matrix<T, 3, 1> B(3, 1);
        B[0] = T(-10);
        B[1] = T(0);
        B[2] = T(-25);
        Map< Matrix<T, 3, 1> > R(residual);
        R = A * B;
        return true;
    }
};
