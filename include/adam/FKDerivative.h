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

template<typename Derived, int rows, int cols, int option>
void projection_Derivative(Eigen::Map<Eigen::Matrix<Derived, rows, cols, option>>& dPdI, Eigen::Matrix<Derived, rows, cols, option>& dJdI, double* XYZ, double* pK_, int offsetP, int offsetJ, float weight=1.0f)
{
	// Dx/Dt = dx/dX * dX/dt + dx/dY * dY/dt + dx/dZ * dZ/dt
	const double X = XYZ[0], Y = XYZ[1], Z = XYZ[2];
	dPdI.row(offsetP + 0) = weight * (pK_[0] / Z * dJdI.row(offsetJ + 0)
						 + pK_[1] / Z * dJdI.row(offsetJ + 1)
						 - (pK_[0] * X + pK_[1] * Y) / Z / Z * dJdI.row(offsetJ + 2));
	dPdI.row(offsetP + 1) = weight * (pK_[4] / Z * dJdI.row(offsetJ + 1)
						 - pK_[4] * Y / Z / Z * dJdI.row(offsetJ + 2));
}
