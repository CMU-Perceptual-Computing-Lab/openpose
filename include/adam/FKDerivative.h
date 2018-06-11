#ifndef FK_DERIVATIVE
#define FK_DERIVATIVE
#include <Eigen/Dense>
#include <ceres/rotation.h>
#include <iostream>
#include <vector>
#include <totalmodel.h>

void AngleAxisToRotationMatrix_Derivative(const double* pose, double* dR_data, const int idj, const int numberColumns=TotalModel::NUM_JOINTS * 3);
void EulerAnglesToRotationMatrix_Derivative(const double* pose, double* dR_data, const int idj, const int numberColumns=TotalModel::NUM_JOINTS * 3);
void Product_Derivative(const double* const A_data, const double* const dA_data, const double* const B_data,
                        const double* const dB_data, double* dAB_data, const int B_col=3);
void SparseProductDerivative(const double* const A_data, const double* const dA_data, const double* const B_data,
                             const double* const dB_data, const int colIndex,
                             const std::vector<int>& parentIndexes, double* dAB_data, const int numberColumns=TotalModel::NUM_JOINTS * 3);
void SparseProductDerivative(const double* const dA_data, const double* const B_data,
                             const std::vector<int>& parentIndexes, double* dAB_data, const int numberColumns=TotalModel::NUM_JOINTS * 3);
void SparseProductDerivativeConstA(const double* const A_data, const double* const dB_data,
                             const std::vector<int>& parentIndexes, double* dAB_data, const int numberColumns=TotalModel::NUM_JOINTS * 3);
void SparseAdd(const double* const B_data, const std::vector<int>& parentIndexes, double* A_data, const int numberColumns=TotalModel::NUM_JOINTS * 3);
void SparseSubtract(const double* const B_data, const std::vector<int>& parentIndexes, double* A_data, const int numberColumns=TotalModel::NUM_JOINTS * 3);

void projection_Derivative(double* dPdI_data, const double* dJdI_data, const int ncol, double* XYZ, const double* pK_, int offsetP, int offsetJ, float weight=1.0f);

template<typename Derived, int rows, int cols, int option>
void projection_Derivative(Eigen::Map<Eigen::Matrix<Derived, rows, cols, option>>& dPdI, Eigen::Matrix<Derived, rows, cols, option>& dJdI, double* XYZ, const double* pK_, int offsetP, int offsetJ, float weight=1.0f)
{
	// Dx/Dt = dx/dX * dX/dt + dx/dY * dY/dt + dx/dZ * dZ/dt
	assert(option == Eigen::RowMajor);
	const double X = XYZ[0], Y = XYZ[1], Z = XYZ[2];
	const int ncol = dJdI.cols();
	auto* P_row0 = dPdI.data() + offsetP * ncol, P_row1 = dPdI.data() + (offsetP + 1) * ncol;
	auto* J_row0 = dJdI.data() + offsetJ * ncol, J_row1 = dJdI.data() + (offsetJ + 1) * ncol, J_row2 = dJdI.data() + (offsetJ + 2) * ncol;
	for (int i = 0; i < ncol; i++)
		P_row0[i] = weight * ( pK_[0] * J_row0[i] + pK_[1] * J_row1[i] - (pK_[0] * X + pK_[1] * Y) * J_row2[i] / Z ) / Z;
	for (int i = 0; i < ncol; i++)
		P_row1[i] = weight * pK_[4] * ( J_row1[i] - Y / Z * J_row2[i] ) / Z;
	// equivalent to
	// dPdI.row(offsetP + 0) = weight * (pK_[0] / Z * dJdI.row(offsetJ + 0)
	// 					 + pK_[1] / Z * dJdI.row(offsetJ + 1)
	// 					 - (pK_[0] * X + pK_[1] * Y) / Z / Z * dJdI.row(offsetJ + 2));
	// dPdI.row(offsetP + 1) = weight * (pK_[4] / Z * dJdI.row(offsetJ + 1)
	// 					 - pK_[4] * Y / Z / Z * dJdI.row(offsetJ + 2));
}

#endif