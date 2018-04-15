#pragma once
#include<iostream>
#include <Eigen/Dense>
#include <ceres/rotation.h>

void AngleAxisToRotationMatrix_Derivative(const double* pose, double* dR_data, int idj);
void EulerAnglesToRotationMatrix_Derivative(const double* pose, double* dR_data, int idj);
void Product_Derivative(double* A_data, double* dA_data, double* B_data, double* dB_data, double* dAB_data, const int B_col=3);

struct CallAngleAxis
{
	CallAngleAxis() {}

	template <typename T>
	bool operator()(const T* const pose,
		T* residual
	) const
	{
		using namespace Eigen;
		Map< const Matrix<T, 62, 3, RowMajor> > P(pose);
		Matrix<T, 3, 3, RowMajor> A(3, 3);
		Matrix<T, 3, 1> B(3, 1);
		Map< Matrix<T, 3, 1> > R(residual);	
		T angles[3];
		angles[0] = pose[1 * 3];
		angles[1] = pose[1 * 3 + 1];
		angles[2] = pose[1 * 3 + 2];
		ceres::AngleAxisToRotationMatrix(angles, A.data());
		B[0] = T(-10);
		B[1] = T(0);
		B[2] = T(-25);
		R = A * B;
		return true;
	}
};
