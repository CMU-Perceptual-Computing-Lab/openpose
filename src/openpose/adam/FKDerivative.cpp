#include "FKDerivative.h"
#include <Eigen/Dense>
#include <totalmodel.h>
#include <cmath>
#include <cassert>

using namespace Eigen;

void AngleAxisToRotationMatrix_Derivative(const double* pose, double* dR_data, int idj)
{
	Eigen::Map< Eigen::Matrix<double, 9, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dR(dR_data);
	dR.setZero();
	const double theta2 = pose[0] * pose[0] + pose[1] * pose[1] + pose[2] * pose[2];
	if (theta2 > std::numeric_limits<double>::epsilon())
	{
		const double theta = sqrt(theta2);
		const double s = sin(theta);
		const double c = cos(theta);
		const Eigen::Map< const Eigen::Matrix<double, 3, 1> > u(pose);
		Eigen::VectorXd e(3);
		e[0] = pose[0] / theta; e[1] = pose[1] / theta; e[2] = pose[2] / theta; 

		// dR / dtheta
		Eigen::Matrix<double, 9, 1> dRdth(9, 1);
		Eigen::Map< Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > dRdth_(dRdth.data());
		// skew symmetric
		dRdth_ << 0.0, -e[2], e[1],
			      e[2], 0.0, -e[0],
			      -e[1], e[0], 0.0;
		// dRdth_ = dRdth_ * c - Matrix<double, 3, 3>::Identity() * s + s * e * e.transpose();
		dRdth_ = - dRdth_ * c - Matrix<double, 3, 3>::Identity() * s + s * e * e.transpose();

		// dR / de
		Eigen::Matrix<double, 9, 3, RowMajor> dRde(9, 3);
		// d(ee^T) / de
		dRde <<
			2 * e[0], 0., 0.,
			e[1], e[0], 0.,
			e[2], 0., e[0],
			e[1], e[0], 0.,
			0., 2 * e[1], 0.,
			0., e[2], e[1],
			e[2], 0., e[0],
			0., e[2], e[1],
			0., 0., 2 * e[2];
		Eigen::Matrix<double, 9, 3, RowMajor> dexde(9, 3);
		dexde <<
			0, 0, 0,
			0, 0, -1,
			0, 1, 0,
			0, 0, 1,
			0, 0, 0,
			-1, 0, 0,
			0, -1, 0,
			1, 0, 0,
			0, 0, 0;
		// dRde = dRde * (1. - c) + c * dexde;
		dRde = dRde * (1. - c) - s * dexde;
		Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> dedu = Matrix<double, 3, 3>::Identity() / theta - u * u.transpose() / theta2 / theta;

		dR.block(0, 3 * idj, 9, 3) = dRdth * e.transpose() + dRde * dedu;
	}
	else
	{
		dR(1, 3 * idj + 2) = 1;
		dR(2, 3 * idj + 1) = -1;
		dR(3, 3 * idj + 2) = -1;
		dR(5, 3 * idj + 0) = 1;
		dR(6, 3 * idj + 1) = 1;
		dR(7, 3 * idj + 0) = -1;
	}
}

void EulerAnglesToRotationMatrix_Derivative(const double* pose, double* dR_data, int idj)
{
	Eigen::Map< Eigen::Matrix<double, 9, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dR(dR_data);
	dR.setZero();
	Eigen::Matrix<double, 9, 3, Eigen::RowMajor> dRdp(9, 3);
	const double degrees_to_radians = 3.14159265358979323846 / 180.0;
	const double pitch(pose[0] * degrees_to_radians);
	const double roll(pose[1] * degrees_to_radians);
	const double yaw(pose[2] * degrees_to_radians);

	const double c1 = cos(yaw);
	const double s1 = sin(yaw);
	const double c2 = cos(roll);
	const double s2 = sin(roll);
	const double c3 = cos(pitch);
	const double s3 = sin(pitch);

	// dRdp << 
	// 	-s1 * c2, -s2 * c1, 0.,
	// 	-c1 * c3 - s1 * s2 * s3, c1 * c2 * s3, s1 * s3 + c1 * s2 * c3,
	// 	c1 * s3 - s1 * s2 * c3, c1 * c2 * c3, s1 * c3 - c1 * s2 * s3,
	// 	c1 * c2, -s1 * s2, 0.,
	// 	-s1 * c3 + c1 * s2 * s3, s1 * c2 * s3, -c1 * s3 + s1 * s2 * c3,
	// 	s1 * s3 + c1 * s2 * c3, s1 * c2 * c3, -c1 * c3 - s1 * s2 * s3,
	// 	0, -c2, 0,
	// 	0, -s2 * s3, c2 * c3,
	// 	0, -s2 * c3, -c2 * s3;

	dRdp << 
		0., -s2 * c1, -s1 * c2,
		s1 * s3 + c1 * s2 * c3, c1 * c2 * s3, -c1 * c3 - s1 * s2 * s3,
		s1 * c3 - c1 * s2 * s3, c1 * c2 * c3, c1 * s3 - s1 * s2 * c3,
		0., -s1 * s2, c1 * c2,
		-c1 * s3 + s1 * s2 * c3, s1 * c2 * s3, -s1 * c3 + c1 * s2 * s3,
		-c1 * c3 - s1 * s2 * s3, s1 * c2 * c3, s1 * s3 + c1 * s2 * c3, 
		0, -c2, 0,
		c2 * c3, -s2 * s3, 0,
		-c2 * s3, -s2 * c3, 0;

	dR.block(0, 3 * idj, 9, 3) = degrees_to_radians * dRdp;
}

void Product_Derivative(double* A_data, double* dA_data, double* B_data, double* dB_data, double* dAB_data, const int B_col)
{
	assert(B_col == 3 || B_col == 1);  // matrix multiplication or matrix-vector multiplication
	Eigen::Map< Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > A(A_data);
	Eigen::Map< Eigen::Matrix<double, 9, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dA(dA_data);
	if (dB_data != NULL)
	{
		if (B_col == 1)
		{
			Eigen::Map< Eigen::Matrix<double, 9, 1> > B(B_data);
			Eigen::Map< Eigen::Matrix<double, 3, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dB(dB_data);
			Eigen::Map< Eigen::Matrix<double, 3, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dAB(dAB_data);
			dAB.setZero();
			for (int r = 0; r < 3; r++)
				dAB.row(r) = A(r, 0) * dB.row(0) + A(r, 1) * dB.row(1) + A(r, 2) * dB.row(2) + 
					B[0] * dA.row(3 * r + 0) + B[1] * dA.row(3 * r + 1) + B[2] * dA.row(3 * r + 2);  // d(AB) = AdB + (dA)B
		}
		else
		{
			// B_col == 3
			Eigen::Map< Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > B(B_data);
			Eigen::Map< Eigen::Matrix<double, 9, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dB(dB_data);
			Eigen::Map< Eigen::Matrix<double, 9, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dAB(dAB_data);
			dAB.setZero();
			for (int r = 0; r < 3; r++)
				for (int c = 0; c < 3; c++)
					dAB.row(3 * r + c) = A(r, 0) * dB.row(0 + c) + A(r, 1) * dB.row(3 + c) + A(r, 2) * dB.row(6 + c) +
						B(0, c) * dA.row(3 * r + 0) + B(1, c) * dA.row(3 * r + 1) + B(2, c) * dA.row(3 * r + 2);  // d(AB) = AdB + (dA)B
		}
	}
	else  // B is a constant matrix / vector, no derivative
	{
		if (B_col == 1)
		{
			Eigen::Map< Eigen::Matrix<double, 9, 1> > B(B_data);
			Eigen::Map< Eigen::Matrix<double, 3, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dAB(dAB_data);
			dAB.setZero();
			for (int r = 0; r < 3; r++)
				dAB.row(r) = B[0] * dA.row(3 * r + 0) + B[1] * dA.row(3 * r + 1) + B[2] * dA.row(3 * r + 2);  // d(AB) = AdB + (dA)B
		}
		else
		{
			// B_col == 3
			Eigen::Map< Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > B(B_data);
			Eigen::Map< Eigen::Matrix<double, 9, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dAB(dAB_data);
			dAB.setZero();
			for (int r = 0; r < 3; r++)
				for (int c = 0; c < 3; c++)
					dAB.row(3 * r + c) = B(0, c) * dA.row(3 * r + 0) + B(1, c) * dA.row(3 * r + 1) + B(2, c) * dA.row(3 * r + 2);  // d(AB) = AdB + (dA)B
		}
	}
}