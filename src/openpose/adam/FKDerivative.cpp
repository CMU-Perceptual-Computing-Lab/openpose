#include "FKDerivative.h"
#include <Eigen/Dense>
#include <totalmodel.h>
#include <cmath>
#include <cassert>
#include <chrono>

using namespace Eigen;

void AngleAxisToRotationMatrix_Derivative(const double* pose, double* dR_data, const int idj, const int numberColumns)
{
    Eigen::Map< Eigen::Matrix<double, 9, Eigen::Dynamic, Eigen::RowMajor> > dR(dR_data, 9, numberColumns);
    std::fill(dR_data, dR_data + 9 * numberColumns, 0.0);
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
        dR(5, 3 * idj) = 1;
        dR(6, 3 * idj + 1) = 1;
        dR(7, 3 * idj) = -1;
    }
}

void EulerAnglesToRotationMatrix_Derivative(const double* pose, double* dR_data, const int idj, const int numberColumns)
{
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

    Eigen::Matrix<double, 9, 3, Eigen::RowMajor> dRdp(9, 3);
    // dRdp <<
    //  -s1 * c2, -s2 * c1, 0.,
    //  -c1 * c3 - s1 * s2 * s3, c1 * c2 * s3, s1 * s3 + c1 * s2 * c3,
    //  c1 * s3 - s1 * s2 * c3, c1 * c2 * c3, s1 * c3 - c1 * s2 * s3,
    //  c1 * c2, -s1 * s2, 0.,
    //  -s1 * c3 + c1 * s2 * s3, s1 * c2 * s3, -c1 * s3 + s1 * s2 * c3,
    //  s1 * s3 + c1 * s2 * c3, s1 * c2 * c3, -c1 * c3 - s1 * s2 * s3,
    //  0, -c2, 0,
    //  0, -s2 * s3, c2 * c3,
    //  0, -s2 * c3, -c2 * s3;

    dRdp <<
        0.,                       -s2 * c1,      -s1 * c2,
        s1 * s3 + c1 * s2 * c3,    c1 * c2 * s3, -c1 * c3 - s1 * s2 * s3,
        s1 * c3 - c1 * s2 * s3,    c1 * c2 * c3, c1 * s3 - s1 * s2 * c3,
        0.,                       -s1 * s2,      c1 * c2,
        -c1 * s3 + s1 * s2 * c3,   s1 * c2 * s3, -s1 * c3 + c1 * s2 * s3,
        -c1 * c3 - s1 * s2 * s3,   s1 * c2 * c3, s1 * s3 + c1 * s2 * c3,
        0,                        -c2,           0,
        c2 * c3,                  -s2 * s3,      0,
        -c2 * s3,                 -s2 * c3,      0;

    Eigen::Map< Eigen::Matrix<double, 9, Eigen::Dynamic, Eigen::RowMajor> > dR(dR_data, 9, numberColumns);
    std::fill(dR_data, dR_data + 9 * numberColumns, 0.0);
    dR.block(0, 3 * idj, 9, 3) = degrees_to_radians * dRdp;
}

void Product_Derivative(const double* const A_data, const double* const dA_data, const double* const B_data,
                        const double* const dB_data, double* dAB_data, const int B_col)
{
    assert(dA_data != NULL || dB_data != NULL);
    assert(B_col == 3 || B_col == 1);  // matrix multiplication or matrix-vector multiplication
    if (dA_data != NULL && dB_data != NULL)
    {
        const Eigen::Map<const Eigen::Matrix<double, 9, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dA(dA_data);
        if (B_col == 1)
        {
            // B_col == 1
            // d(AB) = AdB + (dA)B
            const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > A(A_data);
            const Eigen::Map<const Eigen::Matrix<double, 3, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dB(dB_data);
            Eigen::Map< Eigen::Matrix<double, 3, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dAB(dAB_data);
            for (int r = 0; r < 3; r++)
            {
                const int baseIndex = 3*r;
                dAB.row(r) = A(r, 0) * dB.row(0) + A(r, 1) * dB.row(1) + A(r, 2) * dB.row(2) +
                    B_data[0] * dA.row(baseIndex) + B_data[1] * dA.row(baseIndex + 1) + B_data[2] * dA.row(baseIndex + 2);
            }
        }
        else
        {
            // B_col == 3
            // d(AB) = AdB + (dA)B
            const Eigen::Map<const Eigen::Matrix<double, 9, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dB(dB_data);
            Eigen::Map< Eigen::Matrix<double, 9, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dAB(dAB_data);
            for (int r = 0; r < 3; r++)
            {
                const int baseIndex = 3*r;
                for (int c = 0; c < 3; c++)
                {
                    dAB.row(baseIndex + c) = A_data[baseIndex] * dB.row(c) + A_data[baseIndex+1] * dB.row(3 + c) + A_data[baseIndex+2] * dB.row(6 + c) +
                        B_data[c] * dA.row(baseIndex) + B_data[3 + c] * dA.row(baseIndex + 1) + B_data[6 + c] * dA.row(baseIndex + 2);
                }
            }
        }
    }
    else if (dA_data != NULL && dB_data == NULL)  // B is a constant matrix / vector, no derivative
    {
        const Eigen::Map<const Eigen::Matrix<double, 9, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dA(dA_data);
        if (B_col == 1)
        {
            // d(AB) = AdB + (dA)B
            Eigen::Map< Eigen::Matrix<double, 3, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dAB(dAB_data);
            // // Matrix form (slower)
            // for (int r = 0; r < 3; r++)
            //     dABAux.row(r) = B * dA.block<3, TotalModel::NUM_JOINTS * 3>(r, 0);
            // For loop form
            for (int r = 0; r < 3; r++)
            {
                const int baseIndex = 3*r;
                dAB.row(r) = B_data[0] * dA.row(baseIndex) + B_data[1] * dA.row(baseIndex + 1) + B_data[2] * dA.row(baseIndex + 2);
            }
        }
        else
        {
            // B_col == 3
            Eigen::Map< Eigen::Matrix<double, 9, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dAB(dAB_data);
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    dAB.row(3 * r + c) = B_data[c] * dA.row(3 * r) + B_data[3 + c] * dA.row(3 * r + 1) + B_data[6 + c] * dA.row(3 * r + 2);  // d(AB) = AdB + (dA)B
        }
    }
    else // A is a constant matrix, no derivative
    {
        const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > A(A_data);
        // dA_data == NULL && dB_data != NULL
        if (B_col == 1)
        {
            const Eigen::Map<const Eigen::Matrix<double, 3, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dB(dB_data);
            Eigen::Map< Eigen::Matrix<double, 3, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dAB(dAB_data);
            dAB.setZero();
            for (int r = 0; r < 3; r++)
                dAB.row(r) = A(r, 0) * dB.row(0) + A(r, 1) * dB.row(1) + A(r, 2) * dB.row(2);
        }
        else
        {
            // B_col == 3
            const Eigen::Map<const Eigen::Matrix<double, 9, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dB(dB_data);
            Eigen::Map< Eigen::Matrix<double, 9, TotalModel::NUM_JOINTS * 3, Eigen::RowMajor> > dAB(dAB_data);
            dAB.setZero();
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    dAB.row(3 * r + c) = A(r, 0) * dB.row(0 + c) + A(r, 1) * dB.row(3 + c) + A(r, 2) * dB.row(6 + c);
        }
    }
}

void SparseProductDerivative(const double* const A_data, const double* const dA_data, const double* const B_data,
                             const double* const dB_data, const int colIndex, const std::vector<int>& parentIndexes, double* dAB_data, const int numberColumns)
{
    // d(AB) = AdB + (dA)B
    Eigen::Map< Eigen::Matrix<double, 9, Eigen::Dynamic, Eigen::RowMajor> > dAB(dAB_data, 9, numberColumns);

    std::fill(dAB_data, dAB_data + 9 * numberColumns, 0.0);
    // // Dense dAB (sparse dB) version
    // const Eigen::Map<const Eigen::Matrix<double, 9, numberColumns, Eigen::RowMajor> > dA(dA_data);
    // const Eigen::Map<const Eigen::Matrix<double, 9, numberColumns, Eigen::RowMajor> > dB(dB_data);
    // dAB.row(baseIndex + c) = B_data[c] * dA.row(baseIndex) + B_data[3 + c] * dA.row(baseIndex + 1) + B_data[6 + c] * dA.row(baseIndex + 2);
    // dAB.block<1,3>(baseIndex + c, 3*colIndex) += A_data[baseIndex] * dB.block<1,3>(c, 3*colIndex)
    //                                            + A_data[baseIndex+1] * dB.block<1,3>(3+c, 3*colIndex)
    //                                            + A_data[baseIndex+2] * dB.block<1,3>(6+c, 3*colIndex);
    // Sparse sped up equivalent
    const auto colOffset = 3*colIndex;
    for (int r = 0; r < 3; r++)
    {
        const int baseIndex = 3*r;
        for (int c = 0; c < 3; c++)
        {
            // AdB
            for (int subIndex = 0; subIndex < 3; subIndex++)
            {
                const auto finalOffset = colOffset + subIndex;
                dAB_data[numberColumns*(baseIndex + c) + finalOffset] +=
                    A_data[baseIndex] * dB_data[numberColumns*c + finalOffset]
                    + A_data[baseIndex+1] * dB_data[numberColumns*(3+c) + finalOffset]
                    + A_data[baseIndex+2] * dB_data[numberColumns*(6+c) + finalOffset];
            }
            // // AdB - Slower equivalent
            // dAB.block<1,3>(baseIndex + c, colOffset) += A_data[baseIndex] * dB.block<1,3>(c, colOffset)
            //                                            + A_data[baseIndex+1] * dB.block<1,3>(3+c, colOffset)
            //                                            + A_data[baseIndex+2] * dB.block<1,3>(6+c, colOffset);
            // (dA)B
            for (const auto& parentIndex : parentIndexes)
            {
                const auto parentOffset = 3*parentIndex;
                for (int subIndex = 0; subIndex < 3; subIndex++)
                {
                    const auto finalOffset = parentOffset + subIndex;
                    dAB_data[numberColumns*(baseIndex + c) + finalOffset] +=
                        B_data[c] * dA_data[numberColumns*baseIndex + finalOffset]
                        + B_data[3 + c] * dA_data[numberColumns*(baseIndex+1) + finalOffset]
                        + B_data[6 + c] * dA_data[numberColumns*(baseIndex+2) + finalOffset];
                }
            }
            // // (dA)B - Slower equivalent
            // for (const auto& parentIndex : parentIndexes)
            // {
            //     const auto parentOffset = 3*parentIndex;
            //     dAB.block<1,3>(baseIndex + c, parentOffset) += B_data[c] * dA.block<1,3>(baseIndex, parentOffset)
            //                                                  + B_data[3 + c] * dA.block<1,3>(baseIndex+1, parentOffset)
            //                                                  + B_data[6 + c] * dA.block<1,3>(baseIndex+2, parentOffset);
            // }
        }
    }
}

void SparseProductDerivative(const double* const dA_data, const double* const B_data,
                             const std::vector<int>& parentIndexes, double* dAB_data, const int numberColumns)
{
    // d(AB) = AdB + (dA)B
    // Sparse for loop form
    std::fill(dAB_data, dAB_data + 3 * numberColumns, 0.0);
    for (int r = 0; r < 3; r++)
    {
        const int baseIndex = 3*r;
        for (const auto& parentIndex : parentIndexes)
        {
            const auto parentOffset = 3*parentIndex;
            for (int subIndex = 0; subIndex < 3; subIndex++)
            {
                const auto finalOffset = parentOffset + subIndex;
                dAB_data[numberColumns*r + finalOffset] +=
                    B_data[0] * dA_data[numberColumns*baseIndex + finalOffset]
                    + B_data[1] * dA_data[numberColumns*(baseIndex+1) + finalOffset]
                    + B_data[2] * dA_data[numberColumns*(baseIndex+2) + finalOffset];
            }
        }
    }
    // // Dense Matrix form (slower)
    // Eigen::Map< Eigen::Matrix<double, 3, numberColumns, Eigen::RowMajor> > dAB(dAB_data);
    // const Eigen::Map<const Eigen::Matrix<double, 9, numberColumns, Eigen::RowMajor> > dA(dA_data);
    // for (int r = 0; r < 3; r++)
    //     dABAux.row(r) = B * dA.block<3, numberColumns>(r, 0);
    // // Dense for loop form
    // for (int r = 0; r < 3; r++)
    // {
    //     const int baseIndex = 3*r;
    //     dAB.row(r) = B_data[0] * dA.row(baseIndex) + B_data[1] * dA.row(baseIndex + 1) + B_data[2] * dA.row(baseIndex + 2);
    // }
}

void SparseProductDerivativeConstA(const double* const A_data, const double* const dB_data,
                             const std::vector<int>& parentIndexes, double* dAB_data, const int numberColumns)
{
	// d(AB) = AdB (A is a constant.)
    // Sparse for loop form
    std::fill(dAB_data, dAB_data + 3 * numberColumns, 0.0);
    for (int r = 0; r < 3; r++)
    {
		for (const auto& parentIndex : parentIndexes)
	    {
	    	const auto parentOffset = 3*parentIndex;
	    	for (int subIndex = 0; subIndex < 3; subIndex++)
            {
            	const auto finalOffset = parentOffset + subIndex;
            	dAB_data[numberColumns * r + finalOffset] = A_data[3 * r + 0] * dB_data[finalOffset] + A_data[3 * r + 1] * dB_data[finalOffset + numberColumns] +
            		A_data[3 * r + 2] * dB_data[finalOffset + numberColumns + numberColumns];
            }
		}
	}
}

void SparseAdd(const double* const B_data, const std::vector<int>& parentIndexes, double* A_data, const int numberColumns)
{
    // d(AB) += d(AB)_parent
    Eigen::Map< Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> A(A_data, 3, numberColumns);
    const Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> B(B_data, 3, numberColumns);
    // Sparse for loop
    for (int r = 0; r < 3; r++)
    {
        for (const auto& parentIndex : parentIndexes)
        {
            const auto parentOffset = 3*parentIndex;
            for (int subIndex = 0; subIndex < 3; subIndex++)
            {
                const auto finalOffset = parentOffset + subIndex;
                A_data[numberColumns*r + finalOffset] += B_data[numberColumns*r + finalOffset];
            }
        }
    }
    // // Dense equivalent
    // dMtdPIdj += dJdP.block<3, numberColumns>(3 * ipar, 0);
    // A += B;
}

void SparseSubtract(const double* const B_data, const std::vector<int>& parentIndexes, double* A_data, const int numberColumns)
{
    // d(AB) += d(AB)_parent
    Eigen::Map< Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> A(A_data, 3, numberColumns);
    const Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> B(B_data, 3, numberColumns);
    // Sparse for loop
    for (int r = 0; r < 3; r++)
    {
        for (const auto& parentIndex : parentIndexes)
        {
            const auto parentOffset = 3*parentIndex;
            for (int subIndex = 0; subIndex < 3; subIndex++)
            {
                const auto finalOffset = parentOffset + subIndex;
                A_data[numberColumns*r + finalOffset] -= B_data[numberColumns*r + finalOffset];
            }
        }
    }
    // // Dense equivalent
    // dMtdPIdj -= dJdP.block<3, numberColumns>(3 * ipar, 0);
    // A -= B;
}

void projection_Derivative(double* dPdI_data, const double* dJdI_data, const int ncol, double* XYZ, const double* pK_, int offsetP, int offsetJ, float weight)
{
	// Dx/Dt = dx/dX * dX/dt + dx/dY * dY/dt + dx/dZ * dZ/dt
	const double X = XYZ[0], Y = XYZ[1], Z = XYZ[2];
	double* P_row0 = dPdI_data + offsetP * ncol;
	double* P_row1 = dPdI_data + (offsetP + 1) * ncol;
	const double* J_row0 = dJdI_data + offsetJ * ncol;
	const double* J_row1 = dJdI_data + (offsetJ + 1) * ncol;
	const double* J_row2 = dJdI_data + (offsetJ + 2) * ncol;
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