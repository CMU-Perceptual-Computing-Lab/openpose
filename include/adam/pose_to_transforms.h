#include "simple.h"
#include "handm.h"
#include "totalmodel.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <iostream>
#include <chrono>
#include "FKDerivative.h"

namespace smpl {
struct PoseToTransformsNoLR_Eulers	{
	PoseToTransformsNoLR_Eulers(const SMPLModel &mod)
		: mod_(mod) {}

	template <typename T>
	bool operator()(const T* const pose,  // SMPLModel::NUM_JOINTS*3
		const T* const joints,  // (SMPLModel::NUM_JOINTS)*3
		T* transforms // (SMPLModel::NUM_JOINTS)*3*4
		) const {

		using namespace Eigen;
		Map< const Matrix<T, SMPLModel::NUM_JOINTS, 3, RowMajor> > J(joints);
		Map< Matrix<T, 3 * SMPLModel::NUM_JOINTS, 4, RowMajor> > outT(transforms);
		Matrix<T, Dynamic, 4, RowMajor> Ms(4 * SMPLModel::NUM_JOINTS, 4);
		Matrix<T, 3, 3, ColMajor> R; // Interface with ceres

		ceres::AngleAxisToRotationMatrix(pose, R.data());
		//ceres::EulerAnglesToRotationMatrix(pose, R.data());
		Ms.setZero();
		Ms.block(0, 0, 3, 3) = R;
		Ms(0, 3) = J(0, 0);
		Ms(1, 3) = J(0, 1);
		Ms(2, 3) = J(0, 2);
		Ms(3, 3) = T(1.0);

		for (int idj = 1; idj < mod_.NUM_JOINTS; idj++)
		{
			int ipar = mod_.parent_[idj];
			//std::cout << idj << " " << ipar << "\n\n";
			//ceres::AngleAxisToRotationMatrix(pose + idj * 3, R.data());

			//ceres::EulerAnglesToRotationMatrix(pose + idj * 3, 3,  R.data());
			T angles[3];
			angles[0] = pose[idj * 3];
			angles[1] = pose[idj * 3 + 1];
			angles[2] = pose[idj * 3 + 2];

			//Freezing joints here  //////////////////////////////////////////////////////
			if (idj == 10 || idj == 11)	//foot ends
			{
				//R.setIdentity();
				angles[0] = T(0.0);
				angles[1] = T(0.0);
				angles[2] = T(0.0);
			}
			if (idj == 7 || idj == 8)	//foot ankle. Restrict side movement
			{
				angles[2] = T(0.0);
			}

			if (idj == 18 || idj == 19)	//Elbow. Restrict side movement (allowing twist and trival? move)
			{
				angles[0] = T(0.0);
				//angles[1] = T(0.0);
			}

	/*				if (idj == 20 || idj == 21)	//wrist. Restrict twist
			{
				angles[0] = T(0.0);		
				//angles[1] = T(0.0);
			}*/
			//R.setIdentity();
			//////////////////////////////////////////////////////////////////////////////

			ceres::EulerAnglesToRotationMatrix(angles, 3,  R.data());


			Ms.block(idj * 4, 0, 3, 3) = Ms.block(ipar * 4, 0, 3, 3)*R;
			Ms.block(idj * 4, 3, 3, 1) = Ms.block(ipar * 4, 3, 3, 1) +
			Ms.block(ipar * 4, 0, 3, 3)*(J.row(idj).transpose() - J.row(ipar).transpose());
			Ms(idj * 4 + 3, 3) = T(1.0);
		}
		for (int idj = 0; idj < mod_.NUM_JOINTS; idj++) {
			Ms.block(idj * 4, 3, 3, 1) -= Ms.block(idj * 4, 0, 3, 3)*J.row(idj).transpose();
		}
		for (int idj = 0; idj < mod_.NUM_JOINTS; idj++) {
			outT.block(idj * 3, 0, 3, 4) = Ms.block(idj * 4, 0, 3, 4);
		}
		return true;
	}

	const SMPLModel &mod_;
};

struct PoseToTransformsHand {
	PoseToTransformsHand(const HandModel &mod)
		: mod_(mod) {}

	template <typename T>
	bool operator()(const T* const coeffs,
		const T* const pose,  // HandModel::NUM_JOINTS*3
		T* transforms // (SMPLModel::NUM_JOINTS)*3*4
		) const
	{
		using namespace Eigen;
		Map< const Matrix<T, HandModel::NUM_JOINTS, 3, RowMajor> > C(coeffs);

		Map< Matrix<T, 3 * HandModel::NUM_JOINTS, 4, RowMajor> > outT(transforms);		//Transforms
		Map< Matrix<T, 3 * HandModel::NUM_JOINTS, 4, RowMajor> > outJoint(transforms + 3 * HandModel::NUM_JOINTS * 4);		//Joint location
		Matrix<T, Dynamic, 4, RowMajor> Ms(4 * HandModel::NUM_JOINTS, 4);
		Matrix<T, 4, 4, RowMajor> Mthis;

		Matrix<T, 3, 3, ColMajor> R; // Interface with ceres
		Matrix<T, 3, 3, RowMajor> Rr; // Interface with ceres
		Mthis.setZero();
		Mthis(3, 3) = T(1.0);
		Ms.setZero();
		int idj = mod_.update_inds_(0);
		ceres::AngleAxisToRotationMatrix(pose + idj * 3, R.data());
		Mthis.block(0, 0, 3, 3) = R * DiagonalMatrix<T, 3>(T(C(idj, 0)), T(C(idj, 0)), T(C(idj, 0)));
		// Mthis.block(0, 0, 3, 3) = R * DiagonalMatrix<T, 3>(T(C(idj, 0)), T(C(idj, 1)), T(C(idj, 2)));
		Ms.block(idj * 4, 0, 4, 4) = mod_.m_M_l2pl.block(idj * 4, 0, 4, 4).cast<T>()*Mthis;
		outT.block(idj * 3, 0, 3, 4) = Ms.block(idj * 4, 0, 3, 4)*mod_.m_M_w2l.block(idj * 4, 0, 4, 4).cast<T>();
		outJoint.block(idj * 3, 0, 3, 4) = Ms.block(idj * 4, 0, 3, 4);

		for (int idji = 1; idji < mod_.NUM_JOINTS; idji++) {
			idj = mod_.update_inds_(idji);
			int ipar = mod_.parents_(idj);

			ceres::EulerAnglesToRotationMatrix(pose + idj * 3, 3, Rr.data());
			// Mthis.block(0, 0, 3, 3) = Rr * DiagonalMatrix<T, 3>(T(C(idj, 0)), T(C(idj, 0)), T(C(idj, 0)));
			Mthis.block(0, 0, 3, 3) = Rr * DiagonalMatrix<T, 3>(T(C(idj, 0)), T(1), T(1));

			Ms.block(idj * 4, 0, 4, 4) = Ms.block(ipar * 4, 0, 4, 4)*mod_.m_M_l2pl.block(idj * 4, 0, 4, 4).cast<T>()*Mthis;
			outT.block(idj * 3, 0, 3, 4) = Ms.block(idj * 4, 0, 3, 4)*mod_.m_M_w2l.block(idj * 4, 0, 4, 4).cast<T>();

			// This is the joint position NOT THE TRANSFORM!
			outJoint.block(idj * 3, 0, 3, 4) = Ms.block(idj * 4, 0, 3, 4);
		}
		return true;
	}

	const HandModel &mod_;
};

/*
struct ForwardKinematics {
	ForwardKinematics(const HandModel &mod)
		: mod_(mod) {}

	void forward(const double* coeffs,
		const double* pose,  // HandModel::NUM_JOINTS*3
		double* transforms // (SMPLModel::NUM_JOINTS)*3*4
		) const
	{
		using namespace Eigen;
		Map< const Matrix<double, HandModel::NUM_JOINTS, 3, RowMajor> > C(coeffs);

		Map< Matrix<double, 3 * HandModel::NUM_JOINTS, 4, RowMajor> > outT(transforms);		//Transforms
		Map< Matrix<double, 3 * HandModel::NUM_JOINTS, 4, RowMajor> > outJoint(transforms + 3 * HandModel::NUM_JOINTS * 4);		//Joint location
		Matrix<double, Dynamic, 4, RowMajor> Ms(4 * HandModel::NUM_JOINTS, 4);
		Matrix<double, 4, 4, RowMajor> Mthis;

		Matrix<double, 3, 3, ColMajor> R; // Interface with ceres
		Matrix<double, 3, 3, RowMajor> Rr; // Interface with ceres
		Mthis.setZero();
		Mthis(3, 3) = double(1.0);
		Ms.setZero();
		int idj = mod_.update_inds_(0);
		// std::cout << "first" << idj << "\n";
		ceres::AngleAxisToRotationMatrix(pose + idj * 3, R.data());
		Mthis.block(0, 0, 3, 3) = R * DiagonalMatrix<double, 3>(double(C(idj, 0)), double(C(idj, 1)), double(C(idj, 2)));
		Ms.block(idj * 4, 0, 4, 4) = mod_.m_M_l2pl.block(idj * 4, 0, 4, 4)*Mthis;
		outT.block(idj * 3, 0, 3, 4) = Ms.block(idj * 4, 0, 3, 4)*mod_.m_M_w2l.block(idj * 4, 0, 4, 4);
		outJoint.block(idj * 3, 0, 3, 4) = Ms.block(idj * 4, 0, 3, 4);
		// std::cout << "this\n" << Ms.block(idj*4, 0, 4, 4) <<  "\nl2pl\n" << mod_.m_M_l2pl.block(idj * 4, 0, 4, 4) << std::endl;

		for (int idji = 1; idji < mod_.NUM_JOINTS; idji++) {
			idj = mod_.update_inds_(idji);
			int ipar = mod_.parents_(idj);

			// std::cout << "idj" << idj << " " << ipar << "\n";

			ceres::EulerAnglesToRotationMatrix(pose + idj * 3, 3, Rr.data());
			// Mthis.block(0, 0, 3, 3) = Rr * DiagonalMatrix<double, 3>(double(C(idj, 0)), double(C(idj, 1)), double(C(idj, 2)));
			Mthis.block(0, 0, 3, 3) = Rr * DiagonalMatrix<double, 3>(double(C(idj, 0)), double(1), double(1));

			Ms.block(idj * 4, 0, 4, 4) = Ms.block(ipar * 4, 0, 4, 4)*mod_.m_M_l2pl.block(idj * 4, 0, 4, 4)*Mthis;
			// std::cout << "this\n" << Ms.block(idj*4, 0, 4, 4) << "\npar\n" << Ms.block(ipar * 4, 0, 4, 4) <<  "\nl2pl\n" << mod_.m_M_l2pl.block(idj * 4, 0, 4, 4) << std::endl;
			outT.block(idj * 3, 0, 3, 4) = Ms.block(idj * 4, 0, 3, 4)*mod_.m_M_w2l.block(idj * 4, 0, 4, 4);

			// This is the joint position NOT THE TRANSFORM!
			outJoint.block(idj * 3, 0, 3, 4) = Ms.block(idj * 4, 0, 3, 4);
		}
	}

	const HandModel &mod_;
};
*/

struct PoseToTransformsNoLR_Eulers_adamModel {
	PoseToTransformsNoLR_Eulers_adamModel(const TotalModel &mod)
		: mod_(mod)
	{
	}

	template <typename T>
	bool operator()(const T* const pose,  // SMPLModel::NUM_JOINTS*3
		const T* const joints,  // (SMPLModel::NUM_JOINTS)*3
		T* transforms // (SMPLModel::NUM_JOINTS)*3*4
		) const
	{
		using namespace Eigen;
		Map< const Matrix<T, TotalModel::NUM_JOINTS, 3, RowMajor> > J(joints);
		Map< Matrix<T, 3 * TotalModel::NUM_JOINTS, 4, RowMajor> > outT(transforms);
		Map< Matrix<T, TotalModel::NUM_JOINTS, 3, RowMajor> > outJoint(transforms + 3 * TotalModel::NUM_JOINTS * 4);
		Matrix<T, Dynamic, 4, RowMajor> Ms(4 * TotalModel::NUM_JOINTS, 4);
		// Matrix<T, 3, 3, ColMajor> R; // Interface with ceres
		Matrix<T, 3, 3, RowMajor> R; // Interface with ceres

		ceres::AngleAxisToRotationMatrix(pose, R.data());
		Ms.setZero();
		Ms.block(0, 0, 3, 3) = R;
		Ms(0, 3) = J(0, 0);
		Ms(1, 3) = J(0, 1);
		Ms(2, 3) = J(0, 2);
		Ms(3, 3) = T(1.0);

		for (int idj = 1; idj < mod_.NUM_JOINTS; idj++)
		{
			int ipar = mod_.m_parent[idj];
			//std::cout << idj << " " << ipar << "\n\n";
			// ceres::AngleAxisToRotationMatrix(pose + idj * 3, R.data());

			T angles[3];
			angles[0] = pose[idj * 3];
			angles[1] = pose[idj * 3 + 1];
			angles[2] = pose[idj * 3 + 2];

			//Freezing joints here  //////////////////////////////////////////////////////
			if (idj == 10 || idj == 11)	//foot ends
			{
				//R.setIdentity();
				angles[0] = T(0.0);
				angles[1] = T(0.0);
				angles[2] = T(0.0);
			}
			if (idj == 7 || idj == 8)	//foot ankle. Restrict side movement
			{
				angles[2] = T(0.0);
			}

			if (idj == 24 || idj == 27 || idj == 28 || idj == 31 || idj == 32 || idj == 35 || idj == 26 || idj == 39 || idj == 40)	//all hands
			{
				angles[0] = T(0.0);
				angles[1] = T(0.0);
			}

			if (idj == 44 || idj == 47 || idj == 48 || idj == 51 || idj == 52 || idj == 55 || idj == 56 || idj == 59 || idj == 60)	//all hands
			{
				angles[0] = T(0.0);
				angles[1] = T(0.0);
			}
			ceres::EulerAnglesToRotationMatrix(angles, 3, R.data());

			Ms.block(idj * 4, 0, 3, 3) = Ms.block(ipar * 4, 0, 3, 3)*R;
			Ms.block(idj * 4, 3, 3, 1) = Ms.block(ipar * 4, 3, 3, 1) +
										Ms.block(ipar * 4, 0, 3, 3)*(J.row(idj).transpose() - J.row(ipar).transpose());
			Ms(idj * 4 + 3, 3) = T(1.0);
		}
		for (int idj = 0; idj < mod_.NUM_JOINTS; idj++) {
			Ms.block(idj * 4, 3, 3, 1) -= Ms.block(idj * 4, 0, 3, 3)*J.row(idj).transpose();
		}
		for (int idj = 0; idj < mod_.NUM_JOINTS; idj++) {
			outT.block(idj * 3, 0, 3, 4) = Ms.block(idj * 4, 0, 3, 4);
			outJoint.row(idj) = (Ms.block(idj * 4, 0, 3, 3) * J.row(idj).transpose() + Ms.block(idj * 4, 3, 3, 1)).transpose();
		}
		return true;
	}

	const TotalModel &mod_;
};

class PoseToTransformsNoLR_Eulers_adamModel_withDiff: public ceres::CostFunction
{
public:
	PoseToTransformsNoLR_Eulers_adamModel_withDiff(const TotalModel &mod, const Eigen::Matrix<double, TotalModel::NUM_JOINTS, 3, Eigen::RowMajor>& J0):
		mod_(mod), J0_(J0)
	{
		CostFunction::set_num_residuals(3 * TotalModel::NUM_JOINTS);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(TotalModel::NUM_JOINTS * 3); // SMPL Pose
		// mParentIndexes
		// Resize
		mParentIndexes.resize(mod_.NUM_JOINTS);
		// Rest
        for (auto idj = 1; idj < mod_.NUM_JOINTS; idj++)
        {
            const int ipar = mod_.m_parent[idj];
            mParentIndexes[idj] = std::vector<int>(1, ipar);
            while (mParentIndexes[idj].back() != 0)
                mParentIndexes[idj].emplace_back(mod_.m_parent[mParentIndexes[idj].back()]);
            // Sort by index
            std::sort(mParentIndexes[idj].begin(), mParentIndexes[idj].end());
        }
	}
	virtual ~PoseToTransformsNoLR_Eulers_adamModel_withDiff() {}

	virtual bool Evaluate(double const* const* parameters,
		double* residuals,
		double** jacobians) const;

	const TotalModel &mod_;
	const Eigen::Matrix<double, TotalModel::NUM_JOINTS, 3, Eigen::RowMajor>& J0_;
    std::vector<std::vector<int>> mParentIndexes;
};

}
