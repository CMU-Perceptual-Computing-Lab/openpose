#include "totalmodel.h"
#include "handm.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <unsupported/Eigen/KroneckerProduct>
#include <cassert>
#define SMPL_VIS_SCALING 100.0f

struct CoeffsParameterNorm {
	CoeffsParameterNorm(int num_parameters)
		: num_parameters_(num_parameters) {}

	template <typename T>
	inline bool operator()(const T* const p,
		T* residuals) const {
			for (int i = 0; i < num_parameters_; i++) 
			{
				residuals[i] = T(1.0)*p[i];
			}
		return true;
	}
	const double num_parameters_;
};

struct CoeffsParameterLogNorm {
	CoeffsParameterLogNorm(int num_parameters)
		: num_parameters_(num_parameters) {}

	template <typename T>
	inline bool operator()(const T* const p,
		T* residuals) const {
			for (int i = 0; i < num_parameters_; i++) 
			{
				residuals[i] = T(1.0)*log(p[i]);
			}
		return true;
	}
	const double num_parameters_;
};

struct Hand3DCostPose
{
	Hand3DCostPose(smpl::HandModel &handm,
		bool is_left,
		Eigen::MatrixXd &Joints)
		: handm_(handm), is_left_(is_left), Joints_(Joints) {}

	template <typename T>
	bool operator()(
		const T* const trans,
		const T* const pose,  // SMPLModel::NUM_JOINTS*3
		const T* const coeffs,
		T* residuals) 
		const 
	{
		using namespace Eigen;
		// using namespace kinoptic;
		using namespace smpl;

		Map< const Matrix<T, 3, 1> > Tr(trans);
		Map< const Matrix<T, HandModel::NUM_JOINTS, 3, RowMajor> > P(pose);
		Map< const Matrix<T, HandModel::NUM_JOINTS, 3, RowMajor> > C(coeffs);

		Matrix<T, Dynamic, 4, RowMajor>  outJoint(3 * HandModel::NUM_JOINTS,4);		//Joint location

		Matrix<T, Dynamic, 4, RowMajor> Ms(4 * HandModel::NUM_JOINTS, 4);
		Matrix<T, 4, 4, RowMajor> Mthis;
		Matrix<T, 3, 3, ColMajor> R; // Interface with ceres
		Matrix<T, 3, 3, RowMajor> Rr; // Interface with ceres
		Mthis.setZero();
		Mthis(3, 3) = T(1.0);
		Ms.setZero();
		int idj = handm_.update_inds_(0);
		ceres::AngleAxisToRotationMatrix(pose + idj * 3, R.data());
		Mthis.block(0, 0, 3, 3) = R * DiagonalMatrix<T, 3>(T(C(idj, 0)), T(C(idj, 0)), T(C(idj, 0)));
		// Mthis.block(0, 0, 3, 3) = R * DiagonalMatrix<T, 3>(T(C(idj, 0)), T(C(idj, 1)), T(C(idj, 2)));
		Ms.block(idj * 4, 0, 4, 4) = handm_.m_M_l2pl.block(idj * 4, 0, 4, 4).cast<T>()*Mthis;
		outJoint.block(idj * 3, 0, 3, 4) = Ms.block(idj * 4, 0, 3, 4);

		int idji = 0;
		residuals[idj * 3 + 0] = outJoint(idj * 3, 3) + Tr(0, 0)  - T(Joints_(0, idji)) ;
		residuals[idj * 3 + 1] = outJoint(idj * 3 + 1, 3) + Tr(1, 0)  - T(Joints_(1, idji));
		residuals[idj * 3 + 2] = outJoint(idj * 3 + 2, 3) + Tr(2, 0) - T(Joints_(2, idji)) ;

		for (int idji = 1; idji < handm_.NUM_JOINTS; idji++)
		{
			idj = handm_.update_inds_(idji);		//mine -> matlab
			int ipar = handm_.parents_(idj);

			ceres::EulerAnglesToRotationMatrix(pose + idj * 3, 3, Rr.data());
			// Mthis.block(0, 0, 3, 3) = Rr * DiagonalMatrix<T, 3>(T(C(idj, 0)), T(C(idj, 0)), T(C(idj, 0)));
			Mthis.block(0, 0, 3, 3) = Rr * DiagonalMatrix<T, 3>(T(C(idj, 0)), T(1), T(1));

			Ms.block(idj * 4, 0, 4, 4) = Ms.block(ipar * 4, 0, 4, 4)*handm_.m_M_l2pl.block(idj * 4, 0, 4, 4).cast<T>()*Mthis;
			outJoint.block(idj * 3, 0, 3, 4) = Ms.block(idj * 4, 0, 3, 4);
		}

		for (int idji = 1; idji < handm_.NUM_JOINTS; idji++)
		{
			idj = handm_.m_jointmap_pm2model(idji);  // joint_order vs update_inds_ // joint_order(SMC -> idj)
			{
				residuals[idj * 3 + 0] = outJoint(idj * 3, 3) + Tr(0, 0) - T(Joints_(0, idji));
				residuals[idj * 3 + 1] = outJoint(idj * 3 + 1, 3) + Tr(1, 0) - T(Joints_(1, idji));
				residuals[idj * 3 + 2] = outJoint(idj * 3 + 2, 3) + Tr(2, 0) - T(Joints_(2, idji));
			}
		}
		return true;
	}

	smpl::HandModel &handm_;
	bool is_left_;
	Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> Joints_;
};

struct Hand2DProjectionCost
{
	Hand2DProjectionCost(smpl::HandModel &handm, Eigen::MatrixXd &Joints2d, Eigen::MatrixXd &K): handm_(handm), Joints2d_(Joints2d), K_(K) { }
	smpl::HandModel &handm_;
	Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor> Joints2d_;
	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_;

	template<typename T>
	bool operator()(const T* const trans, const T* const pose, const T* const coeffs, T* residuals) const
	{
		using namespace Eigen;
		using namespace smpl;
		Map< const Matrix<T, 3, 1> > Tr(trans);
		Map< const Matrix<T, HandModel::NUM_JOINTS, 3, RowMajor> > P(pose);
		Map< const Matrix<T, HandModel::NUM_JOINTS, 3, RowMajor> > C(coeffs);
		Matrix<T, 3, Dynamic, RowMajor>  outJoint(3, HandModel::NUM_JOINTS);		//Joint location
		Matrix<T, 2, Dynamic, RowMajor>  outJoint2d(2, HandModel::NUM_JOINTS);		//Joint location
		Matrix<T, 3, Dynamic, RowMajor> Km(3, HandModel::NUM_JOINTS);
		Matrix<T, Dynamic, 4, RowMajor> Ms(4 * HandModel::NUM_JOINTS, 4);
		Matrix<T, 4, 4, RowMajor> Mthis;
		Matrix<T, 3, 3, ColMajor> R; // Interface with ceres
		Matrix<T, 3, 3, RowMajor> Rr; // Interface with ceres

		Mthis.setZero();
		Mthis(3, 3) = T(1.0);
		Ms.setZero();

		int idj = handm_.update_inds_(0);
		ceres::AngleAxisToRotationMatrix(pose + idj * 3, R.data());
		Mthis.block(0, 0, 3, 3) = R * DiagonalMatrix<T, 3>(T(C(idj, 0)), T(C(idj, 0)), T(C(idj, 0)));
		// Mthis.block(0, 0, 3, 3) = R * DiagonalMatrix<T, 3>(T(C(idj, 0)), T(C(idj, 1)), T(C(idj, 2)));
		Ms.block(idj * 4, 0, 4, 4) = handm_.m_M_l2pl.block(idj * 4, 0, 4, 4).cast<T>()*Mthis;
		outJoint.block(0, idj, 3, 1) = Ms.block(idj * 4, 3, 3, 1) + Tr; // add translation here!

		for (int idji = 1; idji < handm_.NUM_JOINTS; idji++)
		{
			idj = handm_.update_inds_(idji);		//mine -> matlab
			int ipar = handm_.parents_(idj);
			ceres::EulerAnglesToRotationMatrix(pose + idj * 3, 3, Rr.data());
			// Mthis.block(0, 0, 3, 3) = Rr * DiagonalMatrix<T, 3>(T(C(idj, 0)), T(C(idj, 0)), T(C(idj, 0)));
			Mthis.block(0, 0, 3, 3) = Rr * DiagonalMatrix<T, 3>(T(C(idj, 0)), T(1), T(1));
			Ms.block(idj * 4, 0, 4, 4) = Ms.block(ipar * 4, 0, 4, 4)*handm_.m_M_l2pl.block(idj * 4, 0, 4, 4).cast<T>()*Mthis;
			outJoint.block(0, idj, 3, 1) = Ms.block(idj * 4, 3, 3, 1) + Tr; // add translation here! 
		}

		outJoint = outJoint * T(SMPL_VIS_SCALING);
		Km = K_.cast<T>() * outJoint;

		for(int idji = 0; idji < handm_.NUM_JOINTS; idji++)
		{
			outJoint2d(0, idji) = Km(0, idji) / Km(2, idji);
			outJoint2d(1, idji) = Km(1, idji) / Km(2, idji);
		}

		int idji = 0;
		idj = handm_.update_inds_(idji);
		residuals[idj * 2 + 0] = outJoint2d(0, idj) - T(Joints2d_(0, idji));
		residuals[idj * 2 + 1] = outJoint2d(1, idj) - T(Joints2d_(1, idji));
		for (int idji = 1; idji < handm_.NUM_JOINTS; idji++)
		{
			idj = handm_.m_jointmap_pm2model(idji);  // joint_order vs update_inds_ // joint_order(SMC -> idj)
			residuals[idj * 2 + 0] = outJoint2d(0, idj) - T(Joints2d_(0, idji));
			residuals[idj * 2 + 1] = outJoint2d(1, idj) - T(Joints2d_(1, idji));
			// residuals[idj * 2 + 0] = T(0);
			// residuals[idj * 2 + 1] = T(0);
		}
		return true;
	}
};

struct AdamBodyPoseParamPrior {
	AdamBodyPoseParamPrior(int num_parameters)
		: num_parameters_(num_parameters) {}

	template <typename T>
	bool operator()(const T* const p, T* residuals) const {
		//Put stronger prior for spine body joints
		for (int i = 0; i < num_parameters_; i++)
		{
			if (i >= 0 && i < 3)
			{
				residuals[i] = T(0.0);
				// residuals[i] = T(3)*p[i];
			}
			else if ((i >= 9 && i < 12) || (i >= 18 && i < 21) || (i >= 27 && i < 30))
			{
				residuals[i] = T(12)*p[i];
			}
			else if ((i >= 42 && i < 45) || (i >= 39 && i < 41))
				residuals[i] = T(2)*p[i];
			else if (i >= 54 && i < 60)		//18, 19 (elbows)
			{
				if (i == 54 || i == 57)		//twist 
					residuals[i] = T(3)*p[i];
				else
					residuals[i] = T(0.1)*p[i];
			}
			else if (i >= 60 && i < 66)		//20, 21 (wrist)
			{
				if (i == 60 || i == 63)		//twist of wrist
					residuals[i] = T(1)*p[i];
				else
					residuals[i] = T(0.1)*p[i];
			}
			else if (i >= 66) //fingers
				residuals[i] = T(1.0)*p[i];
			else
				residuals[i] = T(1.0)*p[i];;// *p[i];	//Do nothing*/
		}
		return true;
	}
	const double num_parameters_;
};

template<typename Derived, int rows, int cols, int option>
void projection_jacobian(Eigen::Map<Eigen::Matrix<Derived, rows, cols, option>>& dPdI, Eigen::Matrix<Derived, rows, cols, option>& dJdI, double XYZ[3], double* pK_, int offsetP, int offsetJ, float weight=1.0f)
{
	// Dx/Dt = dx/dX * dX/dt + dx/dY * dY/dt + dx/dZ * dZ/dt
	double X = XYZ[0], Y = XYZ[1], Z = XYZ[2];
	dPdI.row(offsetP + 0) += weight * (pK_[0] / Z * dJdI.row(offsetJ + 0)
						 + pK_[1] / Z * dJdI.row(offsetJ + 1)
						 - (pK_[0] * X + pK_[1] * Y) / Z / Z * dJdI.row(offsetJ + 2));
	dPdI.row(offsetP + 1) += weight * (pK_[4] / Z * dJdI.row(offsetJ + 1)
						 - pK_[4] * Y / Z / Z * dJdI.row(offsetJ + 2));
}

template<typename Derived, int rows, int cols, int option, typename DerivedS, int optionS>
void projection_jacobian(Eigen::Map<Eigen::Matrix<Derived, rows, cols, option>>& dPdI, const Eigen::SparseMatrix<DerivedS, optionS>& dJdI, double XYZ[3], double* pK_, int offsetP, int offsetJ, float weight=1.0f)
{
	// Dx/Dt = dx/dX * dX/dt + dx/dY * dY/dt + dx/dZ * dZ/dt
	double X = XYZ[0], Y = XYZ[1], Z = XYZ[2];
	dPdI.row(offsetP + 0) += weight * (pK_[0] / Z * dJdI.row(offsetJ + 0)
						 + pK_[1] / Z * dJdI.row(offsetJ + 1)
						 - (pK_[0] * X + pK_[1] * Y) / Z / Z * dJdI.row(offsetJ + 2));
	dPdI.row(offsetP + 1) += weight * (pK_[4] / Z * dJdI.row(offsetJ + 1)
						 - pK_[4] * Y / Z / Z * dJdI.row(offsetJ + 2));
}

class CostFunc_Adam_keypoints_withFoot: public ceres::CostFunction
{
public:
	CostFunc_Adam_keypoints_withFoot(TotalModel &adam,
		Eigen::MatrixXd &Joints,
		Eigen::MatrixXd &rFoot,		//3x2 //Heel, Toe
		Eigen::MatrixXd &lFoot,
		Eigen::MatrixXd &faceJoints,
		Eigen::MatrixXd &lHandJoints,
		Eigen::MatrixXd &rHandJoints,
		double* K = NULL,
		bool fit_face=false,
		uint projection=0u
		)		//3x2, //Heel, Toe
		: m_adam(adam), m_bodyJoints(Joints), m_rfoot_joints(rFoot), m_lfoot_joints(lFoot), m_FaceJoints(faceJoints),
			m_lHandJoints(lHandJoints), m_rHandJoints(rHandJoints), pK_(K), projection_(projection), fit_face_(fit_face)
	{   
		if (projection_ == 0u) res_dim = 3;
		else if(projection_ == 1u) res_dim = 2;
		else if(projection_ == 2u) res_dim = 5;
		else {printf("Error: Invalid projection type.\n"); exit(1);}
		SetupCost();
	}
	virtual ~CostFunc_Adam_keypoints_withFoot() {}

	void SetupCost()
	{
		using namespace cv;
		using namespace Eigen;

		joint_only = false;
		double BODYJOINT_WEIGHT_Strong = 1;
		double BODYJOINT_WEIGHT = 1;
		double HANDJOINT_WEIGHT = 1;
		WEAK_WEIGHT = 1;		//for foot and face
		// WEAK_WEIGHT = 0;		//for foot and face
		// Joint correspondences  ////////////////////////////////////////////////
		std::vector<Eigen::Triplet<double> > IJV;

		m_nCorrespond_adam2joints = m_adam.m_indices_jointConst_adamIdx.rows();
		std::cout << "m_nCorrespond_adam2joints " << m_nCorrespond_adam2joints << std::endl;

		if (m_lHandJoints.size() > 0)
			m_nCorrespond_adam2joints += m_adam.m_correspond_adam2lHand_adamIdx.rows();
		std::cout << "m_nCorrespond_adam2joints " << m_nCorrespond_adam2joints << std::endl;

		if (m_rHandJoints.size() > 0)
			m_nCorrespond_adam2joints += m_adam.m_correspond_adam2rHand_adamIdx.rows();
		std::cout << "m_nCorrespond_adam2joints " << m_nCorrespond_adam2joints << std::endl;

		IJV.reserve(m_nCorrespond_adam2joints * 3);			//A sparse selection (permutation) matrix to select parameters from SMPL joints order
		m_targetPts.resize(m_nCorrespond_adam2joints * 5);			//reordered target joint (CPM)
		m_targetPts.setZero();
		m_targetPts_weight.resize(m_nCorrespond_adam2joints * res_dim);
		m_targetPts_weight_buffer.resize(m_nCorrespond_adam2joints * res_dim);

		int offset = 0;
		for (int ic = 0; ic<m_adam.m_indices_jointConst_adamIdx.rows(); ic++)
		{
			IJV.push_back(Triplet<double>(ic * 3 + 0, m_adam.m_indices_jointConst_adamIdx(ic) * 3 + 0, 1.0));
			IJV.push_back(Triplet<double>(ic * 3 + 1, m_adam.m_indices_jointConst_adamIdx(ic) * 3 + 1, 1.0));
			IJV.push_back(Triplet<double>(ic * 3 + 2, m_adam.m_indices_jointConst_adamIdx(ic) * 3 + 2, 1.0));
			m_targetPts(ic * 5 + 0) = m_bodyJoints(0, m_adam.m_indices_jointConst_smcIdx(ic));		//Joints_: 3 x #Joints matrix
			m_targetPts(ic * 5 + 1) = m_bodyJoints(1, m_adam.m_indices_jointConst_smcIdx(ic));
			m_targetPts(ic * 5 + 2) = m_bodyJoints(2, m_adam.m_indices_jointConst_smcIdx(ic));
			m_targetPts(ic * 5 + 3) = m_bodyJoints(3, m_adam.m_indices_jointConst_smcIdx(ic));
			m_targetPts(ic * 5 + 4) = m_bodyJoints(4, m_adam.m_indices_jointConst_smcIdx(ic));

			int smcjoint = m_adam.m_indices_jointConst_smcIdx(ic);

			if (smcjoint == 4 || smcjoint == 10 || smcjoint == 3 || smcjoint == 9 || smcjoint == 7 || smcjoint == 13)
			{
				for (int i = 0; i < res_dim; i++)
				{
					m_targetPts_weight[ic * res_dim + i] = BODYJOINT_WEIGHT_Strong;
					m_targetPts_weight_buffer[ic * res_dim + i] = BODYJOINT_WEIGHT_Strong;
				}
			}
			else
			{
				for (int i = 0; i < res_dim; i++)
				{
					m_targetPts_weight[ic * res_dim + i] = BODYJOINT_WEIGHT;
					m_targetPts_weight_buffer[ic * res_dim + i] = BODYJOINT_WEIGHT;
				}
			}
		}
		offset += m_adam.m_indices_jointConst_adamIdx.rows();

		m_spineResIdx = -1;

		//Hands 
		if (m_lHandJoints.size() > 0)
		{
			//correspondences.reserve(nCorrespond_);
			for (int ic = 0; ic < m_adam.m_correspond_adam2lHand_adamIdx.rows(); ic++)
			{
				IJV.push_back(Triplet<double>(offset * 3 + ic * 3 + 0, m_adam.m_correspond_adam2lHand_adamIdx(ic) * 3 + 0, 1.0));
				IJV.push_back(Triplet<double>(offset * 3 + ic * 3 + 1, m_adam.m_correspond_adam2lHand_adamIdx(ic) * 3 + 1, 1.0));
				IJV.push_back(Triplet<double>(offset * 3 + ic * 3 + 2, m_adam.m_correspond_adam2lHand_adamIdx(ic) * 3 + 2, 1.0));
				m_targetPts(offset * 5 + ic * 5 + 0) = m_lHandJoints(0, m_adam.m_correspond_adam2lHand_lHandIdx(ic));		//Joints_: 3 x #Joints matrix
				m_targetPts(offset * 5 + ic * 5 + 1) = m_lHandJoints(1, m_adam.m_correspond_adam2lHand_lHandIdx(ic));
				m_targetPts(offset * 5 + ic * 5 + 2) = m_lHandJoints(2, m_adam.m_correspond_adam2lHand_lHandIdx(ic));
				m_targetPts(offset * 5 + ic * 5 + 3) = m_lHandJoints(3, m_adam.m_correspond_adam2lHand_lHandIdx(ic));
				m_targetPts(offset * 5 + ic * 5 + 4) = m_lHandJoints(4, m_adam.m_correspond_adam2lHand_lHandIdx(ic));

				for (int i = 0; i < res_dim; i++)
				{
					m_targetPts_weight[(ic + offset) * res_dim + i] = HANDJOINT_WEIGHT;
					m_targetPts_weight_buffer[(ic + offset) * res_dim + i] = HANDJOINT_WEIGHT;
				}
			}
			offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();
		}

		if (m_rHandJoints.size() > 0)
		{
			//correspondences.reserve(nCorrespond_);
			for (int ic = 0; ic < m_adam.m_correspond_adam2rHand_adamIdx.rows(); ic++)
			{
				IJV.push_back(Triplet<double>(offset * 3 + ic * 3 + 0, m_adam.m_correspond_adam2rHand_adamIdx(ic) * 3 + 0, 1.0));
				IJV.push_back(Triplet<double>(offset * 3 + ic * 3 + 1, m_adam.m_correspond_adam2rHand_adamIdx(ic) * 3 + 1, 1.0));
				IJV.push_back(Triplet<double>(offset * 3 + ic * 3 + 2, m_adam.m_correspond_adam2rHand_adamIdx(ic) * 3 + 2, 1.0));
				m_targetPts(offset * 5 + ic * 5 + 0) = m_rHandJoints(0, m_adam.m_correspond_adam2rHand_rHandIdx(ic));		//Joints_: 3 x #Joints matrix
				m_targetPts(offset * 5 + ic * 5 + 1) = m_rHandJoints(1, m_adam.m_correspond_adam2rHand_rHandIdx(ic));
				m_targetPts(offset * 5 + ic * 5 + 2) = m_rHandJoints(2, m_adam.m_correspond_adam2rHand_rHandIdx(ic));
				m_targetPts(offset * 5 + ic * 5 + 3) = m_rHandJoints(3, m_adam.m_correspond_adam2rHand_rHandIdx(ic));
				m_targetPts(offset * 5 + ic * 5 + 4) = m_rHandJoints(4, m_adam.m_correspond_adam2rHand_rHandIdx(ic));

				for (int i = 0; i < res_dim; i++)
				{
					m_targetPts_weight[(ic + offset) * res_dim + i] = HANDJOINT_WEIGHT;
					m_targetPts_weight_buffer[(ic + offset) * res_dim + i] = HANDJOINT_WEIGHT;
				}
			}

		}

		// Precomputing
		m_Ajoints.resize(m_nCorrespond_adam2joints * 3, TotalModel::NUM_JOINTS * 3);
		m_Ajoints.reserve(m_nCorrespond_adam2joints * 3);
		m_Ajoints.setFromTriplets(IJV.begin(), IJV.end());
		m_mat_vert2joints = m_Ajoints*m_adam.m_J_reg_smc;

		// //////////////////////////////////////////////////////////////////////////
		// // Surface point correspondences  ////////////////////////////////////////////////
		// if (m_rfoot_joints.cols() > 0)		 //3xNumofJoint (bigToe, smallToe, heel)
		// {
		// 	cv::Point3d targetPt = cv::Point3d(m_rfoot_joints(0, 0), m_rfoot_joints(1, 0), m_rfoot_joints(2, 0));
		// 	corres_vertex2targetpt.push_back(std::make_pair(14238, targetPt));		//right_bigToe		//-1 due to 1based in matlab
		// 	targetPt = cv::Point3d(m_rfoot_joints(0, 1), m_rfoot_joints(1, 1), m_rfoot_joints(2, 1));
		// 	corres_vertex2targetpt.push_back(std::make_pair(14288, targetPt));		//right_smallToe
		// 	targetPt = cv::Point3d(m_rfoot_joints(0, 2), m_rfoot_joints(1, 2), m_rfoot_joints(2, 2));
		// 	corres_vertex2targetpt.push_back(std::make_pair(14357, targetPt));		//right_heel
		// 	targetPt = cv::Point3d(m_rfoot_joints(0, 2), m_rfoot_joints(1, 2), m_rfoot_joints(2, 2));
		// 	corres_vertex2targetpt.push_back(std::make_pair(14361, targetPt));		//right_heel
		// }
		// if (m_lfoot_joints.cols() > 0)
		// {
		// 	cv::Point3d  targetPt = cv::Point3d(m_lfoot_joints(0, 1), m_lfoot_joints(1, 0), m_lfoot_joints(2, 0));
		// 	corres_vertex2targetpt.push_back(std::make_pair(12239, targetPt));		//left big toe
		// 	targetPt = cv::Point3d(m_lfoot_joints(0, 1), m_lfoot_joints(1, 1), m_lfoot_joints(2, 1));
		// 	corres_vertex2targetpt.push_back(std::make_pair(12289, targetPt));		//left small toe
		// 	targetPt = cv::Point3d(m_lfoot_joints(0, 2), m_lfoot_joints(1, 2), m_lfoot_joints(2, 2));
		// 	corres_vertex2targetpt.push_back(std::make_pair(12368, targetPt));		//left heel
		// 	targetPt = cv::Point3d(m_lfoot_joints(0, 2), m_lfoot_joints(1, 2), m_lfoot_joints(2, 2));
		// 	corres_vertex2targetpt.push_back(std::make_pair(12357, targetPt));		//left heel
		// }

		// Faces.  m_FaceJoints: 3x face70JointsNum
		if (this->fit_face_ && m_FaceJoints.cols() > 0)
		{
			for (int r = 0; r < m_adam.m_correspond_adam2face70_adamIdx.rows(); ++r)
			{
				int adamVertexIdx = m_adam.m_correspond_adam2face70_adamIdx(r);
				int face70ID = m_adam.m_correspond_adam2face70_face70Idx(r);

				if (face70ID < 0)
					continue;

				if (m_FaceJoints(0, face70ID) == 0.0)
					continue;

				std::vector<double> targetPt = {m_FaceJoints(0, face70ID), m_FaceJoints(1, face70ID), m_FaceJoints(2, face70ID), m_FaceJoints(3, face70ID), m_FaceJoints(4, face70ID)};
				corres_vertex2targetpt.push_back(std::make_pair(adamVertexIdx, targetPt));
			}
		}

		//CoCo Faces
		if (m_bodyJoints(0, 1) != 0.0)		//nose
		{
			std::vector<double> targetPt = {m_bodyJoints(0, 1), m_bodyJoints(1, 1), m_bodyJoints(2, 1), m_bodyJoints(3, 1), m_bodyJoints(4, 1)};
			corres_vertex2targetpt.push_back(std::make_pair(8130, targetPt));		

		}
		if (m_bodyJoints(0, 16) != 0.0)		//left ear
		{
			std::vector<double> targetPt = {m_bodyJoints(0, 16), m_bodyJoints(1, 16), m_bodyJoints(2, 16), m_bodyJoints(3, 16), m_bodyJoints(4, 16)};
			corres_vertex2targetpt.push_back(std::make_pair(6970, targetPt));		
		}

		if (m_bodyJoints(0, 18) != 0.0)		//right ear
		{
			std::vector<double> targetPt = {m_bodyJoints(0, 18), m_bodyJoints(1, 18), m_bodyJoints(2, 18), m_bodyJoints(3, 18), m_bodyJoints(4, 18)};
			corres_vertex2targetpt.push_back(std::make_pair(10088, targetPt));		
		}

		m_nCorrespond_adam2pts = corres_vertex2targetpt.size();

		// Other setting ////////////////////////////////////////////////////////////
		SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
		m_dVdt = kroneckerProduct(VectorXd::Ones(TotalModel::NUM_VERTICES), eye3);		//Precomputing
		m_nResiduals = m_nCorrespond_adam2joints * res_dim + m_nCorrespond_adam2pts * res_dim;
		std::cout << "m_nResiduals " << m_nResiduals << std::endl;
		CostFunction::set_num_residuals(m_nResiduals);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(3); // Translation
		parameter_block_sizes->push_back(TotalModel::NUM_JOINTS * 3); // SMPL Pose  
		parameter_block_sizes->push_back(TotalModel::NUM_SHAPE_COEFFICIENTS); // Shape coefficients
		if (this->fit_face_) parameter_block_sizes->push_back(TotalModel::NUM_EXP_BASIS_COEFFICIENTS); // Face expression coefficients
	}

	void toggle_activate(bool limb, bool finger)
	{
		for (int ic = 0; ic<m_adam.m_indices_jointConst_adamIdx.rows(); ic++)
		{
			int smcjoint = m_adam.m_indices_jointConst_smcIdx(ic);
			if (smcjoint != 3 && smcjoint != 6 && smcjoint != 9 && smcjoint != 12)
			{
				for (int i = 0; i < res_dim; i++)
				{
					m_targetPts_weight[ic * res_dim + i] = m_targetPts_weight_buffer[ic * res_dim + i] * double(limb);
				}
			}
		}

		int offset = m_adam.m_indices_jointConst_smcIdx.rows();
		for (int ic = 0; ic < m_adam.m_correspond_adam2lHand_adamIdx.rows(); ic++)
		{
			for (int i = 0; i < res_dim; i++)
			{
				m_targetPts_weight[(ic + offset) * res_dim + i] = m_targetPts_weight_buffer[(ic + offset) * res_dim + i] * double(finger);
			}
		}
		offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();
		for (int ic = 0; ic < m_adam.m_correspond_adam2rHand_adamIdx.rows(); ic++)
		{
			for (int i = 0; i < res_dim; i++)
			{
				m_targetPts_weight[(ic + offset) * res_dim + i] = m_targetPts_weight_buffer[(ic + offset) * res_dim + i] * double(finger);
			}
		}
		// for (int i = 0; i < 53 * 3; i++) std::cout << m_targetPts_weight << std::endl;
	}

	virtual bool Evaluate(double const* const* parameters,
		double* residuals,
		double** jacobians) const
	{
		using namespace Eigen;

		typedef double T;
		const T * t = parameters[0];
		const T * p_eulers = parameters[1];
		const T * c = parameters[2];
		const T * c_faceEx = parameters[3];		//Facial expression

		MatrixXdr V(TotalModel::NUM_VERTICES, 3);
		Map< VectorXd > V_vec(V.data(), TotalModel::NUM_VERTICES * 3);
		Map< const Vector3d > t_vec(t);
		Map< const VectorXd > pose_vec(p_eulers, TotalModel::NUM_JOINTS * 3);
		Map< const VectorXd > coeffs_vec(c, TotalModel::NUM_SHAPE_COEFFICIENTS);
		Map< const VectorXd > coeffs_vec_faceEx(c_faceEx, TotalModel::NUM_EXP_BASIS_COEFFICIENTS);

		Matrix<double, Dynamic, Dynamic, RowMajor> dVdPfr(TotalModel::NUM_VERTICES * 3,
			TotalModel::NUM_JOINTS * 3);
		Matrix<double, Dynamic, Dynamic, RowMajor> dVdcfr(TotalModel::NUM_VERTICES * 3,
			TotalModel::NUM_SHAPE_COEFFICIENTS);
		Matrix<double, Dynamic, Dynamic, RowMajor> dTJdPfr((TotalModel::NUM_JOINTS) * 3, 3 * TotalModel::NUM_JOINTS);
		Matrix<double, Dynamic, Dynamic, RowMajor> dTJdcfr((TotalModel::NUM_JOINTS) * 3, 3 * TotalModel::NUM_SHAPE_COEFFICIENTS);

		Matrix<double, Dynamic, Dynamic, RowMajor> dVdfc(TotalModel::NUM_VERTICES * 3, TotalModel::NUM_EXP_BASIS_COEFFICIENTS);
		dVdfc.setZero();

		Eigen::VectorXd outJointv = adam_reconstruct_withDerivative_eulers(m_adam, c, p_eulers, c_faceEx, V.data(), dVdcfr, dVdPfr, dVdfc, dTJdcfr, dTJdPfr, joint_only, fit_face_);
		// adam_reconstruct_withDerivative_eulers(m_adam, m_facem, c, p_eulers, c_faceEx, V.data(), dVdcfr, dVdPfr, dVdfc);
		V_vec += m_dVdt * t_vec;			//Translation

		Map< VectorXd > res(residuals, m_nResiduals);
		res.setZero();

		// Joint Constraints   //////////////////////
		VectorXd tempJoints(m_mat_vert2joints.rows());
		Eigen::Map< Matrix<T, TotalModel::NUM_JOINTS, 3, RowMajor> > outJoint(outJointv.data());
		if (this->joint_only) 
		{
			// std::cout << outT << std::endl;
			for (int i = 0; i < m_adam.m_indices_jointConst_adamIdx.rows(); i++)
			{
				tempJoints.block(3*i, 0, 3, 1) = outJoint.row(m_adam.m_indices_jointConst_adamIdx(i)).transpose() + t_vec;
			}
			int offset = m_adam.m_indices_jointConst_adamIdx.rows();
			for (int i = 0; i < m_adam.m_correspond_adam2lHand_adamIdx.rows(); i++)
			{
				tempJoints.block(3*(i + offset), 0, 3, 1) = outJoint.row(m_adam.m_correspond_adam2lHand_adamIdx(i)).transpose() + t_vec;
			}
			offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();
			for (int i = 0; i < m_adam.m_correspond_adam2rHand_adamIdx.rows(); i++)
			{
				tempJoints.block(3*(i + offset), 0, 3, 1) = outJoint.row(m_adam.m_correspond_adam2rHand_adamIdx(i)).transpose() + t_vec;
			}
		}
		else
			tempJoints = m_mat_vert2joints * V_vec;
		
		int vertexCons_res_startIdx = m_nCorrespond_adam2joints * res_dim;
		if (projection_ == 0u)
		// Fit 3D joints
		{
			for(int i = 0; i < m_nCorrespond_adam2joints; i++)
			{
				if (m_targetPts.block(5 * i, 0, 3, 1).isZero(0)) continue;
				res.block(3 * i, 0, 3, 1) << tempJoints.block(3 * i, 0, 3, 1) - m_targetPts.block(5 * i, 0, 3, 1);
			}

			for (int j = 0; j < 3 * m_nCorrespond_adam2joints; ++j)
			{
				residuals[j] *= m_targetPts_weight[j];
			}

			// Vertex Constraints //////////////////////
			if (!this->joint_only)
			{
				for (int i = 0; i < corres_vertex2targetpt.size(); ++i)
				{
					if (corres_vertex2targetpt[i].second[0] == 0.0 && corres_vertex2targetpt[i].second[1] == 0.0 && corres_vertex2targetpt[i].second[2] == 0.0) continue;
					res[vertexCons_res_startIdx + 3 * i + 0] = WEAK_WEIGHT * (V_vec(3 * corres_vertex2targetpt[i].first + 0) - corres_vertex2targetpt[i].second[0]); //- m_targetPts(res_offset + 3 * i + 0));
					res[vertexCons_res_startIdx + 3 * i + 1] = WEAK_WEIGHT * (V_vec(3 * corres_vertex2targetpt[i].first + 1) - corres_vertex2targetpt[i].second[1]); //- m_targetPts(res_offset + 3 * i + 1));
					res[vertexCons_res_startIdx + 3 * i + 2] = WEAK_WEIGHT * (V_vec(3 * corres_vertex2targetpt[i].first + 2) - corres_vertex2targetpt[i].second[2]); //- m_targetPts(res_offset + 3 * i + 2));
				}
			}
			else res.block(vertexCons_res_startIdx, 0, 3 * corres_vertex2targetpt.size(), 1).setZero();

			// compute jacobian
			if (jacobians)
			{
				if (jacobians[0])
				{
					// Construct full system.
					Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdt(jacobians[0], m_nResiduals, 3);
					drdt.block(0, 0, 3 * m_nCorrespond_adam2joints, 3) = m_mat_vert2joints * m_dVdt;

					for (int j = 0; j < m_nCorrespond_adam2joints; j++)
					{
						if (m_targetPts.block(5 * j, 0, 3, 1).isZero(0)) drdt.block(3 * j, 0, 3, 3).setZero();
					}

					for (int j = 0; j < 3 * m_nCorrespond_adam2joints; ++j)
					{
						drdt.row(j) *= m_targetPts_weight[j];
					}

					if (!this->joint_only)
					{
						for (int i = 0; i < corres_vertex2targetpt.size(); ++i)
						{
							if (corres_vertex2targetpt[i].second[0] == 0.0 && corres_vertex2targetpt[i].second[1] == 0.0 && corres_vertex2targetpt[i].second[2] == 0.0) continue;
							drdt.row(vertexCons_res_startIdx + 3 * i + 0) = WEAK_WEIGHT * m_dVdt.row(3 * corres_vertex2targetpt[i].first + 0);
							drdt.row(vertexCons_res_startIdx + 3 * i + 1) = WEAK_WEIGHT * m_dVdt.row(3 * corres_vertex2targetpt[i].first + 1);
							drdt.row(vertexCons_res_startIdx + 3 * i + 2) = WEAK_WEIGHT * m_dVdt.row(3 * corres_vertex2targetpt[i].first + 2);
						}
					}
					else
					{
						drdt.block(vertexCons_res_startIdx, 0, 3 * corres_vertex2targetpt.size(), 3).setZero();
					}
				}
				if (jacobians[1])
				{
					Map< Matrix<double, Dynamic, Dynamic, RowMajor> > dr_dPose(jacobians[1],
						m_nResiduals,
						TotalModel::NUM_JOINTS * 3);
					dr_dPose.setZero();
					if (this->joint_only)
					{
						for (int i = 0; i < m_adam.m_indices_jointConst_adamIdx.rows(); i++)
						{
							if (m_targetPts.block(5 * i, 0, 3, 1).isZero(0)) continue;
							dr_dPose.block(3 * i, 0, 3, TotalModel::NUM_POSE_PARAMETERS) = dTJdPfr.block(3 * m_adam.m_indices_jointConst_adamIdx(i), 0, 3, TotalModel::NUM_POSE_PARAMETERS);
						}
						int offset = m_adam.m_indices_jointConst_adamIdx.rows();
						for (int i = 0; i < m_adam.m_correspond_adam2lHand_adamIdx.rows(); i++)
						{
							if (m_targetPts.block(5 * (i + offset), 0, 3, 1).isZero(0)) continue;
							dr_dPose.block(3*(i + offset), 0, 3, TotalModel::NUM_POSE_PARAMETERS) = dTJdPfr.block(3 * m_adam.m_correspond_adam2lHand_adamIdx(i), 0, 3, TotalModel::NUM_POSE_PARAMETERS);
						}
						offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();
						for (int i = 0; i < m_adam.m_correspond_adam2rHand_adamIdx.rows(); i++)
						{
							if (m_targetPts.block(5 * (i + offset), 0, 3, 1).isZero(0)) continue;
							dr_dPose.block(3*(i + offset), 0, 3, TotalModel::NUM_POSE_PARAMETERS) = dTJdPfr.block(3 * m_adam.m_correspond_adam2rHand_adamIdx(i), 0, 3, TotalModel::NUM_POSE_PARAMETERS);
						}
					}
					else
						dr_dPose.block(0, 0, m_mat_vert2joints.rows(), TotalModel::NUM_JOINTS * 3) = m_mat_vert2joints*dVdPfr;

					for (int j = 0; j < 3 * m_nCorrespond_adam2joints; ++j)
					{
						dr_dPose.row(j) *= m_targetPts_weight[j];
					}

					if(!this->joint_only)
					{
						for (int i = 0; i < corres_vertex2targetpt.size(); ++i)
						{
							if (corres_vertex2targetpt[i].second[0] == 0.0 && corres_vertex2targetpt[i].second[1] == 0.0 && corres_vertex2targetpt[i].second[2] == 0.0) continue;
							dr_dPose.row(vertexCons_res_startIdx + 3 * i + 0) = WEAK_WEIGHT* dVdPfr.row(3 * corres_vertex2targetpt[i].first + 0);
							dr_dPose.row(vertexCons_res_startIdx + 3 * i + 1) = WEAK_WEIGHT* dVdPfr.row(3 * corres_vertex2targetpt[i].first + 1);
							dr_dPose.row(vertexCons_res_startIdx + 3 * i + 2) = WEAK_WEIGHT* dVdPfr.row(3 * corres_vertex2targetpt[i].first + 2);
						}
					}
					else
					{
						dr_dPose.block(vertexCons_res_startIdx, 0, 3 * corres_vertex2targetpt.size(), TotalModel::NUM_POSE_PARAMETERS).setZero();
					}
				}
				if (jacobians[2])
				{
					Map< Matrix<double, Dynamic, Dynamic, RowMajor> > dr_dCoeff(jacobians[2],
						m_nResiduals,
						TotalModel::NUM_SHAPE_COEFFICIENTS);
					dr_dCoeff.setZero();
					if (this->joint_only)
					{
						for (int i = 0; i < m_adam.m_indices_jointConst_adamIdx.rows(); i++)
						{
							if (m_targetPts.block(5 * i, 0, 3, 1).isZero(0)) continue;
							dr_dCoeff.block(3 * i, 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS) = dTJdcfr.block(3 * m_adam.m_indices_jointConst_adamIdx(i), 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS);
						}
						int offset = m_adam.m_indices_jointConst_adamIdx.rows();
						for (int i = 0; i < m_adam.m_correspond_adam2lHand_adamIdx.rows(); i++)
						{
							if (m_targetPts.block(5 * (i + offset), 0, 3, 1).isZero(0)) continue;
							dr_dCoeff.block(3*(i + offset), 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS) = dTJdcfr.block(3 * m_adam.m_correspond_adam2lHand_adamIdx(i), 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS);
						}
						offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();
						for (int i = 0; i < m_adam.m_correspond_adam2rHand_adamIdx.rows(); i++)
						{
							if (m_targetPts.block(5 * (i + offset), 0, 3, 1).isZero(0)) continue;
							dr_dCoeff.block(3*(i + offset), 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS) = dTJdcfr.block(3 * m_adam.m_correspond_adam2rHand_adamIdx(i), 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS);
						}
					}
					else dr_dCoeff.block(0, 0, m_mat_vert2joints.rows(), TotalModel::NUM_SHAPE_COEFFICIENTS) = m_mat_vert2joints*dVdcfr;

					for (int j = 0; j < 3 * m_nCorrespond_adam2joints; ++j)
					{
						dr_dCoeff.row(j) *= m_targetPts_weight[j];
					}

					if (!this->joint_only)
					{
						for (int i = 0; i < corres_vertex2targetpt.size(); ++i)
						{
							if (corres_vertex2targetpt[i].second[0] == 0.0 && corres_vertex2targetpt[i].second[1] == 0.0 && corres_vertex2targetpt[i].second[2] == 0.0) continue;
							dr_dCoeff.row(vertexCons_res_startIdx + 3 * i + 0) = WEAK_WEIGHT* dVdcfr.row(3 * corres_vertex2targetpt[i].first + 0);
							dr_dCoeff.row(vertexCons_res_startIdx + 3 * i + 1) = WEAK_WEIGHT* dVdcfr.row(3 * corres_vertex2targetpt[i].first + 1);
							dr_dCoeff.row(vertexCons_res_startIdx + 3 * i + 2) = WEAK_WEIGHT* dVdcfr.row(3 * corres_vertex2targetpt[i].first + 2);
						}
					}
					else
					{
						int vertexCons_res_startIdx = m_nCorrespond_adam2joints * 3;
						dr_dCoeff.block(vertexCons_res_startIdx, 0, 3 * corres_vertex2targetpt.size(), TotalModel::NUM_SHAPE_COEFFICIENTS).setZero();
					}
				}

				if (this->fit_face_ && jacobians[3])		//face
				{
					Map< Matrix<double, Dynamic, Dynamic, RowMajor> > dr_dFaceCoef(jacobians[3], m_nResiduals, TotalModel::NUM_EXP_BASIS_COEFFICIENTS);
					dr_dFaceCoef.setZero();

					if (!this->joint_only)
					{
						for (int i = 0; i < corres_vertex2targetpt.size(); ++i)
						{
							if (corres_vertex2targetpt[i].second[0] == 0.0 && corres_vertex2targetpt[i].second[1] == 0.0 && corres_vertex2targetpt[i].second[2] == 0.0) continue;
							dr_dFaceCoef.row(vertexCons_res_startIdx + 3 * i + 0) = WEAK_WEIGHT * dVdfc.row(3 * corres_vertex2targetpt[i].first + 0);
							dr_dFaceCoef.row(vertexCons_res_startIdx + 3 * i + 1) = WEAK_WEIGHT * dVdfc.row(3 * corres_vertex2targetpt[i].first + 1);
							dr_dFaceCoef.row(vertexCons_res_startIdx + 3 * i + 2) = WEAK_WEIGHT * dVdfc.row(3 * corres_vertex2targetpt[i].first + 2);
						}
					}
				}
			}
		}

		else if(projection_ == 1u)
		// fit projection of 3D joints
		{
			assert(this->pK_);
			Eigen::Map< Matrix<double, Dynamic, 3, RowMajor> > jointArray(tempJoints.data(), m_nCorrespond_adam2joints, 3);
			Eigen::Map< Matrix<double, 3, 3, RowMajor> > K(pK_);
			Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> jointProjection = jointArray * K.transpose();

			for (int i = 0; i < m_nCorrespond_adam2joints; i++)
			{
				if (m_targetPts(5 * i + 3) == 0.0)
					residuals[res_dim * i] = residuals[res_dim * i + 1] = 0.0;
				else
				{
					residuals[2*i+0] = jointProjection(i, 0) / jointProjection(i, 2) - m_targetPts(5 * i + 3);
					residuals[2*i+1] = jointProjection(i, 1) / jointProjection(i, 2) - m_targetPts(5 * i + 4);
				}
			}
			for (int j = 0; j < res_dim * m_nCorrespond_adam2joints; ++j)
			{
				residuals[j] *= m_targetPts_weight[j];
			}

			if (!this->joint_only)
			{
				double x, y, X, Y, Z;
				for (int i = 0; i < corres_vertex2targetpt.size(); i++)
				{
					if (corres_vertex2targetpt[i].second[3] == 0.0)
						res.block(vertexCons_res_startIdx + 2 * i, 0, 2, 1).setZero();
					else
					{
						X = V_vec(3 * corres_vertex2targetpt[i].first + 0);
						Y = V_vec(3 * corres_vertex2targetpt[i].first + 1);
						Z = V_vec(3 * corres_vertex2targetpt[i].first + 2);

						x = (K(0, 0) * X + K(0, 1) * Y) / Z + K(0, 2);
						y = K(1, 1) * Y / Z + K(1, 2);

						res[vertexCons_res_startIdx + res_dim * i + 0] = WEAK_WEIGHT * (x - corres_vertex2targetpt[i].second[3]);
						res[vertexCons_res_startIdx + res_dim * i + 1] = WEAK_WEIGHT * (y - corres_vertex2targetpt[i].second[4]);
					}
					// std::cout << i << " " << x << " " << corres_vertex2targetpt[i].second.x << " " << y << " " << corres_vertex2targetpt[i].second.y << std::endl;
				}
			}
			else res.block(vertexCons_res_startIdx, 0, res_dim * corres_vertex2targetpt.size(), 1).setZero();

			if (jacobians)
			{
				if (jacobians[0])
				{
					Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdt(jacobians[0], m_nResiduals, 3);
					drdt.setZero();
					Matrix<double, Dynamic, Dynamic, RowMajor> dJdt = m_mat_vert2joints * m_dVdt;

					for (int i = 0; i < m_nCorrespond_adam2joints; i++)
					{
						if (m_targetPts(5 * i + 3) == 0.0) continue;
						double XYZ[3] = {jointArray(i, 0), jointArray(i, 1), jointArray(i, 2)};
						projection_jacobian(drdt, dJdt, XYZ, pK_, res_dim * i, 3 * i);
					}
					for (int j = 0; j < res_dim * m_nCorrespond_adam2joints; ++j)
					{
						drdt.row(j) *= m_targetPts_weight[j];
					}

					// set the vertex gradient
					if (!this->joint_only)
					{
						for (int i = 0; i < corres_vertex2targetpt.size(); ++i)
						{
							if (corres_vertex2targetpt[i].second[3] == 0.0) continue;
							double XYZ[3] = {V_vec(3 * corres_vertex2targetpt[i].first + 0), V_vec(3 * corres_vertex2targetpt[i].first + 1), V_vec(3 * corres_vertex2targetpt[i].first + 2)};
							projection_jacobian(drdt, m_dVdt, XYZ, pK_, vertexCons_res_startIdx + res_dim * i, 3 * corres_vertex2targetpt[i].first, WEAK_WEIGHT);
						}
					}
					else
					{
						drdt.block(vertexCons_res_startIdx, 0, res_dim * corres_vertex2targetpt.size(), 3).setZero();
					}
				}

				if (jacobians[1])
				{
					Map< Matrix<double, Dynamic, Dynamic, RowMajor> > dr_dPose(jacobians[1],
						m_nResiduals,
						TotalModel::NUM_JOINTS * 3);	
					dr_dPose.setZero();
					if (this->joint_only)
					{
						for (int i = 0; i < m_adam.m_indices_jointConst_adamIdx.rows(); i++)
						{
							if (m_targetPts(5 * i + 3) == 0.0) continue;
							double XYZ[3] = {jointArray(i, 0), jointArray(i, 1), jointArray(i, 2)};
							projection_jacobian(dr_dPose, dTJdPfr, XYZ, pK_, res_dim * i, 3 * m_adam.m_indices_jointConst_adamIdx(i));
						}
						int offset = m_adam.m_indices_jointConst_adamIdx.rows();
						for (int i = 0; i < m_adam.m_correspond_adam2lHand_adamIdx.rows(); i++)
						{
							if (m_targetPts(5 * (i + offset) + 3) == 0.0) continue;
							double XYZ[3] = {jointArray(i + offset, 0), jointArray(i + offset, 1), jointArray(i + offset, 2)};
							projection_jacobian(dr_dPose, dTJdPfr, XYZ, pK_, res_dim * (i + offset), 3 * m_adam.m_correspond_adam2lHand_adamIdx(i));
						}
						offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();
						for (int i = 0; i < m_adam.m_correspond_adam2rHand_adamIdx.rows(); i++)
						{
							if (m_targetPts(5 * (i + offset) + 3) == 0.0) continue;
							double XYZ[3] = {jointArray(i + offset, 0), jointArray(i + offset, 1), jointArray(i + offset, 2)};
							projection_jacobian(dr_dPose, dTJdPfr, XYZ, pK_, res_dim * (i + offset), 3 * m_adam.m_correspond_adam2rHand_adamIdx(i));
						}
					}
					else
					{
						Matrix<double, Dynamic, Dynamic, RowMajor> dJdP = m_mat_vert2joints * dVdPfr;	
						for (int i = 0; i < m_nCorrespond_adam2joints; i++)
						{
							if (m_targetPts(5 * i + 3) == 0.0) continue;
							double XYZ[3] = {jointArray(i, 0), jointArray(i, 1), jointArray(i, 2)};
							projection_jacobian(dr_dPose, dJdP, XYZ, pK_, res_dim * i, 3 * i);
						}	
					}
					// multiply by weight
					for (int j = 0; j < res_dim * m_nCorrespond_adam2joints; ++j)
					{
						dr_dPose.row(j) *= m_targetPts_weight[j];
					}

					if(!this->joint_only)
					{
						for (int i = 0; i < corres_vertex2targetpt.size(); ++i)
						{
							if (corres_vertex2targetpt[i].second[3] == 0.0) continue;
							double XYZ[3] = {V_vec(3 * corres_vertex2targetpt[i].first + 0), V_vec(3 * corres_vertex2targetpt[i].first + 1), V_vec(3 * corres_vertex2targetpt[i].first + 2)};
							projection_jacobian(dr_dPose, dVdPfr, XYZ, pK_, vertexCons_res_startIdx + res_dim * i, 3 * corres_vertex2targetpt[i].first, WEAK_WEIGHT);
						}
					}
					else
					{
						dr_dPose.block(vertexCons_res_startIdx, 0, res_dim * corres_vertex2targetpt.size(), TotalModel::NUM_POSE_PARAMETERS).setZero();
					}
				}

				if (jacobians[2])
				{
					Map< Matrix<double, Dynamic, Dynamic, RowMajor> > dr_dCoeff(jacobians[2],
						m_nResiduals,
						TotalModel::NUM_SHAPE_COEFFICIENTS);
					dr_dCoeff.setZero();
					if (this->joint_only)
					{
						for (int i = 0; i < m_adam.m_indices_jointConst_adamIdx.rows(); i++)
						{
							if (m_targetPts(5 * i + 3) == 0.0) continue;
							double XYZ[3] = {jointArray(i, 0), jointArray(i, 1), jointArray(i, 2)};
							projection_jacobian(dr_dCoeff, dTJdcfr, XYZ, pK_, res_dim * i, 3 * m_adam.m_indices_jointConst_adamIdx(i));
						}
						int offset = m_adam.m_indices_jointConst_adamIdx.rows();
						for (int i = 0; i < m_adam.m_correspond_adam2lHand_adamIdx.rows(); i++)
						{
							if (m_targetPts(5 * (i + offset) + 3) == 0.0) continue;
							double XYZ[3] = {jointArray(i + offset, 0), jointArray(i + offset, 1), jointArray(i + offset, 2)};
							projection_jacobian(dr_dCoeff, dTJdcfr, XYZ, pK_, res_dim * (i + offset), 3 * m_adam.m_correspond_adam2lHand_adamIdx(i));
						}
						offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();
						for (int i = 0; i < m_adam.m_correspond_adam2rHand_adamIdx.rows(); i++)
						{
							if (m_targetPts(5 * (i + offset) + 3) == 0.0) continue;
							double XYZ[3] = {jointArray(i + offset, 0), jointArray(i + offset, 1), jointArray(i + offset, 2)};
							projection_jacobian(dr_dCoeff, dTJdcfr, XYZ, pK_, res_dim * (i + offset), 3 * m_adam.m_correspond_adam2rHand_adamIdx(i));
						}
					}
					else
					{
						Matrix<double, Dynamic, Dynamic, RowMajor> dJdC = m_mat_vert2joints * dVdcfr;
						for (int i = 0; i < m_nCorrespond_adam2joints; i++)
						{
							if (m_targetPts(5 * i + 3) == 0.0) continue;
							double XYZ[3] = {jointArray(i, 0), jointArray(i, 1), jointArray(i, 2)};
							projection_jacobian(dr_dCoeff, dJdC, XYZ, pK_, res_dim * i, 3 * i);
						}	
					}
					// multiply by weight
					for (int j = 0; j < res_dim * m_nCorrespond_adam2joints; ++j)
					{
						dr_dCoeff.row(j) *= m_targetPts_weight[j];
					}

					if(!this->joint_only)
					{
						for (int i = 0; i < corres_vertex2targetpt.size(); ++i)
						{
							if (corres_vertex2targetpt[i].second[3] == 0.0) continue;
							double XYZ[3] = {V_vec(3 * corres_vertex2targetpt[i].first + 0), V_vec(3 * corres_vertex2targetpt[i].first + 1), V_vec(3 * corres_vertex2targetpt[i].first + 2)};
							projection_jacobian(dr_dCoeff, dVdcfr, XYZ, pK_, vertexCons_res_startIdx + res_dim * i, 3 * corres_vertex2targetpt[i].first, WEAK_WEIGHT);
						}
					}
					else
					{
						dr_dCoeff.block(vertexCons_res_startIdx, 0, res_dim * corres_vertex2targetpt.size(), TotalModel::NUM_SHAPE_COEFFICIENTS).setZero();
					}
				}

				if (this->fit_face_ && jacobians[3])
				{
					Map< Matrix<double, Dynamic, Dynamic, RowMajor> > dr_dFaceCoef(jacobians[3], m_nResiduals, TotalModel::NUM_EXP_BASIS_COEFFICIENTS);
					dr_dFaceCoef.setZero();

					// only related to vertex weights
					if(!this->joint_only)
					{
						for (int i = 0; i < corres_vertex2targetpt.size(); ++i)
						{
							if (corres_vertex2targetpt[i].second[3] == 0.0) continue;
							double XYZ[3] = {V_vec(3 * corres_vertex2targetpt[i].first + 0), V_vec(3 * corres_vertex2targetpt[i].first + 1), V_vec(3 * corres_vertex2targetpt[i].first + 2)};
							projection_jacobian(dr_dFaceCoef, dVdfc, XYZ, pK_, vertexCons_res_startIdx + res_dim * i, 3 * corres_vertex2targetpt[i].first, WEAK_WEIGHT);
						}
					}
				}
			}
		}

		else
		{
			assert(projection_ == 2u);
			assert(pK_);

			int root_index = 12; // last target point, neck
			Eigen::Map< Matrix<double, Dynamic, 3, RowMajor> > jointArray(tempJoints.data(), m_nCorrespond_adam2joints, 3);
			Eigen::Map< Matrix<double, 3, 3, RowMajor> > K(pK_);
			Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> jointProjection = jointArray * K.transpose();

			for(int i = 0; i < m_nCorrespond_adam2joints; i++)
			{
				res.block(5 * i, 0, 3, 1) = tempJoints.block(3 * i, 0, 3, 1) - tempJoints.block(3 * root_index, 0, 3, 1);
				res.block(5 * i, 0, 3, 1) += (- m_targetPts.block(5 * i, 0, 3, 1) + m_targetPts.block(5 * root_index, 0, 3, 1));
				if (m_targetPts(5 * i + 3) != 0.0)
				{
					residuals[5*i+3] = weight2d * (jointProjection(i, 0) / jointProjection(i, 2) - m_targetPts(5 * i + 3));
					residuals[5*i+4] = weight2d * (jointProjection(i, 1) / jointProjection(i, 2) - m_targetPts(5 * i + 4));
				}
			}

			for (int j = 0; j < res_dim * m_nCorrespond_adam2joints; ++j)
			{
				residuals[j] *= m_targetPts_weight[j];
			}

			// Vertex Constraints //////////////////////
			if (!this->joint_only)
			{
				double X, Y, Z, x, y;
				for (int i = 0; i < corres_vertex2targetpt.size(); ++i)
				{
					res[vertexCons_res_startIdx + res_dim * i + 0] = WEAK_WEIGHT * (V_vec(3 * corres_vertex2targetpt[i].first + 0) - tempJoints(3 * root_index + 0) - corres_vertex2targetpt[i].second[0] + m_targetPts(5 * root_index)); //- m_targetPts(res_offset + 3 * i + 0));
					res[vertexCons_res_startIdx + res_dim * i + 1] = WEAK_WEIGHT * (V_vec(3 * corres_vertex2targetpt[i].first + 1) - tempJoints(3 * root_index + 1) - corres_vertex2targetpt[i].second[1] + m_targetPts(5 * root_index + 1)); //- m_targetPts(res_offset + 3 * i + 1));
					res[vertexCons_res_startIdx + res_dim * i + 2] = WEAK_WEIGHT * (V_vec(3 * corres_vertex2targetpt[i].first + 2) - tempJoints(3 * root_index + 2) - corres_vertex2targetpt[i].second[2] + m_targetPts(5 * root_index + 2)); //- m_targetPts(res_offset + 3 * i + 2));
					if (corres_vertex2targetpt[i].second[3] != 0.0)
					{
						X = V_vec(3 * corres_vertex2targetpt[i].first + 0);
						Y = V_vec(3 * corres_vertex2targetpt[i].first + 1);
						Z = V_vec(3 * corres_vertex2targetpt[i].first + 2);

						x = (K(0, 0) * X + K(0, 1) * Y) / Z + K(0, 2);
						y = K(1, 1) * Y / Z + K(1, 2);

						res[vertexCons_res_startIdx + res_dim * i + 3] = WEAK_WEIGHT * weight2d * (x - corres_vertex2targetpt[i].second[3]);
						res[vertexCons_res_startIdx + res_dim * i + 4] = WEAK_WEIGHT * weight2d * (y - corres_vertex2targetpt[i].second[4]);
					}
				}
			}
			else res.block(vertexCons_res_startIdx, 0, res_dim * corres_vertex2targetpt.size(), 1).setZero();

			// compute jacobian
			if (jacobians)
			{
				if (jacobians[0])
				{
					Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdt(jacobians[0], m_nResiduals, 3);
					drdt.setZero();
					// no jacobian from 3D residual
					Matrix<double, Dynamic, Dynamic, RowMajor> dJdt = m_mat_vert2joints * m_dVdt;

					for (int i = 0; i < m_nCorrespond_adam2joints; i++)
					{
						if (m_targetPts(5 * i + 3) == 0.0) continue;
						double XYZ[3] = {jointArray(i, 0), jointArray(i, 1), jointArray(i, 2)};
						projection_jacobian(drdt, dJdt, XYZ, pK_, res_dim * i + 3, 3 * i, weight2d);
					}

					for (int j = 0; j < res_dim * m_nCorrespond_adam2joints; ++j)
					{
						drdt.row(j) *= m_targetPts_weight[j];
					}

					// No jacobian from 3D vertex
					if (!this->joint_only)
					{
						for (int i = 0; i < corres_vertex2targetpt.size(); ++i)
						{
							if (corres_vertex2targetpt[i].second[3] == 0.0) continue;
							double XYZ[3] = {V_vec(3 * corres_vertex2targetpt[i].first + 0), V_vec(3 * corres_vertex2targetpt[i].first + 1), V_vec(3 * corres_vertex2targetpt[i].first + 2)};
							projection_jacobian(drdt, m_dVdt, XYZ, pK_, vertexCons_res_startIdx + res_dim * i + 3, 3 * corres_vertex2targetpt[i].first, weight2d * WEAK_WEIGHT);
						}
					}
				}
				if (jacobians[1])
				{
					Map< Matrix<double, Dynamic, Dynamic, RowMajor> > dr_dPose(jacobians[1],
						m_nResiduals,
						TotalModel::NUM_JOINTS * 3);
					dr_dPose.setZero();
					Matrix<double, Dynamic, Dynamic, RowMajor> dJdP = m_mat_vert2joints * dVdPfr;  // used to compute jacobian when joint_only is false
					if (this->joint_only)
					{
						for (int i = 0; i < m_adam.m_indices_jointConst_adamIdx.rows(); i++)
						{
						// 	tempJoints(0, 3*i, 3, 1) = outT(3*m_adam.m_indices_jointConst_adamIdx(i), 3, 3, 1);
							dr_dPose.block(res_dim * i, 0, 3, TotalModel::NUM_POSE_PARAMETERS) = dTJdPfr.block(3 * m_adam.m_indices_jointConst_adamIdx(i), 0, 3, TotalModel::NUM_POSE_PARAMETERS) - 
								dTJdPfr.block(3 * m_adam.m_indices_jointConst_adamIdx(root_index), 0, 3, TotalModel::NUM_POSE_PARAMETERS);
							if (m_targetPts(5 * i + 3) == 0.0) continue;
							double XYZ[3] = {jointArray(i, 0), jointArray(i, 1), jointArray(i, 2)};
							projection_jacobian(dr_dPose, dTJdPfr, XYZ, pK_, res_dim * i + 3, 3 * m_adam.m_indices_jointConst_adamIdx(i), weight2d);
						}
						int offset = m_adam.m_indices_jointConst_adamIdx.rows();
						for (int i = 0; i < m_adam.m_correspond_adam2lHand_adamIdx.rows(); i++)
						{
							dr_dPose.block(res_dim * (i + offset), 0, 3, TotalModel::NUM_POSE_PARAMETERS) = dTJdPfr.block(3 * m_adam.m_correspond_adam2lHand_adamIdx(i), 0, 3, TotalModel::NUM_POSE_PARAMETERS)
								- dTJdPfr.block(3 * m_adam.m_indices_jointConst_adamIdx(root_index), 0, 3, TotalModel::NUM_POSE_PARAMETERS);
							if (m_targetPts(5 * (i + offset) + 3) == 0.0) continue;
							double XYZ[3] = {jointArray(i + offset, 0), jointArray(i + offset, 1), jointArray(i + offset, 2)};
							projection_jacobian(dr_dPose, dTJdPfr, XYZ, pK_, res_dim * (i + offset) + 3, 3 * m_adam.m_correspond_adam2lHand_adamIdx(i), weight2d);
						}
						offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();
						for (int i = 0; i < m_adam.m_correspond_adam2rHand_adamIdx.rows(); i++)
						{
							dr_dPose.block(res_dim * (i + offset), 0, 3, TotalModel::NUM_POSE_PARAMETERS) = dTJdPfr.block(3 * m_adam.m_correspond_adam2rHand_adamIdx(i), 0, 3, TotalModel::NUM_POSE_PARAMETERS)
								- dTJdPfr.block(3 * m_adam.m_indices_jointConst_adamIdx(root_index), 0, 3, TotalModel::NUM_POSE_PARAMETERS);
							if (m_targetPts(5 * (i + offset) + 3) == 0.0) continue;
							double XYZ[3] = {jointArray(i + offset, 0), jointArray(i + offset, 1), jointArray(i + offset, 2)};
							projection_jacobian(dr_dPose, dTJdPfr, XYZ, pK_, res_dim * (i + offset) + 3, 3 * m_adam.m_correspond_adam2rHand_adamIdx(i), weight2d);
						}
					}
					else
					{
						for (int i = 0; i < m_nCorrespond_adam2joints; i++)
						{
							dr_dPose.block(res_dim * i, 0, 3, TotalModel::NUM_JOINTS * 3) = dJdP.block(3 * i, 0, 3, TotalModel::NUM_JOINTS * 3)
								- dJdP.block(3 * root_index, 0, 3, TotalModel::NUM_JOINTS * 3);
							if (m_targetPts(5 * i + 3) == 0.0) continue;
							double XYZ[3] = {jointArray(i, 0), jointArray(i, 1), jointArray(i, 2)};
							projection_jacobian(dr_dPose, dJdP, XYZ, pK_, res_dim * i + 3, 3 * i, weight2d);
						}
					}

					for (int j = 0; j < res_dim * m_nCorrespond_adam2joints; ++j)
					{
						dr_dPose.row(j) *= m_targetPts_weight[j];
					}

					if(!this->joint_only)
					{
						for (int i = 0; i < corres_vertex2targetpt.size(); ++i)
						{
							dr_dPose.row(vertexCons_res_startIdx + res_dim * i + 0) = WEAK_WEIGHT * (dVdPfr.row(3 * corres_vertex2targetpt[i].first + 0) - dJdP.row(3 * root_index + 0));
							dr_dPose.row(vertexCons_res_startIdx + res_dim * i + 1) = WEAK_WEIGHT * (dVdPfr.row(3 * corres_vertex2targetpt[i].first + 1) - dJdP.row(3 * root_index + 1));
							dr_dPose.row(vertexCons_res_startIdx + res_dim * i + 2) = WEAK_WEIGHT * (dVdPfr.row(3 * corres_vertex2targetpt[i].first + 2) - dJdP.row(3 * root_index + 2));
							if (corres_vertex2targetpt[i].second[3] == 0.0) continue;
							double XYZ[3] = {V_vec(3 * corres_vertex2targetpt[i].first + 0), V_vec(3 * corres_vertex2targetpt[i].first + 1), V_vec(3 * corres_vertex2targetpt[i].first + 2)};
							projection_jacobian(dr_dPose, dVdPfr, XYZ, pK_, vertexCons_res_startIdx + res_dim * i + 3, 3 * corres_vertex2targetpt[i].first, weight2d * WEAK_WEIGHT);
						}
					}
					else
					{
						dr_dPose.block(vertexCons_res_startIdx, 0, res_dim * corres_vertex2targetpt.size(), TotalModel::NUM_POSE_PARAMETERS).setZero();
					}
				}
				if (jacobians[2])
				{
					Map< Matrix<double, Dynamic, Dynamic, RowMajor> > dr_dCoeff(jacobians[2],
						m_nResiduals,
						TotalModel::NUM_SHAPE_COEFFICIENTS);
					dr_dCoeff.setZero();
					Matrix<double, Dynamic, Dynamic, RowMajor> dJdC = m_mat_vert2joints*dVdcfr;
					if (this->joint_only)
					{
						for (int i = 0; i < m_adam.m_indices_jointConst_adamIdx.rows(); i++)
						{
						// 	tempJoints(0, 3*i, 3, 1) = outT(3*m_adam.m_indices_jointConst_adamIdx(i), 3, 3, 1);
							dr_dCoeff.block(res_dim * i, 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS) = dTJdcfr.block(3 * m_adam.m_indices_jointConst_adamIdx(i), 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS)
								- dTJdcfr.block(3 * m_adam.m_indices_jointConst_adamIdx(root_index), 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS);
							if (m_targetPts(5 * i + 3) == 0.0) continue;
							double XYZ[3] = {jointArray(i, 0), jointArray(i, 1), jointArray(i, 2)};
							projection_jacobian(dr_dCoeff, dTJdcfr, XYZ, pK_, res_dim * i + 3, 3 * m_adam.m_indices_jointConst_adamIdx(i), weight2d);
						}
						int offset = m_adam.m_indices_jointConst_adamIdx.rows();
						for (int i = 0; i < m_adam.m_correspond_adam2lHand_adamIdx.rows(); i++)
						{
							dr_dCoeff.block(res_dim*(i + offset), 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS) = dTJdcfr.block(3 * m_adam.m_correspond_adam2lHand_adamIdx(i), 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS)
								- dTJdcfr.block(3 * m_adam.m_indices_jointConst_adamIdx(root_index), 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS);
							if (m_targetPts(5 * (i + offset) + 3) == 0.0) continue;
							double XYZ[3] = {jointArray(i + offset, 0), jointArray(i + offset, 1), jointArray(i + offset, 2)};
							projection_jacobian(dr_dCoeff, dTJdcfr, XYZ, pK_, res_dim * (i + offset) + 3, 3 * m_adam.m_correspond_adam2lHand_adamIdx(i), weight2d);
						}
						offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();
						for (int i = 0; i < m_adam.m_correspond_adam2rHand_adamIdx.rows(); i++)
						{
							dr_dCoeff.block(res_dim * (i + offset), 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS) = dTJdcfr.block(3 * m_adam.m_correspond_adam2rHand_adamIdx(i), 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS)
								- dTJdcfr.block(3 * m_adam.m_indices_jointConst_adamIdx(root_index), 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS);
							if (m_targetPts(5 * (i + offset) + 3) == 0.0) continue;
							double XYZ[3] = {jointArray(i + offset, 0), jointArray(i + offset, 1), jointArray(i + offset, 2)};
							projection_jacobian(dr_dCoeff, dTJdcfr, XYZ, pK_, res_dim * (i + offset) + 3, 3 * m_adam.m_correspond_adam2rHand_adamIdx(i), weight2d);
						}
					}
					else
					{
						for (int i = 0; i < m_nCorrespond_adam2joints; i++)
						{
							dr_dCoeff.block(res_dim * i, 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS) = dJdC.block(3 * i, 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS)
								- dJdC.block(3 * root_index, 0, 3, TotalModel::NUM_SHAPE_COEFFICIENTS);
							if (m_targetPts(5 * i + 3) == 0.0) continue;
							double XYZ[3] = {jointArray(i, 0), jointArray(i, 1), jointArray(i, 2)};
							projection_jacobian(dr_dCoeff, dJdC, XYZ, pK_, res_dim * i + 3, 3 * i, weight2d);
						}
					}

					for (int j = 0; j < res_dim * m_nCorrespond_adam2joints; ++j)
					{
						dr_dCoeff.row(j) *= m_targetPts_weight[j];
					}

					if (!this->joint_only)
					{
						for (int i = 0; i < corres_vertex2targetpt.size(); ++i)
						{
							// face coeffs do not influence body vertex at all
							dr_dCoeff.row(vertexCons_res_startIdx + res_dim * i + 0) = WEAK_WEIGHT * (dVdcfr.row(3 * corres_vertex2targetpt[i].first + 0) - dJdC.row(3 * root_index + 0));
							dr_dCoeff.row(vertexCons_res_startIdx + res_dim * i + 1) = WEAK_WEIGHT * (dVdcfr.row(3 * corres_vertex2targetpt[i].first + 1) - dJdC.row(3 * root_index + 1));
							dr_dCoeff.row(vertexCons_res_startIdx + res_dim * i + 2) = WEAK_WEIGHT * (dVdcfr.row(3 * corres_vertex2targetpt[i].first + 2) - dJdC.row(3 * root_index + 2));
							if (corres_vertex2targetpt[i].second[3] == 0.0) continue;
							double XYZ[3] = {V_vec(3 * corres_vertex2targetpt[i].first + 0), V_vec(3 * corres_vertex2targetpt[i].first + 1), V_vec(3 * corres_vertex2targetpt[i].first + 2)};
							projection_jacobian(dr_dCoeff, dVdcfr, XYZ, pK_, vertexCons_res_startIdx + res_dim * i, 3 * corres_vertex2targetpt[i].first, weight2d * WEAK_WEIGHT);
						}
					}
					else
					{
						dr_dCoeff.block(vertexCons_res_startIdx, 0, res_dim * corres_vertex2targetpt.size(), TotalModel::NUM_SHAPE_COEFFICIENTS).setZero();
					}
				}

				if (this->fit_face_ && jacobians[3])		//face
				{
					Map< Matrix<double, Dynamic, Dynamic, RowMajor> > dr_dFaceCoef(jacobians[3], m_nResiduals, TotalModel::NUM_EXP_BASIS_COEFFICIENTS);
					dr_dFaceCoef.setZero();

					if (!this->joint_only)
					{
						for (int i = 0; i < corres_vertex2targetpt.size(); ++i)
						{
							// face coeff has no influence on the root joint location
							dr_dFaceCoef.row(vertexCons_res_startIdx + res_dim * i + 0) = WEAK_WEIGHT * dVdfc.row(3 * corres_vertex2targetpt[i].first + 0);
							dr_dFaceCoef.row(vertexCons_res_startIdx + res_dim * i + 1) = WEAK_WEIGHT * dVdfc.row(3 * corres_vertex2targetpt[i].first + 1);
							dr_dFaceCoef.row(vertexCons_res_startIdx + res_dim * i + 2) = WEAK_WEIGHT * dVdfc.row(3 * corres_vertex2targetpt[i].first + 2);
							if (corres_vertex2targetpt[i].second[3] == 0.0) continue;
							double XYZ[3] = {V_vec(3 * corres_vertex2targetpt[i].first + 0), V_vec(3 * corres_vertex2targetpt[i].first + 1), V_vec(3 * corres_vertex2targetpt[i].first + 2)};
							projection_jacobian(dr_dFaceCoef, dVdfc, XYZ, pK_, vertexCons_res_startIdx + res_dim * i + 3, 3 * corres_vertex2targetpt[i].first, weight2d * WEAK_WEIGHT);
						}
					}
				}
			}
		}
		return true;
	}

	TotalModel &m_adam;
	// FaceModel &m_facem;

	double WEAK_WEIGHT;

	Eigen::MatrixXd m_rfoot_joints;
	Eigen::MatrixXd m_lfoot_joints;

	// viewer::Viewer &kviewer_;
	Eigen::SparseMatrix<double, Eigen::ColMajor> m_mat_vert2joints;
	Eigen::SparseMatrix<double, Eigen::ColMajor> m_Ajoints;

	Eigen::SparseMatrix<double, Eigen::ColMajor> m_dVdt;
	Eigen::SparseMatrix<double, Eigen::ColMajor> dJdV_;

	Eigen::VectorXd m_targetPts;		//renamed from sbV_j_
	std::vector<double> m_targetPts_weight;
	std::vector<double> m_targetPts_weight_buffer;

	//Speical Treatment for Spine
	Eigen::Vector3d m_spineDirect;		//from body center to neck
	Eigen::Vector3d m_target_bodyCenter;		//
	Eigen::Vector3d m_target_neck;		//

	Eigen::SparseMatrix<double, Eigen::ColMajor> W_corr_;

	int m_nCorrespond_adam2joints;		//Contraints on SMPL's joint
	int m_nCorrespond_adam2pts;	//Constratins on SMPL's vertex
	int m_nResiduals;

	int m_spineResIdx;
	// std::vector< std::pair<int, cv::Point3d> > corres_vertex2targetpt;		//smpl vertex index vs posemachine joint
	std::vector< std::pair<int, std::vector<double>> > corres_vertex2targetpt;		//smpl vertex index vs posemachine joint

	Eigen::MatrixXd m_bodyJoints;
	Eigen::Matrix<double, 5, Eigen::Dynamic, Eigen::RowMajor> m_FaceJoints;
	Eigen::Matrix<double, 5, Eigen::Dynamic, Eigen::RowMajor> m_lHandJoints;
	Eigen::Matrix<double, 5, Eigen::Dynamic, Eigen::RowMajor> m_rHandJoints;

	bool joint_only;
	uint projection_; // 0: fit 3D only, 1: fit 2D only, 2: fit 2D with relative 3D.
	double* pK_; // a pointer to the K matrix
	bool fit_face_;
	float weight2d;
	int res_dim; // set to 3 for 3D fitting, 2 for 2D fitting, 5 for fit 2D with relative 3D
};
