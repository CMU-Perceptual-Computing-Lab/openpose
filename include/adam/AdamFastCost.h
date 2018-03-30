#include <iostream>
#include "totalmodel.h"
#include <cassert>
#include "pose_to_transforms.h"
#include <cstring>

class AdamFastCost: public ceres::CostFunction
{
public:
	AdamFastCost(TotalModel &adam,
		Eigen::MatrixXd &Joints,
		Eigen::MatrixXd &rFoot,		//3x2 //Heel, Toe
		Eigen::MatrixXd &lFoot,
		Eigen::MatrixXd &faceJoints,
		Eigen::MatrixXd &lHandJoints,
		Eigen::MatrixXd &rHandJoints,
		double* p_adam_coeff
	): m_adam(adam), m_bodyJoints(Joints), m_rfoot_joints(rFoot), m_lfoot_joints(lFoot), m_FaceJoints(faceJoints),
		m_lHandJoints(lHandJoints), m_rHandJoints(rHandJoints), res_dim(3)
	{
		assert(p_adam_coeff != NULL);
		std::copy(p_adam_coeff, p_adam_coeff + TotalModel::NUM_SHAPE_COEFFICIENTS, m_pCoeff);
		SetupCost();
	}
	virtual ~AdamFastCost() {}

	void UpdateJoints(Eigen::MatrixXd &Joints, Eigen::MatrixXd &rFoot, Eigen::MatrixXd &lFoot, Eigen::MatrixXd &faceJoints,
		Eigen::MatrixXd &lHandJoints, Eigen::MatrixXd &rHandJoints)
	{
		m_bodyJoints << Joints;
		m_rfoot_joints << rFoot;
		m_lfoot_joints << lFoot;
		m_FaceJoints << faceJoints;
		m_lHandJoints << lHandJoints;
		m_rHandJoints << rHandJoints;
		UpdateTarget();
	}

	void SetupCost()
	{
		using namespace cv;
		using namespace Eigen;

		double BODYJOINT_WEIGHT_Strong = 1;
		double BODYJOINT_WEIGHT = 1;
		double HANDJOINT_WEIGHT = 1;
		WEAK_WEIGHT = 1;

		m_nCorrespond_adam2joints = m_adam.m_indices_jointConst_adamIdx.rows();
		if(m_lHandJoints.size() > 0) m_nCorrespond_adam2joints += m_adam.m_correspond_adam2lHand_adamIdx.rows();
		if(m_rHandJoints.size() > 0) m_nCorrespond_adam2joints += m_adam.m_correspond_adam2rHand_adamIdx.rows();

		m_targetPts.resize(m_nCorrespond_adam2joints * 5);
		m_targetPts.setZero();
		m_targetPts_weight.resize(m_nCorrespond_adam2joints * res_dim);
		m_targetPts_weight_buffer.resize(m_nCorrespond_adam2joints * res_dim);

		int offset = 0;
		for (int ic = 0; ic<m_adam.m_indices_jointConst_adamIdx.rows(); ic++)
		{
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

		if (m_lHandJoints.size() > 0)
		{
			//correspondences.reserve(nCorrespond_);
			for (int ic = 0; ic < m_adam.m_correspond_adam2lHand_adamIdx.rows(); ic++)
			{
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
				for (int i = 0; i < res_dim; i++)
				{
					m_targetPts_weight[(ic + offset) * res_dim + i] = HANDJOINT_WEIGHT;
					m_targetPts_weight_buffer[(ic + offset) * res_dim + i] = HANDJOINT_WEIGHT;
				}
			}
		}

		if (m_bodyJoints(0, 1) != 0.0)		//nose
			corres_vertex2targetpt.push_back(std::make_pair(8130, std::vector<double>(5)));		
		if (m_bodyJoints(0, 16) != 0.0)		//left ear
			corres_vertex2targetpt.push_back(std::make_pair(6970, std::vector<double>(5)));		
		if (m_bodyJoints(0, 18) != 0.0)		//right ear
			corres_vertex2targetpt.push_back(std::make_pair(10088, std::vector<double>(5)));		
		m_nCorrespond_adam2pts = corres_vertex2targetpt.size();

		UpdateTarget();  // copy target

		// m_nResiduals = m_nCorrespond_adam2joints * res_dim + m_nCorrespond_adam2pts * res_dim;
		m_nResiduals = m_nCorrespond_adam2joints * res_dim;
		std::cout << "m_nResiduals " << m_nResiduals << std::endl;
		CostFunction::set_num_residuals(m_nResiduals);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(3); // Translation
		parameter_block_sizes->push_back(TotalModel::NUM_JOINTS * 3); // SMPL Pose  

		// perform regression from shape coeff to Joint
		Map< const Matrix<double, Dynamic, 1> > c_bodyshape(m_pCoeff, TotalModel::NUM_SHAPE_COEFFICIENTS);
		m_Vt.resize(TotalModel::NUM_VERTICES, 3);
		Map< Matrix<double, Dynamic, 1> > Vt_vec(m_Vt.data(), 3 * TotalModel::NUM_VERTICES);
		Vt_vec = m_adam.m_meanshape + m_adam.m_shapespace_u*c_bodyshape;

		m_J0.resize(TotalModel::NUM_JOINTS, 3);
		Map< Matrix<double, Dynamic, 1> > J_vec(m_J0.data(), TotalModel::NUM_JOINTS * 3);
		J_vec = m_adam.J_mu_ + m_adam.dJdc_ * c_bodyshape;
	}

	void UpdateTarget()
	{
		int offset = 0;
		for (int ic = 0; ic < m_adam.m_indices_jointConst_adamIdx.rows(); ic++)
			m_targetPts.block(ic * 5, 0, 5, 1) = m_bodyJoints.block(0, m_adam.m_indices_jointConst_smcIdx(ic), 5, 1);
		offset += m_adam.m_indices_jointConst_adamIdx.rows();

		if (m_lHandJoints.size() > 0)
		{
			for (int ic = 0; ic < m_adam.m_correspond_adam2lHand_adamIdx.rows(); ic++)
				m_targetPts.block((offset + ic) * 5, 0, 5, 1) = m_lHandJoints.block(0, m_adam.m_correspond_adam2lHand_lHandIdx(ic), 5, 1);
		}
		offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();

		if (m_rHandJoints.size() > 0)
		{
			for (int ic = 0; ic < m_adam.m_correspond_adam2rHand_adamIdx.rows(); ic++)
				m_targetPts.block((offset + ic) * 5, 0, 5, 1) = m_rHandJoints.block(0, m_adam.m_correspond_adam2rHand_rHandIdx(ic), 5, 1);
		}

		offset = 0;
		if (m_bodyJoints(0, 1) != 0.0)
		{
			std::vector<double> targetPt = {m_bodyJoints(0, 1), m_bodyJoints(1, 1), m_bodyJoints(2, 1), m_bodyJoints(3, 1), m_bodyJoints(4, 1)};
			corres_vertex2targetpt[offset].second = targetPt;
			offset++;
		}
		if (m_bodyJoints(0, 16) != 0.0)		//left ear
		{
			std::vector<double> targetPt = {m_bodyJoints(0, 16), m_bodyJoints(1, 16), m_bodyJoints(2, 16), m_bodyJoints(3, 16), m_bodyJoints(4, 16)};
			corres_vertex2targetpt[offset].second = targetPt;
			offset++;
		}

		if (m_bodyJoints(0, 18) != 0.0)		//right ear
		{
			std::vector<double> targetPt = {m_bodyJoints(0, 18), m_bodyJoints(1, 18), m_bodyJoints(2, 18), m_bodyJoints(3, 18), m_bodyJoints(4, 18)};
			corres_vertex2targetpt[offset].second = targetPt;
			offset++;
		}
	}

	void toggle_activate(bool limb, bool finger)
	{
		for (int ic = 0; ic<m_adam.m_indices_jointConst_adamIdx.rows(); ic++)
		{
			int smcjoint = m_adam.m_indices_jointConst_smcIdx(ic);
			if (smcjoint != 3 && smcjoint != 6 && smcjoint != 9 && smcjoint != 12)
			{
				m_targetPts_weight.block(ic * res_dim, 0, res_dim, 1) = double(limb) * m_targetPts_weight_buffer.block(ic * res_dim, 0, res_dim, 1);
			}
		}

		int offset = m_adam.m_indices_jointConst_smcIdx.rows();
		for (int ic = 0; ic < m_adam.m_correspond_adam2lHand_adamIdx.rows(); ic++)
			m_targetPts_weight.block((ic + offset) * res_dim, 0, res_dim, 1) = m_targetPts_weight_buffer.block((ic + offset) * res_dim, 0, res_dim, 1) * double(finger);
		offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();
		for (int ic = 0; ic < m_adam.m_correspond_adam2rHand_adamIdx.rows(); ic++)
			m_targetPts_weight.block((ic + offset) * res_dim, 0, res_dim, 1) = m_targetPts_weight_buffer.block((ic + offset) * res_dim, 0, res_dim, 1) * double(finger);
	}

	virtual bool Evaluate(double const* const* parameters,
		double* residuals,
		double** jacobians) const
	{
		using namespace Eigen;
		typedef double T;
		const T* t = parameters[0];
		const T* p_eulers = parameters[1];
		Map< const Vector3d > t_vec(t);

		// 1st step: forward kinematics
		const int num_t = (TotalModel::NUM_JOINTS) * 3 * 5;  // transform 3 * 4 + joint location 3 * 1
		Matrix<double, Dynamic, 3 * TotalModel::NUM_JOINTS, RowMajor> dTrdP(num_t, 3 * TotalModel::NUM_JOINTS);
		Matrix<double, Dynamic, 3 * TotalModel::NUM_JOINTS, RowMajor> dTrdJ(num_t, 3 * TotalModel::NUM_JOINTS);
		VectorXd transforms_joint(3 * TotalModel::NUM_JOINTS * 4 + 3 * TotalModel::NUM_JOINTS); // the first part is transform, the second part is outJoint

		ceres::AutoDiffCostFunction<smpl::PoseToTransformsNoLR_Eulers_adamModel,
		(TotalModel::NUM_JOINTS) * 3 * 4 + 3 * TotalModel::NUM_JOINTS,
		(TotalModel::NUM_JOINTS) * 3,
		(TotalModel::NUM_JOINTS) * 3> p2t(new smpl::PoseToTransformsNoLR_Eulers_adamModel(m_adam));

		const double* p2t_parameters[2] = { p_eulers, m_J0.data() };
		double* p2t_residuals = transforms_joint.data();
		double* p2t_jacobians[2] = { dTrdP.data(), dTrdJ.data() };
		p2t.Evaluate(p2t_parameters, p2t_residuals, p2t_jacobians);

		// Jacobian: d(Joint) / d(pose)
		Matrix<double, Dynamic, Dynamic, RowMajor> dTJdP = dTrdP.block(3 * TotalModel::NUM_JOINTS * 4, 0, 3 * TotalModel::NUM_JOINTS, 3 * TotalModel::NUM_JOINTS);
		VectorXd outJoint = transforms_joint.block(3 * TotalModel::NUM_JOINTS * 4, 0, 3 * TotalModel::NUM_JOINTS, 1);  // outJoint

		// 2nd step: set residuals
		Map< VectorXd > res(residuals, m_nResiduals);
		res.setZero();

		// Joint Constraints
		VectorXd tempJoints(3 * m_nCorrespond_adam2joints);  // predicted joint given current parameter
		for (int i = 0; i < m_adam.m_indices_jointConst_adamIdx.rows(); i++)
		{
			tempJoints.block(3 * i, 0, 3, 1) = outJoint.block(3 * m_adam.m_indices_jointConst_adamIdx(i), 0, 3, 1) + t_vec;
		}
		int offset = m_adam.m_indices_jointConst_adamIdx.rows();
		for (int i = 0; i < m_adam.m_correspond_adam2lHand_adamIdx.rows(); i++)
		{
			tempJoints.block(3*(i + offset), 0, 3, 1) = outJoint.block(3 * m_adam.m_correspond_adam2lHand_adamIdx(i), 0, 3, 1) + t_vec;
		}
		offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();
		for (int i = 0; i < m_adam.m_correspond_adam2rHand_adamIdx.rows(); i++)
		{
			tempJoints.block(3*(i + offset), 0, 3, 1) = outJoint.block(3 * m_adam.m_correspond_adam2rHand_adamIdx(i), 0, 3, 1) + t_vec;
		}

		for(int i = 0; i < m_nCorrespond_adam2joints; i++)
		{
			if (m_targetPts.block(5 * i, 0, 3, 1).isZero(0)) continue;
			res.block(3 * i, 0, 3, 1) = (tempJoints.block(3 * i, 0, 3, 1) - m_targetPts.block(5 * i, 0, 3, 1)).cwiseProduct(m_targetPts_weight.block(3 * i, 0, 3, 1));
		}
		// 3rd step: set residuals
		if (jacobians)
		{
			if (jacobians[0])  // jacobian w.r.t translation
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdt(jacobians[0], m_nResiduals, 3);
				drdt.setZero();
				for (int i = 0; i < m_nCorrespond_adam2joints; i++)
				{
					if (m_targetPts.block(5 * i, 0, 3, 1).isZero(0)) continue;
					drdt.block(3 * i, 0, 3, 3).setIdentity();
				}
				for (int j = 0; j < 3 * m_nCorrespond_adam2joints; ++j)
				{
					drdt.row(j) *= m_targetPts_weight[j];
				}
			}	

			if (jacobians[1]) // jacobian w.r.t pose
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > dr_dPose(jacobians[1], m_nResiduals, TotalModel::NUM_JOINTS * 3);
				dr_dPose.setZero();
				for (int i = 0; i < m_adam.m_indices_jointConst_adamIdx.rows(); i++)
				{
					if (m_targetPts.block(5 * i, 0, 3, 1).isZero(0)) continue;
					dr_dPose.block(3 * i, 0, 3, TotalModel::NUM_POSE_PARAMETERS) = dTJdP.block(3 * m_adam.m_indices_jointConst_adamIdx(i), 0, 3, TotalModel::NUM_POSE_PARAMETERS);
				}
				int offset = m_adam.m_indices_jointConst_adamIdx.rows();
				for (int i = 0; i < m_adam.m_correspond_adam2lHand_adamIdx.rows(); i++)
				{
					if (m_targetPts.block(5 * (i + offset), 0, 3, 1).isZero(0)) continue;
					dr_dPose.block(3*(i + offset), 0, 3, TotalModel::NUM_POSE_PARAMETERS) = dTJdP.block(3 * m_adam.m_correspond_adam2lHand_adamIdx(i), 0, 3, TotalModel::NUM_POSE_PARAMETERS);
				}
				offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();
				for (int i = 0; i < m_adam.m_correspond_adam2rHand_adamIdx.rows(); i++)
				{
					if (m_targetPts.block(5 * (i + offset), 0, 3, 1).isZero(0)) continue;
					dr_dPose.block(3*(i + offset), 0, 3, TotalModel::NUM_POSE_PARAMETERS) = dTJdP.block(3 * m_adam.m_correspond_adam2rHand_adamIdx(i), 0, 3, TotalModel::NUM_POSE_PARAMETERS);
				}
				for (int j = 0; j < 3 * m_nCorrespond_adam2joints; ++j)
				{
					dr_dPose.row(j) *= m_targetPts_weight[j];
				}
			}
		}

		return true;
	}

	TotalModel m_adam;
	Eigen::Matrix<double, 5, Eigen::Dynamic, Eigen::RowMajor> m_rfoot_joints;
	Eigen::Matrix<double, 5, Eigen::Dynamic, Eigen::RowMajor> m_lfoot_joints;
	Eigen::Matrix<double, 5, Eigen::Dynamic, Eigen::RowMajor> m_bodyJoints;
	Eigen::Matrix<double, 5, Eigen::Dynamic, Eigen::RowMajor> m_FaceJoints;
	Eigen::Matrix<double, 5, Eigen::Dynamic, Eigen::RowMajor> m_lHandJoints;
	Eigen::Matrix<double, 5, Eigen::Dynamic, Eigen::RowMajor> m_rHandJoints;
	Eigen::VectorXd m_targetPts;
	Eigen::VectorXd m_targetPts_weight;
	Eigen::VectorXd m_targetPts_weight_buffer;
	double m_pCoeff[TotalModel::NUM_SHAPE_COEFFICIENTS];
	float WEAK_WEIGHT;

	std::vector< std::pair<int, std::vector<double>> > corres_vertex2targetpt;
	int m_nCorrespond_adam2joints;
	int m_nCorrespond_adam2pts;
	int m_nResiduals;

	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> m_Vt;  // the vertex given current shape
	Eigen::Matrix<double, TotalModel::NUM_JOINTS, 3, Eigen::RowMajor> m_J0; // correspondent default joint given current shape
	int res_dim;
};