#pragma once
#include "handm.h"
#include <Eigen/Dense>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <cassert>

class HandFastCost: public ceres::CostFunction
{
public:
	HandFastCost(smpl::HandModel& handm, Eigen::MatrixXd &HandJoints, Eigen::MatrixXd &PAF, bool fit3d, bool fit2d, bool fitPAF, const double* K=nullptr, int regressor_type=0):
		handm_(handm), HandJoints_(HandJoints), PAF_(PAF), res_dim(0), num_PAF_constraint(PAF.cols()),
		fit3d_(fit3d), fit2d_(fit2d), fitPAF_(fitPAF), K_(K), start_2d_dim(fit3d?3:0), regressor_type(regressor_type)
	{
		assert(HandJoints_.rows() == 5);
		assert(HandJoints_.cols() == 21);
		assert(PAF_.rows() == 3);
		assert(PAF_.cols() == 20);
		assert(regressor_type >= 0 && regressor_type <= 1);
		if (fit3d) res_dim += 3;
		if (fit2d)
		{
			res_dim += 2;
			assert(K_);
		}
		m_nResiduals = res_dim * smpl::HandModel::NUM_JOINTS + (fitPAF ? 3 * num_PAF_constraint : 0);
		start_PAF = res_dim * smpl::HandModel::NUM_JOINTS;
		CostFunction::set_num_residuals(m_nResiduals);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(3); // Translation
		parameter_block_sizes->push_back(smpl::HandModel::NUM_POSE_PARAMETERS); // SMPL Pose  
		parameter_block_sizes->push_back(smpl::HandModel::NUM_SHAPE_COEFFICIENTS); // SMPL Shape Coefficients  
		std::cout << "m_nResiduals: " << m_nResiduals << std::endl;

		parentIndexes[handm_.update_inds_(0)].clear();
		parentIndexes[handm_.update_inds_(0)].push_back(handm_.update_inds_(0));
		for(auto i = 1u; i < smpl::HandModel::NUM_JOINTS; i++)
		{
            parentIndexes[handm_.update_inds_(i)] = std::vector<int>(1, handm_.update_inds_(i));
            while (parentIndexes[handm_.update_inds_(i)].back() != handm_.update_inds_(0))
                parentIndexes[handm_.update_inds_(i)].emplace_back(handm_.parents_[parentIndexes[handm_.update_inds_(i)].back()]);
            std::sort(parentIndexes[handm_.update_inds_(i)].begin(), parentIndexes[handm_.update_inds_(i)].end());
        }

        PAF_connection.resize(num_PAF_constraint);
        PAF_connection = {{ {{0, 21 - 21, 0, 22 - 21}}, {{0, 22 - 21, 0, 23 - 21}}, {{0, 23 - 21, 0, 24 - 21}}, {{0, 24 - 21, 0, 25 - 21}},  // left hand
							{{0, 21 - 21, 0, 26 - 21}}, {{0, 26 - 21, 0, 27 - 21}}, {{0, 27 - 21, 0, 28 - 21}}, {{0, 28 - 21, 0, 29 - 21}},
							{{0, 21 - 21, 0, 30 - 21}}, {{0, 30 - 21, 0, 31 - 21}}, {{0, 31 - 21, 0, 32 - 21}}, {{0, 32 - 21, 0, 33 - 21}},
							{{0, 21 - 21, 0, 34 - 21}}, {{0, 34 - 21, 0, 35 - 21}}, {{0, 35 - 21, 0, 36 - 21}}, {{0, 36 - 21, 0, 37 - 21}},
							{{0, 21 - 21, 0, 38 - 21}}, {{0, 38 - 21, 0, 39 - 21}}, {{0, 39 - 21, 0, 40 - 21}}, {{0, 40 - 21, 0, 41 - 21}},
		}};
		weight_joints = {{ 1, 1, 1, 1, 1,
						   1, 1, 1, 1,
						   1, 1, 1, 1,
						   1, 1, 1, 1,
						   1, 1, 1, 1
						}};

		if (regressor_type == 1)
		{
			total_vertex.clear();
			for (int k = 0; k < handm_.STB_wrist_reg.outerSize(); ++k)
			{
				for (Eigen::SparseMatrix<double>::InnerIterator it(handm_.STB_wrist_reg, k); it; ++it)
			    {
					total_vertex.push_back(k);
					break;  // now this vertex is used, go to next vertex
			    }
			}
		}
	}

	~HandFastCost() {}
	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;
	void ForwardKinematics(double const* p_euler, double const* c, double* J_data, double* dJdP_data, double* dJdc_data) const;
	void select_lbs(
	    std::vector<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>& MR,
	    std::vector<Eigen::Matrix<double, 3, 1>>& Mt,
		std::vector<Eigen::Matrix<double, 9, 3 * smpl::HandModel::NUM_JOINTS, Eigen::RowMajor>>& dMRdP,
		std::vector<Eigen::Matrix<double, 9, 3 * smpl::HandModel::NUM_JOINTS, Eigen::RowMajor>>& dMRdc,
		std::vector<Eigen::Matrix<double, 3, 3 * smpl::HandModel::NUM_JOINTS, Eigen::RowMajor>>& dMtdP,
		std::vector<Eigen::Matrix<double, 3, 3 * smpl::HandModel::NUM_JOINTS, Eigen::RowMajor>>& dMtdc,
		MatrixXdr &outVert, double* dVdP_data, double* dVdc_data) const;
	void SparseRegress(const Eigen::SparseMatrix<double>& reg, const double* V_data, const double* dVdP_data, const double* dVdc_data,
                       double* J_data, double* dJdP_data, double* dJdc_data) const;
	float weight_2d = 1.0f;
	float weight_PAF = 50.0f;

private:
	std::vector<std::array<uint, 4>> PAF_connection;
	bool fit3d_, fit2d_, fitPAF_;
	int res_dim;
	const int start_2d_dim;
	int start_PAF;
	int m_nResiduals;
	const double* K_;
	smpl::HandModel& handm_;
	Eigen::MatrixXd& HandJoints_;
	Eigen::MatrixXd& PAF_;
	std::array<std::vector<int>, smpl::HandModel::NUM_JOINTS> parentIndexes;
	std::array<float, smpl::HandModel::NUM_JOINTS> weight_joints;
	const int num_PAF_constraint;
	const int regressor_type;
	std::vector<int> total_vertex;
};