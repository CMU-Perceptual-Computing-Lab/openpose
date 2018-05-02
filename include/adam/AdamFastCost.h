#include <iostream>
#include "totalmodel.h"
#include <cassert>
#include "pose_to_transforms.h"
#include <cstring>
#include <chrono>

class AdamFastCost: public ceres::CostFunction
{
public:
	AdamFastCost(TotalModel &adam,
		Eigen::MatrixXd &Joints,
		Eigen::MatrixXd &rFoot,     //3x2 //Heel, Toe
		Eigen::MatrixXd &lFoot,
		Eigen::MatrixXd &faceJoints,
		Eigen::MatrixXd &lHandJoints,
		Eigen::MatrixXd &rHandJoints,
		double* p_adam_coeff
	): m_adam(adam), m_rfoot_joints(rFoot), m_lfoot_joints(lFoot), m_bodyJoints(Joints), m_FaceJoints(faceJoints),
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

		if (m_bodyJoints(0, 1) != 0.0)      //nose
			corres_vertex2targetpt.push_back(std::make_pair(8130, std::vector<double>(5)));     
		if (m_bodyJoints(0, 16) != 0.0)     //left ear
			corres_vertex2targetpt.push_back(std::make_pair(6970, std::vector<double>(5)));     
		if (m_bodyJoints(0, 18) != 0.0)     //right ear
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
		if (m_bodyJoints(0, 16) != 0.0)     //left ear
		{
			std::vector<double> targetPt = {m_bodyJoints(0, 16), m_bodyJoints(1, 16), m_bodyJoints(2, 16), m_bodyJoints(3, 16), m_bodyJoints(4, 16)};
			corres_vertex2targetpt[offset].second = targetPt;
			offset++;
		}

		if (m_bodyJoints(0, 18) != 0.0)     //right ear
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
		double** jacobians) const;

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

struct AdamFitData
{
	AdamFitData(TotalModel &adam, Eigen::MatrixXd &Joints, Eigen::MatrixXd &rFoot, Eigen::MatrixXd &lFoot,
		Eigen::MatrixXd &faceJoints, Eigen::MatrixXd &lHandJoints, Eigen::MatrixXd &rHandJoints, Eigen::MatrixXd& PAF,
		bool fit3D=false, bool fit2D=false, double* K=nullptr, bool fitPAF=false): 
		adam(adam), bodyJoints(Joints), rFoot(rFoot), lFoot(lFoot), faceJoints(faceJoints), lHandJoints(lHandJoints), rHandJoints(rHandJoints), PAF(PAF),
		fit3D(fit3D), fit2D(fit2D), K(K), fitPAF(fitPAF)
		{
			inner_weight.clear();
			inner_weight.push_back(5.0f);
		}
	const TotalModel &adam;
	const Eigen::MatrixXd& bodyJoints;
	const Eigen::MatrixXd& rFoot;     //3x2 //Heel, Toe
	const Eigen::MatrixXd& lFoot;
	const Eigen::MatrixXd& faceJoints;
	const Eigen::MatrixXd& lHandJoints;
	const Eigen::MatrixXd& rHandJoints;
	const Eigen::MatrixXd& PAF;
	const bool fit3D;
	const bool fit2D;
	const double* K;
	const bool fitPAF;
	std::vector<float> inner_weight;
};

class AdamFullCost: public ceres::CostFunction
{
public:
	AdamFullCost(const AdamFitData& fit_data): fit_data_(fit_data), res_dim(0), start_2d_dim(0), rigid_body(false),
		num_PAF_constraint(fit_data.PAF.cols()), num_inner(fit_data.inner_weight.size()), total_inner_dim(0)
	{
		if(fit_data_.fit3D)
		{
			res_dim += 3;
			start_2d_dim += 3;
		}
		if(fit_data_.fit2D)
		{
			assert(fit_data_.K);
			res_dim += 2;
		}
		if(fit_data_.fitPAF)
		{
			assert(fit_data_.PAF.size() > 0 && fit_data_.PAF.rows() == 3);
			assert(fit_data_.PAF.cols() == PAF_connection.size());
			assert(fit_data_.PAF.cols() == PAF_weight.size());
		}
		assert(num_inner == DEFINED_INNER_CONSTRAINTS);
		inner_dim.clear();
		inner_dim.push_back(3);
		assert(inner_dim.size() == DEFINED_INNER_CONSTRAINTS);

		SetupCost();

		// setup parent indexes, for fast LBS jacobian computation
		parentIndexes[0].clear();
		parentIndexes[0].push_back(0);
		for(auto i = 0u; i < TotalModel::NUM_JOINTS; i++)
		{
            parentIndexes[i] = std::vector<int>(1, i);
            while (parentIndexes[i].back() != 0)
                parentIndexes[i].emplace_back(fit_data_.adam.m_parent[parentIndexes[i].back()]);
            std::sort(parentIndexes[i].begin(), parentIndexes[i].end());
        }

        dVdP_data = new double[3 * corres_vertex2targetpt.size() * TotalModel::NUM_POSE_PARAMETERS];
        dVdc_data = new double[3 * corres_vertex2targetpt.size() * TotalModel::NUM_SHAPE_COEFFICIENTS];
	}

	~AdamFullCost() {delete[] dVdc_data; delete[] dVdP_data;}

	void SetupCost()
	{
		using namespace cv;
		using namespace Eigen;

		// calculating the dim for 3D / 2D constraints
		m_nCorrespond_adam2joints = fit_data_.adam.m_indices_jointConst_adamIdx.rows();
		if(fit_data_.lHandJoints.size() > 0) m_nCorrespond_adam2joints += fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows();
		if(fit_data_.rHandJoints.size() > 0) m_nCorrespond_adam2joints += fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows();
		int count_dim = res_dim * m_nCorrespond_adam2joints;

		m_targetPts.resize(m_nCorrespond_adam2joints * 5);
		m_targetPts.setZero();
		m_targetPts_weight.resize(m_nCorrespond_adam2joints * res_dim);
		m_targetPts_weight_buffer.resize(m_nCorrespond_adam2joints * res_dim);
		SetupWeight();

		// setting up the vertex
		start_vertex = count_dim;
		corres_vertex2targetpt.push_back(std::make_pair(8130, std::vector<double>(5))); //nose
		corres_vertex2targetpt.push_back(std::make_pair(6970, std::vector<double>(5))); //left ear
		corres_vertex2targetpt.push_back(std::make_pair(10088, std::vector<double>(5))); //right ear
		corres_vertex2targetpt.push_back(std::make_pair(1372, std::vector<double>(5))); //head top
		m_nCorrespond_adam2pts = corres_vertex2targetpt.size();
		count_dim += m_nCorrespond_adam2pts * res_dim;

		// copy the fitting target in place
		UpdateTarget();

		// start counting from PAF
		start_PAF = count_dim;
		if (fit_data_.fitPAF) count_dim += 3 * num_PAF_constraint;

		// counting for inner constraints
		start_inner = count_dim;
		for (auto& n: inner_dim)
		{
			count_dim += n;
			total_inner_dim += n;
		}

		// setting num_residuals
		m_nResiduals = count_dim;
		std::cout << "m_nCorrespond_adam2joints " << m_nCorrespond_adam2joints << std::endl;
		std::cout << "m_nCorrespond_adam2pts " << m_nCorrespond_adam2pts << std::endl;
		std::cout << "m_nResiduals " << m_nResiduals << std::endl;
		std::cout << "res_dim " << res_dim << std::endl;
		std::cout << "start_2d_dim " << start_2d_dim << std::endl;
		std::cout << "start_PAF " << start_PAF << std::endl;
		std::cout << "start_vertex " << start_vertex << std::endl;
		std::cout << "start_inner " << start_inner << std::endl;
		CostFunction::set_num_residuals(m_nResiduals);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(3); // Translation
		parameter_block_sizes->push_back(TotalModel::NUM_JOINTS * 3); // SMPL Pose  
		parameter_block_sizes->push_back(TotalModel::NUM_SHAPE_COEFFICIENTS); // SMPL Pose  
	}

	void SetupWeight()
	{
		double BODYJOINT_WEIGHT_Strong = 1;
		double BODYJOINT_WEIGHT = 1;
		double HANDJOINT_WEIGHT = 0.2;

		int offset = 0;
		for (int ic = 0; ic < fit_data_.adam.m_indices_jointConst_adamIdx.rows(); ic++)
		{
			int smcjoint = fit_data_.adam.m_indices_jointConst_smcIdx(ic);
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
		offset += fit_data_.adam.m_indices_jointConst_adamIdx.rows();

		if (fit_data_.lHandJoints.size() > 0)
		{
			for (int ic = 0; ic < fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows(); ic++)
			{
				for (int i = 0; i < res_dim; i++)
				{
					m_targetPts_weight[(ic + offset) * res_dim + i] = HANDJOINT_WEIGHT;
					m_targetPts_weight_buffer[(ic + offset) * res_dim + i] = HANDJOINT_WEIGHT;
				}
			}
			offset += fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows();
		}

		if (fit_data_.rHandJoints.size() > 0)
		{
			for (int ic = 0; ic < fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows(); ic++)
			{
				for (int i = 0; i < res_dim; i++)
				{
					m_targetPts_weight[(ic + offset) * res_dim + i] = HANDJOINT_WEIGHT;
					m_targetPts_weight_buffer[(ic + offset) * res_dim + i] = HANDJOINT_WEIGHT;
				}
			}
		}
	}

	void UpdateTarget()
	{
		int offset = 0;
		for (int ic = 0; ic < fit_data_.adam.m_indices_jointConst_adamIdx.rows(); ic++)
			m_targetPts.block(ic * 5, 0, 5, 1) = fit_data_.bodyJoints.block(0, fit_data_.adam.m_indices_jointConst_smcIdx(ic), 5, 1);
		offset += fit_data_.adam.m_indices_jointConst_adamIdx.rows();

		if (fit_data_.lHandJoints.size() > 0)
		{
			for (int ic = 0; ic < fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows(); ic++)
				m_targetPts.block((offset + ic) * 5, 0, 5, 1) = fit_data_.lHandJoints.block(0, fit_data_.adam.m_correspond_adam2lHand_lHandIdx(ic), 5, 1);
		}
		offset += fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows();

		if (fit_data_.rHandJoints.size() > 0)
		{
			for (int ic = 0; ic < fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows(); ic++)
				m_targetPts.block((offset + ic) * 5, 0, 5, 1) = fit_data_.rHandJoints.block(0, fit_data_.adam.m_correspond_adam2rHand_rHandIdx(ic), 5, 1);
		}

		corres_vertex2targetpt[0].second = {fit_data_.bodyJoints(0, 1), fit_data_.bodyJoints(1, 1), fit_data_.bodyJoints(2, 1), fit_data_.bodyJoints(3, 1), fit_data_.bodyJoints(4, 1)};
		corres_vertex2targetpt[1].second = {fit_data_.bodyJoints(0, 16), fit_data_.bodyJoints(1, 16), fit_data_.bodyJoints(2, 16), fit_data_.bodyJoints(3, 16), fit_data_.bodyJoints(4, 16)};
		corres_vertex2targetpt[2].second = {fit_data_.bodyJoints(0, 18), fit_data_.bodyJoints(1, 18), fit_data_.bodyJoints(2, 18), fit_data_.bodyJoints(3, 18), fit_data_.bodyJoints(4, 18)};
		corres_vertex2targetpt[3].second = {fit_data_.bodyJoints(0, 19), fit_data_.bodyJoints(1, 19), fit_data_.bodyJoints(2, 19), fit_data_.bodyJoints(3, 19), fit_data_.bodyJoints(4, 19)};
	}

	void toggle_activate(bool limb, bool finger)
	{
		for (int ic = 0; ic < fit_data_.adam.m_indices_jointConst_adamIdx.rows(); ic++)
		{
			int smcjoint = fit_data_.adam.m_indices_jointConst_smcIdx(ic);
			if (smcjoint != 3 && smcjoint != 6 && smcjoint != 9 && smcjoint != 12)
			{
				m_targetPts_weight.block(ic * res_dim, 0, res_dim, 1) = double(limb) * m_targetPts_weight_buffer.block(ic * res_dim, 0, res_dim, 1);
			}
		}

		int offset = fit_data_.adam.m_indices_jointConst_smcIdx.rows();
		for (int ic = 0; ic < fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows(); ic++)
			m_targetPts_weight.block((ic + offset) * res_dim, 0, res_dim, 1) = m_targetPts_weight_buffer.block((ic + offset) * res_dim, 0, res_dim, 1) * double(finger);
		offset += fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows();
		for (int ic = 0; ic < fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows(); ic++)
			m_targetPts_weight.block((ic + offset) * res_dim, 0, res_dim, 1) = m_targetPts_weight_buffer.block((ic + offset) * res_dim, 0, res_dim, 1) * double(finger);
	}

	void toggle_rigid_body(bool rigid)
	{
		rigid_body = rigid;
	}

	void select_lbs(
		const double* c,
		const Eigen::VectorXd& T,  // transformation
		const MatrixXdr &dTdP,
		const MatrixXdr &dTdc,
		MatrixXdr &outVert,
	    double* dVdP_data,    //output
	    double* dVdc_data     //output
		// MatrixXdr &dVdP,	//output
		// MatrixXdr &dVdc     //output
	) const;

	void select_lbs(
	    const double* c,
	    const Eigen::VectorXd& T,  // transformation
	    MatrixXdr &outVert
	) const;

	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

	Eigen::VectorXd m_targetPts_weight;
	Eigen::VectorXd m_targetPts_weight_buffer;
	std::array<float, 4> vertex_weight{{0.0, 0.0, 0.0, 0.0}};
	std::array<float, 54> PAF_weight{{ 150., 150., 150., 150., 150., 150.,
										 150., 150., 150.,
										 150., 150., 150.,
										 0., 0.,
										 5., 5., 5., 5.,  // left hand
										 5., 5., 5., 5.,
										 5., 5., 5., 5.,
										 5., 5., 5., 5.,
										 5., 5., 5., 5.,
										 5., 5., 5., 5.,  // right hand
										 5., 5., 5., 5.,
										 5., 5., 5., 5.,
										 5., 5., 5., 5.,
										 5., 5., 5., 5.
										}};

private:
	// input data
	const AdamFitData& fit_data_;
	// setting
	bool rigid_body;
	// data for joint / projection fitting
	Eigen::VectorXd m_targetPts;
	int start_2d_dim;
	// data for vertex fitting
	std::vector< std::pair<int, std::vector<double>> > corres_vertex2targetpt;
	int start_vertex;
	// counter
	int m_nCorrespond_adam2joints;
	int m_nCorrespond_adam2pts;
	int m_nResiduals;
	int res_dim;  // number of residuals per joint / vertex constraints
	// data for PAF fitting
	const int num_PAF_constraint;
	const std::array<std::array<uint, 4>, 54> PAF_connection{{ {{0, 12, 0, 2}}, {{0, 2, 0, 5}}, {{0, 5, 0, 8}}, {{0, 12, 0, 1}}, {{0, 1, 0, 4}}, {{0, 4, 0, 7}},
											   {{0, 12, 0, 17}}, {{0, 17, 0, 19}}, {{0, 19, 0, 21}},  // right arm
											   {{0, 12, 0, 16}}, {{0, 16, 0, 18}}, {{0, 18, 0, 20}},  // left arm
											   {{0, 12, 1, 0}}, {{0, 12, 1, 3}},  // neck -> nose, neck -> head top
											   {{0, 20, 0, 22}}, {{0, 22, 0, 23}}, {{0, 23, 0, 24}}, {{0, 24, 0, 25}},  // left hand
											   {{0, 20, 0, 26}}, {{0, 26, 0, 27}}, {{0, 27, 0, 28}}, {{0, 28, 0, 29}},
											   {{0, 20, 0, 30}}, {{0, 30, 0, 31}}, {{0, 31, 0, 32}}, {{0, 32, 0, 33}},
											   {{0, 20, 0, 34}}, {{0, 34, 0, 35}}, {{0, 35, 0, 36}}, {{0, 36, 0, 37}},
											   {{0, 20, 0, 38}}, {{0, 38, 0, 39}}, {{0, 39, 0, 40}}, {{0, 40, 0, 41}},
   											   {{0, 21, 0, 22 + 20}}, {{0, 22 + 20, 0, 23 + 20}}, {{0, 23 + 20, 0, 24 + 20}}, {{0, 24 + 20, 0, 25 + 20}},  // right hand
											   {{0, 21, 0, 26 + 20}}, {{0, 26 + 20, 0, 27 + 20}}, {{0, 27 + 20, 0, 28 + 20}}, {{0, 28 + 20, 0, 29 + 20}},
											   {{0, 21, 0, 30 + 20}}, {{0, 30 + 20, 0, 31 + 20}}, {{0, 31 + 20, 0, 32 + 20}}, {{0, 32 + 20, 0, 33 + 20}},
											   {{0, 21, 0, 34 + 20}}, {{0, 34 + 20, 0, 35 + 20}}, {{0, 35 + 20, 0, 36 + 20}}, {{0, 36 + 20, 0, 37 + 20}},
											   {{0, 21, 0, 38 + 20}}, {{0, 38 + 20, 0, 39 + 20}}, {{0, 39 + 20, 0, 40 + 20}}, {{0, 40 + 20, 0, 41 + 20}},
											   }};

	int start_PAF;
	// data for inner fitting
	std::vector<uint> inner_dim;
	const int num_inner;
	int total_inner_dim;
	static const int DEFINED_INNER_CONSTRAINTS = 1;
	int start_inner;

	// parent index
	std::array<std::vector<int>, TotalModel::NUM_JOINTS> parentIndexes;
	
	// jacobians
	double* dVdP_data;
	double* dVdc_data;
};
