#include <iostream>
#include "totalmodel.h"
#include <cassert>
#include "pose_to_transforms.h"
#include <cstring>
#include <chrono>

class AdamFastCost: public ceres::CostFunction
{
public:
	AdamFastCost(const TotalModel &adam,
		const Eigen::MatrixXd &Joints,
		const Eigen::MatrixXd &rFoot,     //3x2 //Heel, Toe
		const Eigen::MatrixXd &lFoot,
		const Eigen::MatrixXd &faceJoints,
		const Eigen::MatrixXd &lHandJoints,
		const Eigen::MatrixXd &rHandJoints,
		const double* const p_adam_coeff
	): m_adam(adam), m_rfoot_joints(rFoot), m_lfoot_joints(lFoot), m_bodyJoints(Joints), m_FaceJoints(faceJoints),
		m_lHandJoints(lHandJoints), m_rHandJoints(rHandJoints), res_dim(3), verbose(false)
	{
		assert(p_adam_coeff != NULL);
		std::copy(p_adam_coeff, p_adam_coeff + TotalModel::NUM_SHAPE_COEFFICIENTS, m_pCoeff);
		SetupCost();
	}
	virtual ~AdamFastCost() {}

	void UpdateJoints(const Eigen::MatrixXd &Joints, const Eigen::MatrixXd &rFoot, const Eigen::MatrixXd &lFoot, const Eigen::MatrixXd &faceJoints,
		const Eigen::MatrixXd &lHandJoints, const Eigen::MatrixXd &rHandJoints)
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

		const double BODYJOINT_WEIGHT_Strong = 1;
		const double BODYJOINT_WEIGHT = 1;
		const double HANDJOINT_WEIGHT = 1;
		const double VERTEX_WEIGHT = 1;

		m_nCorrespond_adam2joints = m_adam.m_indices_jointConst_adamIdx.rows();
		m_nCorrespond_adam2joints += m_adam.m_correspond_adam2lHand_adamIdx.rows();
		m_nCorrespond_adam2joints += m_adam.m_correspond_adam2rHand_adamIdx.rows();

		// 13 vertices in 11 constraints
		total_vertex.push_back(8130); // nose
		total_vertex.push_back(6731); // left eye
		total_vertex.push_back(6970); // left ear
		total_vertex.push_back(4131); // right eye
		total_vertex.push_back(10088); // right ear
		total_vertex.push_back(14328); // right bigtoe
		total_vertex.push_back(14288); // right littletoe
		total_vertex.push_back(14357); // right heel
		total_vertex.push_back(14361); // right heel
		total_vertex.push_back(12239); // left bigtoe
		total_vertex.push_back(12289); // left littletoe
		total_vertex.push_back(12368); // left heel
		total_vertex.push_back(12357); // left heel
		m_nCorrespond_adam2pts = 11;

		m_targetPts.resize((m_nCorrespond_adam2joints + m_nCorrespond_adam2pts) * 5);
		m_targetPts.setZero();
		m_targetPts_weight.resize(m_nCorrespond_adam2joints + m_nCorrespond_adam2pts);
		m_targetPts_weight_buffer.resize(m_nCorrespond_adam2joints + m_nCorrespond_adam2pts);

		UpdateTarget();  // copy target

		int offset = 0;  // set up the weights
		for (int ic = 0; ic < m_adam.m_indices_jointConst_adamIdx.rows(); ic++)
		{
			int smcjoint = m_adam.m_indices_jointConst_smcIdx(ic);
			if (smcjoint == 4 || smcjoint == 10 || smcjoint == 3 || smcjoint == 9 || smcjoint == 7 || smcjoint == 13)
			{
				m_targetPts_weight[ic] = BODYJOINT_WEIGHT_Strong;
				m_targetPts_weight_buffer[ic] = BODYJOINT_WEIGHT_Strong;
			}
			else
			{
				m_targetPts_weight[ic] = BODYJOINT_WEIGHT;
				m_targetPts_weight_buffer[ic] = BODYJOINT_WEIGHT;
			}
		}
		offset += m_adam.m_indices_jointConst_adamIdx.rows();

		for (int ic = 0; ic < m_adam.m_correspond_adam2lHand_adamIdx.rows(); ic++)
		{
			m_targetPts_weight[ic + offset] = HANDJOINT_WEIGHT;
			m_targetPts_weight_buffer[ic + offset] = HANDJOINT_WEIGHT;
		}
		offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();

		for (int ic = 0; ic < m_adam.m_correspond_adam2rHand_adamIdx.rows(); ic++)
		{
			for (int i = 0; i < res_dim; i++)
			{
				m_targetPts_weight[ic + offset] = HANDJOINT_WEIGHT;
				m_targetPts_weight_buffer[ic + offset] = HANDJOINT_WEIGHT;
			}
		}
		offset += m_adam.m_correspond_adam2rHand_adamIdx.rows();

		for (int ic = 0; ic < m_nCorrespond_adam2pts; ic++)
		{
			m_targetPts_weight[offset + ic] = VERTEX_WEIGHT;
			m_targetPts_weight_buffer[offset + ic] = VERTEX_WEIGHT;
		}

		m_nResiduals = (m_nCorrespond_adam2joints + m_nCorrespond_adam2pts) * res_dim;
		if(verbose) std::cout << "m_nResiduals " << m_nResiduals << std::endl;
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

		for (int ic = 0; ic < m_adam.m_correspond_adam2lHand_adamIdx.rows(); ic++)
			m_targetPts.block((offset + ic) * 5, 0, 5, 1) = m_lHandJoints.block(0, m_adam.m_correspond_adam2lHand_lHandIdx(ic), 5, 1);
		offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();

		for (int ic = 0; ic < m_adam.m_correspond_adam2rHand_adamIdx.rows(); ic++)
			m_targetPts.block((offset + ic) * 5, 0, 5, 1) = m_rHandJoints.block(0, m_adam.m_correspond_adam2rHand_rHandIdx(ic), 5, 1);
		offset += m_adam.m_correspond_adam2rHand_adamIdx.rows();

		m_targetPts.block<5, 1>(offset * 5, 0) = m_bodyJoints.block<5, 1>(0, 1); offset++;
		m_targetPts.block<5, 1>(offset * 5, 0) = m_bodyJoints.block<5, 1>(0, 15); offset++;
		m_targetPts.block<5, 1>(offset * 5, 0) = m_bodyJoints.block<5, 1>(0, 16); offset++;
		m_targetPts.block<5, 1>(offset * 5, 0) = m_bodyJoints.block<5, 1>(0, 17); offset++;
		m_targetPts.block<5, 1>(offset * 5, 0) = m_bodyJoints.block<5, 1>(0, 18); offset++;
		m_targetPts.block<5, 1>(offset * 5, 0) = m_rfoot_joints.block<5, 1>(0, 0); offset++;
		m_targetPts.block<5, 1>(offset * 5, 0) = m_rfoot_joints.block<5, 1>(0, 1); offset++;
		m_targetPts.block<5, 1>(offset * 5, 0) = m_rfoot_joints.block<5, 1>(0, 2); offset++;
		m_targetPts.block<5, 1>(offset * 5, 0) = m_lfoot_joints.block<5, 1>(0, 0); offset++;
		m_targetPts.block<5, 1>(offset * 5, 0) = m_lfoot_joints.block<5, 1>(0, 1); offset++;
		m_targetPts.block<5, 1>(offset * 5, 0) = m_lfoot_joints.block<5, 1>(0, 2); offset++;
	}

	void toggle_activate(bool limb, bool finger)
	{
		for (int ic = 0; ic<m_adam.m_indices_jointConst_adamIdx.rows(); ic++)
		{
			int smcjoint = m_adam.m_indices_jointConst_smcIdx(ic);
			if (smcjoint != 3 && smcjoint != 6 && smcjoint != 9 && smcjoint != 12)
			{
				m_targetPts_weight[ic] = double(limb) * m_targetPts_weight_buffer[ic];
			}
		}

		int offset = m_adam.m_indices_jointConst_smcIdx.rows();
		for (int ic = 0; ic < m_adam.m_correspond_adam2lHand_adamIdx.rows(); ic++)
			m_targetPts_weight[ic + offset] = m_targetPts_weight_buffer[ic + offset] * double(finger);
		offset += m_adam.m_correspond_adam2lHand_adamIdx.rows();
		for (int ic = 0; ic < m_adam.m_correspond_adam2rHand_adamIdx.rows(); ic++)
			m_targetPts_weight[ic + offset] = m_targetPts_weight_buffer[ic + offset] * double(finger);
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

	std::vector<int> total_vertex;
	int m_nCorrespond_adam2joints;
	int m_nCorrespond_adam2pts;
	int m_nResiduals;

	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> m_Vt;  // the vertex given current shape (before translation)
	Eigen::Matrix<double, TotalModel::NUM_JOINTS, 3, Eigen::RowMajor> m_J0; // correspondent default joint given current shape
	int res_dim;
	bool verbose;
};

struct AdamFitData
{
	AdamFitData(const TotalModel &adam, const Eigen::MatrixXd &Joints, const Eigen::MatrixXd &rFoot, const Eigen::MatrixXd &lFoot,
		const Eigen::MatrixXd &faceJoints, const Eigen::MatrixXd &lHandJoints, const Eigen::MatrixXd &rHandJoints, const Eigen::MatrixXd& PAF,
		const bool fit3D=false, const bool fit2D=false, const double* const K=nullptr, const bool fitPAF=false) :
		adam(adam), bodyJoints(Joints), rFoot(rFoot), lFoot(lFoot), faceJoints(faceJoints), lHandJoints(lHandJoints), rHandJoints(rHandJoints), PAF(PAF),
		fit3D(fit3D), fit2D(fit2D), K(K), fitPAF(fitPAF), inner_weight(1, 5.0f)
	{
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
	const std::vector<float> inner_weight;
};

class AdamFullCost: public ceres::CostFunction
{
public:
	AdamFullCost(const AdamFitData& fit_data, const int regressor_type=0, const bool fit_face_exp=false):
		res_dim(0), freeze_missing(true), fit_data_(fit_data), regressor_type(regressor_type), rigid_body(false), start_2d_dim(0),
		num_PAF_constraint(fit_data.PAF.cols()), num_inner(fit_data.inner_weight.size()), total_inner_dim(0),
		dVdfc_data(nullptr), dOdfc_data(nullptr), fit_face_exp(fit_face_exp)
	{
		if (fit_data_.fit3D)
		{
			res_dim += 3;
			start_2d_dim += 3;
		}
		if (fit_data_.fit2D)
		{
			assert(fit_data_.K);
			res_dim += 2;
		}
		if (fit_data_.fitPAF)
		{
			assert(fit_data_.PAF.size() > 0 && fit_data_.PAF.rows() == 3);
		}
		assert(num_inner == DEFINED_INNER_CONSTRAINTS);
		inner_dim.clear();
		inner_dim.push_back(3);
		assert(inner_dim.size() == DEFINED_INNER_CONSTRAINTS);
		assert(regressor_type >= 0 && regressor_type <= 2); // 0 for the default joints, 1 for Human3.6M regressor, 2 for COCO style output
		if (fit_face_exp) assert(regressor_type == 0 || regressor_type == 2); // must have face constraints in order to fit face expression.

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

        dVdP_data = new double[3 * total_vertex.size() * TotalModel::NUM_POSE_PARAMETERS];
        dVdc_data = new double[3 * total_vertex.size() * TotalModel::NUM_SHAPE_COEFFICIENTS];
        dOdP_data = new double[3 * (m_nCorrespond_adam2joints + m_nCorrespond_adam2pts) * TotalModel::NUM_POSE_PARAMETERS];
        dOdc_data = new double[3 * (m_nCorrespond_adam2joints + m_nCorrespond_adam2pts) * TotalModel::NUM_SHAPE_COEFFICIENTS];
        if(fit_face_exp)
    	{
    		dVdfc_data = new double[3 * total_vertex.size() * TotalModel::NUM_EXP_BASIS_COEFFICIENTS];
	        dOdfc_data = new double[3 * (m_nCorrespond_adam2joints + m_nCorrespond_adam2pts) * TotalModel::NUM_EXP_BASIS_COEFFICIENTS];
    	}
	}

	~AdamFullCost()
	{
		delete[] dVdc_data; delete[] dVdP_data; delete[] dOdP_data; delete[] dOdc_data;
		if (fit_face_exp)
		{
			delete[] dVdfc_data;
			delete[] dOdfc_data;
		}
	}

	void SetupCost()
	{
		using namespace cv;
		using namespace Eigen;

		if (regressor_type == 0)
		{
			// calculating the dim for 3D / 2D constraints
			m_nCorrespond_adam2joints = fit_data_.adam.m_indices_jointConst_adamIdx.rows();
			if(fit_data_.lHandJoints.size() > 0) m_nCorrespond_adam2joints += fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows();
			if(fit_data_.rHandJoints.size() > 0) m_nCorrespond_adam2joints += fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows();
			// These are all the vertices to compute
			total_vertex.push_back(8130); //nose
			total_vertex.push_back(6970); //left ear
			total_vertex.push_back(10088); //right ear
			total_vertex.push_back(1372); //head top
			total_vertex.push_back(14328); //right bigtoe
			total_vertex.push_back(14288); //right littletoe
			total_vertex.push_back(14357); //right heel
			total_vertex.push_back(14361); //right heel
			total_vertex.push_back(12239); //left bigtoe
			total_vertex.push_back(12289); //left littletoe
			total_vertex.push_back(12368); //left heel
			total_vertex.push_back(12357); //left heel
			// face detector results
			for (int r = 0; r < fit_data_.adam.m_correspond_adam2face70_adamIdx.rows(); ++r)
			{
				int adamVertexIdx = fit_data_.adam.m_correspond_adam2face70_adamIdx(r);
				total_vertex.push_back(adamVertexIdx);
			}
			// setting up the vertex (specify the vertices with constraints to optimize)
			corres_vertex2targetpt.push_back(std::make_pair(0, std::vector<double>(5))); 
			corres_vertex2targetpt.push_back(std::make_pair(1, std::vector<double>(5))); 
			corres_vertex2targetpt.push_back(std::make_pair(2, std::vector<double>(5))); 
			corres_vertex2targetpt.push_back(std::make_pair(3, std::vector<double>(5))); 
			corres_vertex2targetpt.push_back(std::make_pair(4, std::vector<double>(5))); 
			corres_vertex2targetpt.push_back(std::make_pair(5, std::vector<double>(5))); 
			corres_vertex2targetpt.push_back(std::make_pair(6, std::vector<double>(5))); 
			corres_vertex2targetpt.push_back(std::make_pair(7, std::vector<double>(5))); 
			corres_vertex2targetpt.push_back(std::make_pair(8, std::vector<double>(5))); 
			corres_vertex2targetpt.push_back(std::make_pair(9, std::vector<double>(5))); 
			corres_vertex2targetpt.push_back(std::make_pair(10, std::vector<double>(5))); 
			corres_vertex2targetpt.push_back(std::make_pair(11, std::vector<double>(5))); 
			for (int r = 0; r < fit_data_.adam.m_correspond_adam2face70_adamIdx.rows(); ++r)
			{
				corres_vertex2targetpt.push_back(std::make_pair(12 + r, std::vector<double>(5)));
			}
			m_nCorrespond_adam2pts = corres_vertex2targetpt.size();

			PAF_connection.resize(num_PAF_constraint);
			PAF_connection = {{ {{0, 12, 0, 2}}, {{0, 2, 0, 5}}, {{0, 5, 0, 8}}, {{0, 12, 0, 1}}, {{0, 1, 0, 4}}, {{0, 4, 0, 7}},
											   {{0, 12, 0, 17}}, {{0, 17, 0, 19}}, {{0, 19, 0, 21}},  // right arm
											   {{0, 12, 0, 16}}, {{0, 16, 0, 18}}, {{0, 18, 0, 20}},  // left arm
											   {{0, 12, 1, 0}}, {{0, 12, 1, 3}},  // neck -> nose, neck -> head top
											   {{0, 17, 0, 21}}, {{0, 16, 0, 20}},  // shoulder -> wrist
											   {{0, 2, 0, 21}}, {{0, 1, 0, 20}},  // hip -> wrist
											   {{0, 2, 0, 8}}, {{0, 1, 0, 7}},  // hip -> ankle
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
			PAF_weight.resize(num_PAF_constraint);
			PAF_weight = {{ 150., 150., 150., 150., 150., 150.,
										 150., 150., 150.,
										 150., 150., 150.,
										 0., 0.,
										 50., 50.,
										 50., 50.,
										 50., 50.,
										 50., 50., 50., 50.,  // left hand
										 50., 50., 50., 50.,
										 50., 50., 50., 50.,
										 50., 50., 50., 50.,
										 50., 50., 50., 50.,
										 50., 50., 50., 50.,  // right hand
										 50., 50., 50., 50.,
										 50., 50., 50., 50.,
										 50., 50., 50., 50.,
										 50., 50., 50., 50.
			}};
		}
		else if (regressor_type == 1)
		{
			m_nCorrespond_adam2joints = fit_data_.adam.h36m_jointConst_smcIdx.size();
			int count_vertex = 0;
			for (int k = 0; k < fit_data_.adam.m_cocoplus_reg.outerSize(); ++k)
			{
				for (SparseMatrix<double>::InnerIterator it(fit_data_.adam.m_cocoplus_reg, k); it; ++it)
			    {
					total_vertex.push_back(k);
					count_vertex++;
					break;  // now this vertex is used, go to next vertex
			    }
			}
			corres_vertex2targetpt.clear();
			m_nCorrespond_adam2pts = 0;

			PAF_connection.resize(num_PAF_constraint);
			PAF_connection = {{ {{2, 12, 2, 2}}, {{2, 2, 2, 1}}, {{2, 1, 2, 0}}, {{2, 12, 2, 3}}, {{2, 3, 2, 4}}, {{2, 4, 2, 5}},
											   {{2, 12, 2, 8}}, {{2, 8, 2, 7}}, {{2, 7, 2, 6}},  // right arm
											   {{2, 12, 2, 9}}, {{2, 9, 2, 10}}, {{2, 10, 2, 11}},  // left arm
											   {{2, 12, 2, 14}}, {{2, 12, 2, 13}},  // neck -> nose, neck -> head top
											   {{2, 8, 2, 6}}, {{2, 9, 2, 11}},  // shoulder -> wrist
											   {{2, 2, 2, 6}}, {{2, 3, 2, 11}},  // hip -> wrist
											   {{2, 2, 2, 0}}, {{2, 3, 2, 5}},  // hip -> ankle
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
			PAF_weight.resize(num_PAF_constraint);
			PAF_weight = {{ 150., 150., 150., 150., 150., 150.,
										 150., 150., 150.,
										 150., 150., 150.,
										 0., 0.,
										 50., 50.,
										 50., 50.,
										 50., 50.,
										 50., 50., 50., 50.,  // left hand
										 50., 50., 50., 50.,
										 50., 50., 50., 50.,
										 50., 50., 50., 50.,
										 50., 50., 50., 50.,
										 50., 50., 50., 50.,  // right hand
										 50., 50., 50., 50.,
										 50., 50., 50., 50.,
										 50., 50., 50., 50.,
										 50., 50., 50., 50.
			}};

			map_regressor_to_constraint.clear();
			for (int i = 0; i < 19; i++) map_regressor_to_constraint[i] = i;
		}
		else
		{
			assert(regressor_type == 2);
			m_nCorrespond_adam2joints = fit_data_.adam.h36m_jointConst_smcIdx.size() + 20 * 2;  // COCO keypoints plus fingers
			int count_vertex = 0;
			for (int k = 0; k < fit_data_.adam.m_small_coco_reg.outerSize(); ++k)
			{
				for (SparseMatrix<double>::InnerIterator it(fit_data_.adam.m_small_coco_reg, k); it; ++it)
			    {
					total_vertex.push_back(k);
					count_vertex++;
					break;  // now this vertex is used, go to next vertex
			    }
			}
			// Set up vertex constraints
			corres_vertex2targetpt.clear();
			total_vertex.push_back(14328); //right bigtoe
			corres_vertex2targetpt.push_back(std::make_pair(count_vertex++, std::vector<double>(5))); 
			total_vertex.push_back(14288); //right littletoe
			corres_vertex2targetpt.push_back(std::make_pair(count_vertex++, std::vector<double>(5))); 
			total_vertex.push_back(14357); //right heel
			corres_vertex2targetpt.push_back(std::make_pair(count_vertex++, std::vector<double>(5))); 
			total_vertex.push_back(14361); //right heel
			corres_vertex2targetpt.push_back(std::make_pair(count_vertex++, std::vector<double>(5))); 
			total_vertex.push_back(12239); //left bigtoe
			corres_vertex2targetpt.push_back(std::make_pair(count_vertex++, std::vector<double>(5))); 
			total_vertex.push_back(12289); //left littletoe
			corres_vertex2targetpt.push_back(std::make_pair(count_vertex++, std::vector<double>(5))); 
			total_vertex.push_back(12368); //left heel
			corres_vertex2targetpt.push_back(std::make_pair(count_vertex++, std::vector<double>(5))); 
			total_vertex.push_back(12357); //left heel
			corres_vertex2targetpt.push_back(std::make_pair(count_vertex++, std::vector<double>(5))); 
			// face detector results
			for (int r = 0; r < fit_data_.adam.m_correspond_adam2face70_adamIdx.rows(); ++r)
			{
				int adamVertexIdx = fit_data_.adam.m_correspond_adam2face70_adamIdx(r);
				total_vertex.push_back(adamVertexIdx);
				corres_vertex2targetpt.push_back(std::make_pair(count_vertex++, std::vector<double>(5))); 
			}
			m_nCorrespond_adam2pts = corres_vertex2targetpt.size();

			PAF_connection.resize(num_PAF_constraint);
			PAF_connection = {{ {{2, 12, 2, 2}}, {{2, 2, 2, 1}}, {{2, 1, 2, 0}}, {{2, 12, 2, 3}}, {{2, 3, 2, 4}}, {{2, 4, 2, 5}},
											   {{2, 12, 2, 8}}, {{2, 8, 2, 7}}, {{2, 7, 2, 6}},  // right arm
											   {{2, 12, 2, 9}}, {{2, 9, 2, 10}}, {{2, 10, 2, 11}},  // left arm
											   {{2, 12, 2, 14}}, {{2, 12, 2, 13}},  // neck -> nose, neck -> head top
   											   {{2, 8, 2, 6}}, {{2, 9, 2, 11}},  // shoulder -> wrist
											   {{2, 2, 2, 6}}, {{2, 3, 2, 11}},  // hip -> wrist
											   {{2, 2, 2, 0}}, {{2, 3, 2, 5}},  // hip -> ankle
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
			PAF_weight.resize(num_PAF_constraint);
			PAF_weight = {{ 150., 150., 150., 150., 150., 150.,
										 150., 150., 150.,
										 150., 150., 150.,
										 50., 50.,  // neck to nose + neck to headtop
 										 50., 50.,
										 50., 50.,
										 50., 50.,
										 50., 50., 50., 50.,  // left hand
										 50., 50., 50., 50.,
										 50., 50., 50., 50.,
										 50., 50., 50., 50.,
										 50., 50., 50., 50.,
										 50., 50., 50., 50.,  // right hand
										 50., 50., 50., 50.,
										 50., 50., 50., 50.,
										 50., 50., 50., 50.,
										 50., 50., 50., 50.
			}};

			map_regressor_to_constraint.clear();
			for (int i = 0; i < 19; i++) map_regressor_to_constraint[i] = i;
		}

		m_targetPts.resize((m_nCorrespond_adam2joints + m_nCorrespond_adam2pts) * 5);
		m_targetPts.setZero();
		m_targetPts_weight.resize(m_nCorrespond_adam2joints + m_nCorrespond_adam2pts);
		m_targetPts_weight_buffer.resize(m_nCorrespond_adam2joints + m_nCorrespond_adam2pts);

		// copy the fitting target in place
		SetupWeight();
		UpdateTarget();

		// start counting from PAF
		int count_dim = (m_nCorrespond_adam2joints + m_nCorrespond_adam2pts) * res_dim;
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
		// std::cout << "m_nCorrespond_adam2joints " << m_nCorrespond_adam2joints << std::endl;
		// std::cout << "m_nCorrespond_adam2pts " << m_nCorrespond_adam2pts << std::endl;
		// std::cout << "m_nResiduals " << m_nResiduals << std::endl;
		// std::cout << "res_dim " << res_dim << std::endl;
		// std::cout << "start_2d_dim " << start_2d_dim << std::endl;
		// std::cout << "start_PAF " << start_PAF << std::endl;
		// std::cout << "start_inner " << start_inner << std::endl;
		CostFunction::set_num_residuals(m_nResiduals);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(3); // Translation
		parameter_block_sizes->push_back(TotalModel::NUM_JOINTS * 3); // SMPL Pose  
		parameter_block_sizes->push_back(TotalModel::NUM_SHAPE_COEFFICIENTS); // SMPL Pose  
		if (fit_face_exp) parameter_block_sizes->push_back(TotalModel::NUM_EXP_BASIS_COEFFICIENTS); // facial expression
	}

	void SetupWeight()
	{
		double BODYJOINT_WEIGHT_Strong = 1;
		double BODYJOINT_WEIGHT = 1;
		double HANDJOINT_WEIGHT = 0.5;
		double VERTEX_WEIGHT = 0;

		if (regressor_type == 0)
		{
			int offset = 0;
			for (int ic = 0; ic < fit_data_.adam.m_indices_jointConst_adamIdx.rows(); ic++)
			{
				int smcjoint = fit_data_.adam.m_indices_jointConst_smcIdx(ic);
				if (smcjoint == 4 || smcjoint == 10 || smcjoint == 3 || smcjoint == 9 || smcjoint == 7 || smcjoint == 13)
				{
					m_targetPts_weight[ic] = BODYJOINT_WEIGHT_Strong;
					m_targetPts_weight_buffer[ic] = BODYJOINT_WEIGHT_Strong;
				}
				else
				{
					m_targetPts_weight[ic] = BODYJOINT_WEIGHT;
					m_targetPts_weight_buffer[ic] = BODYJOINT_WEIGHT;
				}
			}
			offset += fit_data_.adam.m_indices_jointConst_adamIdx.rows();

			if (fit_data_.lHandJoints.size() > 0)
			{
				for (int ic = 0; ic < fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows(); ic++)
				{
					m_targetPts_weight[(ic + offset)] = HANDJOINT_WEIGHT;
					m_targetPts_weight_buffer[(ic + offset)] = HANDJOINT_WEIGHT;
				}
				offset += fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows();
			}

			if (fit_data_.rHandJoints.size() > 0)
			{
				for (int ic = 0; ic < fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows(); ic++)
				{
					m_targetPts_weight[ic + offset] = HANDJOINT_WEIGHT;
					m_targetPts_weight_buffer[ic + offset] = HANDJOINT_WEIGHT;
				}
				offset += fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows();
			}

			for (auto i = 0u; i < corres_vertex2targetpt.size(); i++)
			{
				m_targetPts_weight[i + offset] = VERTEX_WEIGHT;
				m_targetPts_weight_buffer[i + offset] = VERTEX_WEIGHT;
			}
		}
		else if (regressor_type == 1)
		{
			for (auto i = 0u; i < fit_data_.adam.h36m_jointConst_smcIdx.size(); i++)
			{
				m_targetPts_weight[i] = BODYJOINT_WEIGHT;
				m_targetPts_weight_buffer[i] = BODYJOINT_WEIGHT;
			}
		}
		else
		{
			assert(regressor_type == 2);
			for (auto i = 0u; i < fit_data_.adam.h36m_jointConst_smcIdx.size(); i++)
			{
				m_targetPts_weight[i] = BODYJOINT_WEIGHT;
				m_targetPts_weight_buffer[i] = BODYJOINT_WEIGHT;
			}
			int offset = fit_data_.adam.h36m_jointConst_smcIdx.size();
			for (int i = 0; i < 20 * 2; i++)
			{
				m_targetPts_weight[offset + i] = HANDJOINT_WEIGHT;
				m_targetPts_weight_buffer[offset + i] = HANDJOINT_WEIGHT;
			}
			offset += 40;
			for (auto i = 0u; i < corres_vertex2targetpt.size(); i++)
			{
				m_targetPts_weight[i + offset] = VERTEX_WEIGHT;
				m_targetPts_weight_buffer[i + offset] = VERTEX_WEIGHT;
			}
		}
	}

	void UpdateTarget()
	{
		int offset = 0;
		if (regressor_type == 0)
		{
			for (int ic = 0; ic < fit_data_.adam.m_indices_jointConst_adamIdx.rows(); ic++)
				m_targetPts.block(ic * 5, 0, 5, 1) = fit_data_.bodyJoints.block(0, fit_data_.adam.m_indices_jointConst_smcIdx(ic), 5, 1);
			offset += fit_data_.adam.m_indices_jointConst_adamIdx.rows();

			if (fit_data_.lHandJoints.size() > 0)
			{
				for (int ic = 0; ic < fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows(); ic++)
					m_targetPts.block((offset + ic) * 5, 0, 5, 1) = fit_data_.lHandJoints.block(0, fit_data_.adam.m_correspond_adam2lHand_lHandIdx(ic), 5, 1);
				offset += fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows();
			}

			if (fit_data_.rHandJoints.size() > 0)
			{
				for (int ic = 0; ic < fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows(); ic++)
					m_targetPts.block((offset + ic) * 5, 0, 5, 1) = fit_data_.rHandJoints.block(0, fit_data_.adam.m_correspond_adam2rHand_rHandIdx(ic), 5, 1);
				offset += fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows();
			}

			corres_vertex2targetpt[0].second = {fit_data_.bodyJoints(0, 1), fit_data_.bodyJoints(1, 1), fit_data_.bodyJoints(2, 1), fit_data_.bodyJoints(3, 1), fit_data_.bodyJoints(4, 1)};
			corres_vertex2targetpt[1].second = {fit_data_.bodyJoints(0, 16), fit_data_.bodyJoints(1, 16), fit_data_.bodyJoints(2, 16), fit_data_.bodyJoints(3, 16), fit_data_.bodyJoints(4, 16)};
			corres_vertex2targetpt[2].second = {fit_data_.bodyJoints(0, 18), fit_data_.bodyJoints(1, 18), fit_data_.bodyJoints(2, 18), fit_data_.bodyJoints(3, 18), fit_data_.bodyJoints(4, 18)};
			corres_vertex2targetpt[3].second = {fit_data_.bodyJoints(0, 19), fit_data_.bodyJoints(1, 19), fit_data_.bodyJoints(2, 19), fit_data_.bodyJoints(3, 19), fit_data_.bodyJoints(4, 19)};
			corres_vertex2targetpt[4].second = {fit_data_.rFoot(0, 0), fit_data_.rFoot(1, 0), fit_data_.rFoot(2, 0), fit_data_.rFoot(3, 0), fit_data_.rFoot(4, 0)};
			corres_vertex2targetpt[5].second = {fit_data_.rFoot(0, 1), fit_data_.rFoot(1, 1), fit_data_.rFoot(2, 1), fit_data_.rFoot(3, 1), fit_data_.rFoot(4, 1)};
			corres_vertex2targetpt[6].second = {fit_data_.rFoot(0, 2), fit_data_.rFoot(1, 2), fit_data_.rFoot(2, 2), fit_data_.rFoot(3, 2), fit_data_.rFoot(4, 2)};
			corres_vertex2targetpt[7].second = {fit_data_.rFoot(0, 2), fit_data_.rFoot(1, 2), fit_data_.rFoot(2, 2), fit_data_.rFoot(3, 2), fit_data_.rFoot(4, 2)};
			corres_vertex2targetpt[8].second = {fit_data_.lFoot(0, 0), fit_data_.lFoot(1, 0), fit_data_.lFoot(2, 0), fit_data_.lFoot(3, 0), fit_data_.lFoot(4, 0)};
			corres_vertex2targetpt[9].second = {fit_data_.lFoot(0, 1), fit_data_.lFoot(1, 1), fit_data_.lFoot(2, 1), fit_data_.lFoot(3, 1), fit_data_.lFoot(4, 1)};
			corres_vertex2targetpt[10].second = {fit_data_.lFoot(0, 2), fit_data_.lFoot(1, 2), fit_data_.lFoot(2, 2), fit_data_.lFoot(3, 2), fit_data_.lFoot(4, 2)};
			corres_vertex2targetpt[11].second = {fit_data_.lFoot(0, 2), fit_data_.lFoot(1, 2), fit_data_.lFoot(2, 2), fit_data_.lFoot(3, 2), fit_data_.lFoot(4, 2)};
			for (int r = 0; r < fit_data_.adam.m_correspond_adam2face70_adamIdx.rows(); ++r)
			{
				int face70ID = fit_data_.adam.m_correspond_adam2face70_face70Idx(r);
				if (face70ID < 0) corres_vertex2targetpt[12 + r].second = {{0.0, 0.0, 0.0, 0.0, 0.0}};
				else
					corres_vertex2targetpt[12 + r].second = {{fit_data_.faceJoints(0, face70ID), fit_data_.faceJoints(1, face70ID), fit_data_.faceJoints(2, face70ID), fit_data_.faceJoints(3, face70ID), fit_data_.faceJoints(4, face70ID)}};
			}

			for (auto i = 0u; i < corres_vertex2targetpt.size(); i++)
			{
				std::copy(corres_vertex2targetpt[i].second.data(), corres_vertex2targetpt[i].second.data() + 5, m_targetPts.data() + 5 * (i + offset));
			}
		}
		else if (regressor_type == 1)
		{
			for(auto i = 0u; i < fit_data_.adam.h36m_jointConst_smcIdx.size(); i++)
				m_targetPts.block(5 * i, 0, 5, 1) = fit_data_.bodyJoints.col(fit_data_.adam.h36m_jointConst_smcIdx[i]);
		}
		else
		{
			assert (regressor_type == 2);
			for(auto i = 0u; i < fit_data_.adam.h36m_jointConst_smcIdx.size(); i++)
				m_targetPts.block(5 * i, 0, 5, 1) = fit_data_.bodyJoints.col(fit_data_.adam.h36m_jointConst_smcIdx[i]);

			int offset = fit_data_.adam.h36m_jointConst_smcIdx.size();
			if (fit_data_.lHandJoints.size() > 0)
			{
				for (int ic = 0; ic < fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows(); ic++)
					m_targetPts.block((offset + ic) * 5, 0, 5, 1) = fit_data_.lHandJoints.block(0, fit_data_.adam.m_correspond_adam2lHand_lHandIdx(ic), 5, 1);
				offset += fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows();
			}

			if (fit_data_.rHandJoints.size() > 0)
			{
				for (int ic = 0; ic < fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows(); ic++)
					m_targetPts.block((offset + ic) * 5, 0, 5, 1) = fit_data_.rHandJoints.block(0, fit_data_.adam.m_correspond_adam2rHand_rHandIdx(ic), 5, 1);
				offset += fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows();
			}
			assert(offset == m_nCorrespond_adam2joints); // check the number of constraints

			offset = 0;
			corres_vertex2targetpt[offset++].second = {fit_data_.rFoot(0, 0), fit_data_.rFoot(1, 0), fit_data_.rFoot(2, 0), fit_data_.rFoot(3, 0), fit_data_.rFoot(4, 0)};
			corres_vertex2targetpt[offset++].second = {fit_data_.rFoot(0, 1), fit_data_.rFoot(1, 1), fit_data_.rFoot(2, 1), fit_data_.rFoot(3, 1), fit_data_.rFoot(4, 1)};
			corres_vertex2targetpt[offset++].second = {fit_data_.rFoot(0, 2), fit_data_.rFoot(1, 2), fit_data_.rFoot(2, 2), fit_data_.rFoot(3, 2), fit_data_.rFoot(4, 2)};
			corres_vertex2targetpt[offset++].second = {fit_data_.rFoot(0, 2), fit_data_.rFoot(1, 2), fit_data_.rFoot(2, 2), fit_data_.rFoot(3, 2), fit_data_.rFoot(4, 2)};
			corres_vertex2targetpt[offset++].second = {fit_data_.lFoot(0, 0), fit_data_.lFoot(1, 0), fit_data_.lFoot(2, 0), fit_data_.lFoot(3, 0), fit_data_.lFoot(4, 0)};
			corres_vertex2targetpt[offset++].second = {fit_data_.lFoot(0, 1), fit_data_.lFoot(1, 1), fit_data_.lFoot(2, 1), fit_data_.lFoot(3, 1), fit_data_.lFoot(4, 1)};
			corres_vertex2targetpt[offset++].second = {fit_data_.lFoot(0, 2), fit_data_.lFoot(1, 2), fit_data_.lFoot(2, 2), fit_data_.lFoot(3, 2), fit_data_.lFoot(4, 2)};
			corres_vertex2targetpt[offset++].second = {fit_data_.lFoot(0, 2), fit_data_.lFoot(1, 2), fit_data_.lFoot(2, 2), fit_data_.lFoot(3, 2), fit_data_.lFoot(4, 2)};
			for (int r = 0; r < fit_data_.adam.m_correspond_adam2face70_adamIdx.rows(); ++r)
			{
				int face70ID = fit_data_.adam.m_correspond_adam2face70_face70Idx(r);
				if (face70ID < 0) corres_vertex2targetpt[offset++].second = {{0.0, 0.0, 0.0, 0.0, 0.0}};
				else
					corres_vertex2targetpt[offset++].second = {{fit_data_.faceJoints(0, face70ID), fit_data_.faceJoints(1, face70ID), fit_data_.faceJoints(2, face70ID), fit_data_.faceJoints(3, face70ID), fit_data_.faceJoints(4, face70ID)}};
			}
			assert(offset == m_nCorrespond_adam2pts); // check the number of constraints

			offset = m_nCorrespond_adam2joints;
			for (auto i = 0u; i < corres_vertex2targetpt.size(); i++)
			{
				std::copy(corres_vertex2targetpt[i].second.data(), corres_vertex2targetpt[i].second.data() + 5, m_targetPts.data() + 5 * (i + offset));
			}
		}
	}

	void toggle_activate(bool limb, bool palm, bool finger)
	{
		if (regressor_type == 0)
		{
			for (int ic = 0; ic < fit_data_.adam.m_indices_jointConst_adamIdx.rows(); ic++)
			{
				int smcjoint = fit_data_.adam.m_indices_jointConst_smcIdx(ic);
				if (smcjoint != 3 && smcjoint != 6 && smcjoint != 9 && smcjoint != 12)
				{
					m_targetPts_weight[ic] = double(limb) * m_targetPts_weight_buffer[ic];
				}
			}

			int offset = fit_data_.adam.m_indices_jointConst_smcIdx.rows();
			for (int ic = 0; ic < fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows(); ic++)
			{
				if (ic % 5 == 0)
					m_targetPts_weight[ic + offset] = m_targetPts_weight_buffer[ic + offset] * double(palm);
				else
					m_targetPts_weight[ic + offset] = m_targetPts_weight_buffer[ic + offset] * double(finger);
			}
			offset += fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows();
			for (int ic = 0; ic < fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows(); ic++)
			{
				if (ic % 5 == 0)
					m_targetPts_weight[ic + offset] = m_targetPts_weight_buffer[ic + offset] * double(palm);
				else
					m_targetPts_weight[ic + offset] = m_targetPts_weight_buffer[ic + offset] * double(finger);
			}
		}
		else if (regressor_type == 1)
		{
			for (auto ic = 0u; ic < fit_data_.adam.h36m_jointConst_smcIdx.size(); ic++)
			{
				int smcjoint = fit_data_.adam.h36m_jointConst_smcIdx[ic];
				if (smcjoint != 3 && smcjoint != 6 && smcjoint != 9 && smcjoint != 12)
				{
					m_targetPts_weight[ic] = double(limb) * m_targetPts_weight_buffer[ic];
				}
			}
			// no finger joints for Human3.6M
		}
		else
		{
			assert(regressor_type == 2);
			for (auto ic = 0u; ic < fit_data_.adam.h36m_jointConst_smcIdx.size(); ic++)
			{
				int smcjoint = fit_data_.adam.h36m_jointConst_smcIdx[ic];
				if (smcjoint != 3 && smcjoint != 6 && smcjoint != 9 && smcjoint != 12)
				{
					m_targetPts_weight[ic] = double(limb) * m_targetPts_weight_buffer[ic];
				}
			}
			int offset = fit_data_.adam.h36m_jointConst_smcIdx.size();
			for (int ic = 0; ic < fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows(); ic++)
			{
				if (ic % 5 == 0)
					m_targetPts_weight[ic + offset] = m_targetPts_weight_buffer[ic + offset] * double(palm);
				else
					m_targetPts_weight[ic + offset] = m_targetPts_weight_buffer[ic + offset] * double(finger);
			}
			offset += fit_data_.adam.m_correspond_adam2lHand_adamIdx.rows();
			for (int ic = 0; ic < fit_data_.adam.m_correspond_adam2rHand_adamIdx.rows(); ic++)
			{
				if (ic % 5 == 0)
					m_targetPts_weight[ic + offset] = m_targetPts_weight_buffer[ic + offset] * double(palm);
				else
					m_targetPts_weight[ic + offset] = m_targetPts_weight_buffer[ic + offset] * double(finger);
			}
		}
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
	    double* dVdc_data,     //output
	    const double* face_coeff,
	    double* dVdfc_data
	) const;

	void select_lbs(
	    const double* c,
	    const Eigen::VectorXd& T,  // transformation
	    MatrixXdr &outVert,
	    const double* face_coeff
	) const;

	void SparseRegress(const Eigen::SparseMatrix<double>& reg, const double* V_data, const double* dVdP_data, const double* dVdc_data,
					   double* J_data, double* dJdP_data, double* dJdc_data) const;

	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

	Eigen::VectorXd m_targetPts_weight;
	Eigen::VectorXd m_targetPts_weight_buffer;
	std::vector<float> PAF_weight;
	int m_nCorrespond_adam2joints;
	int m_nCorrespond_adam2pts;
	int m_nResiduals;
	int res_dim;  // number of residuals per joint / vertex constraints
	bool freeze_missing;

private:
	// input data
	const AdamFitData& fit_data_;
	const int regressor_type;
	// setting
	bool rigid_body;
	// data for joint / projection fitting
	Eigen::VectorXd m_targetPts;
	int start_2d_dim;
	// data for vertex fitting
	std::vector<int> total_vertex; // all vertices that needs to be computed
	std::vector< std::pair<int, std::vector<double>> > corres_vertex2targetpt; // vertices used as constraints
	// counter
	// data for PAF fitting
	const int num_PAF_constraint;
	std::vector<std::array<uint, 4>> PAF_connection;

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
	double* dOdP_data;
	double* dOdc_data;
	double* dVdfc_data;
	double* dOdfc_data;

	std::map<int, int> map_regressor_to_constraint;
	const bool fit_face_exp;
};

class AdamFaceCost: public ceres::CostFunction
{
public:
	AdamFaceCost(const TotalModel& adam, const smpl::SMPLParams& frame_param, const Eigen::MatrixXd& faceJoints):
		adam_(adam), frame_param_(frame_param), faceJoints_(faceJoints), m_nResiduals(3 * adam_.m_correspond_adam2face70_face70Idx.size()),
			meanVert(adam_.m_correspond_adam2face70_face70Idx.size(), 3), transforms_joint(TotalModel::NUM_JOINTS * 3 * 5), dVdfc(adam_.m_correspond_adam2face70_face70Idx.size() * 3, TotalModel::NUM_EXP_BASIS_COEFFICIENTS)
	{
		CostFunction::set_num_residuals(m_nResiduals);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(TotalModel::NUM_EXP_BASIS_COEFFICIENTS);

		// setup parent indexes, for fast LBS jacobian computation
		parentIndexes[0].clear();
		parentIndexes[0].push_back(0);
		for(auto i = 0u; i < TotalModel::NUM_JOINTS; i++)
		{
            parentIndexes[i] = std::vector<int>(1, i);
            while (parentIndexes[i].back() != 0)
                parentIndexes[i].emplace_back(adam_.m_parent[parentIndexes[i].back()]);
            std::sort(parentIndexes[i].begin(), parentIndexes[i].end());
        }

		// forward kinematics
		using namespace Eigen;
	    Matrix<double, TotalModel::NUM_JOINTS, 3, RowMajor> J;
	    Map< Matrix<double, Dynamic, 1> > J_vec(J.data(), TotalModel::NUM_JOINTS * 3);
	    J_vec = adam.J_mu_ + adam.dJdc_ * frame_param_.m_adam_coeffs;

	    const double* p2t_parameters[2] = { frame_param_.m_adam_pose.data(), J.data() };
	    double* p2t_residuals = transforms_joint.data();

	    smpl::PoseToTransform_AdamFull_withDiff p2t(adam_, parentIndexes, false);
	    p2t.Evaluate(p2t_parameters, p2t_residuals, nullptr );

	    // LBS for the face vertices
	    total_vertex.clear();
		for (int r = 0; r < adam_.m_correspond_adam2face70_adamIdx.rows(); ++r)
		{
			const int adamVertexIdx = adam_.m_correspond_adam2face70_adamIdx(r);
			total_vertex.push_back(adamVertexIdx);
		}

	    select_lbs(frame_param_.m_adam_coeffs.data(), transforms_joint, meanVert);

        std::fill(dVdfc.data(),
                  dVdfc.data() + 3 * total_vertex.size() * TotalModel::NUM_EXP_BASIS_COEFFICIENTS,
                  0.0);
	    const double* face_basis_data = adam_.m_dVdFaceEx.data();
        const int nrow = adam_.m_dVdFaceEx.rows();
        const int ncolc = TotalModel::NUM_EXP_BASIS_COEFFICIENTS;
	    for (auto i = 0u; i < total_vertex.size(); ++i)  // precompute the jacobian w.r.t face coefficients
	    {
	        const int idv = total_vertex[i];
		    for (int idj = 0; idj < TotalModel::NUM_JOINTS; idj++)
		    {	    	
	            const double w = adam_.m_blendW(idv, idj);
	            if (w)
	            {
		            const auto* const Trow_data = transforms_joint.data() + 12 * idj;
		            double* dVdfc_row0 = dVdfc.data() + (i * 3 + 0) * ncolc;
		            double* dVdfc_row1 = dVdfc.data() + (i * 3 + 1) * ncolc;
		            double* dVdfc_row2 = dVdfc.data() + (i * 3 + 2) * ncolc;
		            for (int idc = 0; idc < TotalModel::NUM_EXP_BASIS_COEFFICIENTS; idc++) {
		                dVdfc_row0[idc] += w * (face_basis_data[idc * nrow + idv * 3 + 0] * Trow_data[0 * 4 + 0] + face_basis_data[idc * nrow + idv * 3 + 1] * Trow_data[0 * 4 + 1] + face_basis_data[idc * nrow + idv * 3 + 2] * Trow_data[0 * 4 + 2]);
		                dVdfc_row1[idc] += w * (face_basis_data[idc * nrow + idv * 3 + 0] * Trow_data[1 * 4 + 0] + face_basis_data[idc * nrow + idv * 3 + 1] * Trow_data[1 * 4 + 1] + face_basis_data[idc * nrow + idv * 3 + 2] * Trow_data[1 * 4 + 2]);
		                dVdfc_row2[idc] += w * (face_basis_data[idc * nrow + idv * 3 + 0] * Trow_data[2 * 4 + 0] + face_basis_data[idc * nrow + idv * 3 + 1] * Trow_data[2 * 4 + 1] + face_basis_data[idc * nrow + idv * 3 + 2] * Trow_data[2 * 4 + 2]);
		            }
		        }
	        }
	    }
	}

	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

	void select_lbs(
	    const double* c,
	    const Eigen::VectorXd& T,  // transformation
	    MatrixXdr &outVert
	) const;

private:
	const TotalModel& adam_;
	const smpl::SMPLParams& frame_param_;
	const Eigen::MatrixXd& faceJoints_;
	const int m_nResiduals;
	std::array<std::vector<int>, TotalModel::NUM_JOINTS> parentIndexes;
	std::vector<int> total_vertex; // all vertices that needs to be computed
	MatrixXdr meanVert;  // mean vertex when face coeff is 0
    Eigen::VectorXd transforms_joint;
    MatrixXdr dVdfc;
};