#include "CMeshModelInstance.h"
#include "simple.h"
#include <Eigen/Sparse>

#ifndef TOTALMODEL_H
#define TOTALMODEL_H

struct TotalModelCorresUnit
{
	TotalModelCorresUnit()
	{
		m_sourceMeshType = CMeshModelInstance::MESH_TYPE_UNKNOWN;
	};

	TotalModelCorresUnit(CMeshModelInstance::EnumMeshType type)
	{
		m_sourceMeshType = type;
	};

	CMeshModelInstance::EnumMeshType m_sourceMeshType;// = CMeshModelInstance::MESH_TYPE_SMPL;
	std::vector< std::pair<int, double> > m_corresWeight;		//barycentric, <correspondence, weight>
};

struct TotalModelGlueCorresUnit
{
	TotalModelGlueCorresUnit()
	{
	};


	std::vector< std::pair<int, double> > m_sourceVertices_fromSMPL;		//barycentric, <SMPL_vertex_idx, weight>
	int m_targetVertexIdx;		//Vertices in FW, LHand, or RHand
};
  
struct TotalModel 
{
	static const int NUM_VERTICES = 18540;
	static const int NUM_FACES = 36946;// 111462;

	static const int NUM_JOINTS = 62;		//(SMPL-hands)22 + lhand20 + rhand20
	static const int NUM_POSE_PARAMETERS = NUM_JOINTS * 3;
	static const int NUM_SHAPE_COEFFICIENTS = 30;
	static const int NUM_EXP_BASIS_COEFFICIENTS = 200; //Facial expression

	//Mesh Model
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> m_vertices;
	Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> m_faces;		//has zero-based indexing

	Eigen::Matrix<double, Eigen::Dynamic, 2> m_uvs;		//uv map
	Eigen::Matrix<double, Eigen::Dynamic, 3> m_normals;		//uv map

	std::vector<TotalModelCorresUnit> m_vertexCorresSources;
	Eigen::SparseMatrix<double> m_C_face2total; //3*TotalModel::NUM_VERTICES x 3*FaceModel::NUM_VERTICES
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>	m_dVdFaceEx;		//precompute: m_C_face2total*  g_face_model.U_exp_

	std::vector<bool> m_indicator_noDeltaDefom;	//size should be same as vertice size
	std::vector<double> m_vertex_laplacianWeight;	//size should be same as vertice size

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m_blendW;
	Eigen::SparseMatrix<double> m_cocoplus_reg;
	Eigen::SparseMatrix<double> m_small_coco_reg;  // regressor used by Donglai to implement fast COCO keypoint fitting (SIGGAsia)

	Eigen::Matrix<int, 2, Eigen::Dynamic> m_kintree_table;
	int m_parent[NUM_JOINTS];
	int m_id_to_col[NUM_JOINTS];

	//Adam to vertex correspondences
	Eigen::Matrix<int, Eigen::Dynamic, 1>  m_correspond_adam2face70_face70Idx;		//0based indexing
	Eigen::Matrix<int, Eigen::Dynamic, 1>  m_correspond_adam2face70_adamIdx;		//0based indexing
	Eigen::Matrix<int, Eigen::Dynamic, 1>  m_correspond_adam2cocoear_cocobodyIdx; //0based indexing
	Eigen::Matrix<int, Eigen::Dynamic, 1>  m_correspond_adam2cocoear_adamIdx; //0based indexing

																			//For SMC joints
	Eigen::Matrix<int, Eigen::Dynamic, 1> m_indices_jointConst_adamIdx;	//correspondence between smpl and smc (not all joints are included)
	Eigen::Matrix<int, Eigen::Dynamic, 1> m_indices_jointConst_smcIdx;


	Eigen::Matrix<int, Eigen::Dynamic, 1> m_correspond_adam2lHand_adamIdx;	
	Eigen::Matrix<int, Eigen::Dynamic, 1> m_correspond_adam2lHand_lHandIdx;

	Eigen::Matrix<int, Eigen::Dynamic, 1> m_correspond_adam2rHand_adamIdx;
	Eigen::Matrix<int, Eigen::Dynamic, 1> m_correspond_adam2rHand_rHandIdx;


	//Glue constraints
	std::vector<TotalModelGlueCorresUnit> m_glueConstraint_face;
	std::vector<TotalModelGlueCorresUnit> m_glueConstraint_lhand;
	std::vector<TotalModelGlueCorresUnit> m_glueConstraint_rhand;

	//Total Model PCA space
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m_shapespace_u;
	Eigen::Matrix<double, Eigen::Dynamic, 1> m_shapespace_Ds;
	Eigen::Matrix<double, Eigen::Dynamic, 1> m_meanshape;


	//For Adam
	Eigen::SparseMatrix<double, Eigen::RowMajor> m_J_reg;
	Eigen::SparseMatrix<double, Eigen::RowMajor> m_J_reg_smc;

	Eigen::Matrix<double, NUM_JOINTS * 3, 1> J_mu_;
	Eigen::Matrix<double, Eigen::Dynamic, NUM_SHAPE_COEFFICIENTS, Eigen::RowMajor> dJdc_;
	Eigen::VectorXd face_prior_A_exp;

	const std::array<int, 19> h36m_jointConst_smcIdx{{ 14, 13, 12, 6, 7, 8, 11, 10, 9, 3, 4, 5, 0, 19, 1, 15, 17, 16, 18 }};

	bool m_bInit;

	TotalModel() {
		m_bInit = false;
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

void LoadTotalDataFromJson(TotalModel &totalm, const std::string &path, const std::string &pca_path, const std::string &correspondence_path);
void LoadTotalModelFromObj(TotalModel &totalm, const std::string &path);
void LoadCocoplusRegressor(TotalModel &totalm, const std::string &path);
void adam_reconstruct_Eulers(const TotalModel& smpl,
	const double *parm_coeffs,
	const double *parm_pose_eulers,
	const double *parm_faceEx_coeffs,
	double *outVerts,
	Eigen::VectorXd &transforms);
void adam_reconstruct_Eulers_Fast(const TotalModel& smpl,
	const Eigen::Matrix<double, Eigen::Dynamic, 1> &Vt,
	const Eigen::Matrix<double, Eigen::Dynamic, 1> &J0,
	const double *parm_pose_eulers,
	const double *parm_faceEx_coeffs,
	double *outVerts,
	Eigen::VectorXd &transforms);
Eigen::VectorXd adam_reconstruct_withDerivative_eulers(const TotalModel &mod,
	const double *coeffs,
	const double *pose,		//input: euler angles
	const double *parm_faceEx_coeffs,
	double *outVerts,		//output
	MatrixXdr &dVdc,		//output
	MatrixXdr &dVdP,
	MatrixXdr &dVdfc,
	MatrixXdr &dTJdc,
	MatrixXdr &dTJdP,
	bool joint_only,
	bool fit_face);		//output
void adam_lbs(const TotalModel &totalm,
	const double *verts,
	const MatrixXdr& T,
	double *outVerts);
void adam_lbs(const TotalModel &totalm,
	const double *verts,
	const MatrixXdr& T,
	double *outVerts,		//output
	const MatrixXdr &dVsdc,
	const MatrixXdr &dTdP,
	const MatrixXdr &dTdc,
	MatrixXdr &dVdc,	//output
	MatrixXdr &dVdP);	//output

#endif