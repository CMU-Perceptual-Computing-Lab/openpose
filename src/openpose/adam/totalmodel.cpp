#include "totalmodel.h"
#include "stdio.h"
#include "handm.h"
#include <json/json.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <iostream>
#include "ceres/ceres.h"
#include "pose_to_transforms.h"
#include <chrono>
#include <omp.h>

const int TotalModel::NUM_SHAPE_COEFFICIENTS;
const int TotalModel::NUM_VERTICES;
const int TotalModel::NUM_JOINTS;
const int TotalModel::NUM_POSE_PARAMETERS;
const int TotalModel::NUM_FACES;
const int TotalModel::NUM_EXP_BASIS_COEFFICIENTS;

template<typename Derived, int rows, int cols>
void initMatrix(Eigen::Matrix<Derived, rows, cols>& m, const Json::Value& value)
{
    if(m.cols() == 1) { // a vector
        m.resize(value.size(), 1);
        auto* m_data = m.data();
        if (strcmp(typeid(Derived).name(), "i") == 0) // the passed in matrix is Int
		    for (uint i = 0; i < value.size(); i++)
	            m_data[i] = value[i].asInt();
        else // the passed in matrix should be double
		    for (uint i = 0; i < value.size(); i++)
                m_data[i] = value[i].asDouble();
    }
    else  { // a matrix (column major)
    	const int nrow = value.size(), ncol = value[0u].size();
        m.resize(nrow, ncol);
        auto* m_data = m.data();
        if (strcmp(typeid(Derived).name(), "i") == 0)
	        for (uint i = 0; i < value.size(); i++)
	            for (uint j = 0; j < value[i].size(); j++)
	            	m_data[j * nrow + i] = value[i][j].asInt();
        else
	        for (uint i = 0; i < value.size(); i++)
	            for (uint j = 0; j < value[i].size(); j++)
                    m_data[j * nrow + i] = value[i][j].asDouble();
    }
    std::cout << "rows " << m.rows() << " cols " << m.cols() << std::endl;
}

template<typename Derived, int rows, int cols, int option>
void initRowMajorMatrix(Eigen::Matrix<Derived, rows, cols, option>& m, const Json::Value& value)
{
    if(m.cols() == 1) { // a vector
        m.resize(value.size(), 1);
        auto* m_data = m.data();
        if (strcmp(typeid(Derived).name(), "i") == 0) // the passed in matrix is Int
		    for (uint i = 0; i < value.size(); i++)
	            m_data[i] = value[i].asInt();
        else // the passed in matrix should be double
		    for (uint i = 0; i < value.size(); i++)
                m_data[i] = value[i].asDouble();
    }
    else  { // a matrix (column major)
    	const int nrow = value.size(), ncol = value[0u].size();
        m.resize(nrow, ncol);
        auto* m_data = m.data();
        if (strcmp(typeid(Derived).name(), "i") == 0)
	        for (uint i = 0; i < value.size(); i++)
	            for (uint j = 0; j < value[i].size(); j++)
	            	m_data[i * ncol + j] = value[i][j].asInt();
        else
	        for (uint i = 0; i < value.size(); i++)
	            for (uint j = 0; j < value[i].size(); j++)
                    m_data[i * ncol + j] = value[i][j].asDouble();
    }
    std::cout << "rows " << m.rows() << " cols " << m.cols() << std::endl;
}

template<typename Derived, int option>
void initSparseMatrix(Eigen::SparseMatrix<Derived, option>& m, const Json::Value& value)
{
	if (strcmp(typeid(Derived).name(), "i") == 0)
	{
		// The first row specifies the size of the sparse matrix
		m.resize(value[0u][0u].asInt(), value[0u][1u].asInt());
		for (uint k = 1u; k < value.size(); k++)
		{
			assert(value[k].size() == 3);
			// From the second row on, triplet correspond to matrix entries
			const int i = value[k][0u].asInt();
			const int j = value[k][1u].asInt();
			m.insert(i, j) = value[k][2u].asInt();
		}
	}
	else
	{
		// The first row specifies the size of the sparse matrix
		m.resize(value[0u][0u].asInt(), value[0u][1u].asInt());
		for (uint k = 1u; k < value.size(); k++)
		{
			assert(value[k].size() == 3);
			// From the second row on, triplet correspond to matrix entries
			const int i = value[k][0u].asInt();
			const int j = value[k][1u].asInt();
			m.insert(i, j) = value[k][2u].asDouble();
		}
	}
	std::cout << "rows " << m.rows() << " cols " << m.cols() << std::endl;
}

void LoadTotalDataFromJson(TotalModel &totalm, const std::string &path, const std::string &pca_path, const std::string &correspondence_path)
{
	printf("Loading from: %s\n", path.c_str());
	Json::Value root, pca_root;
	std::ifstream file;
	file.open(path.c_str(), std::ifstream::in);
    file >> root;

    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> stmp;
    initSparseMatrix(totalm.m_J_reg, root["adam_J_regressor_big"]);
    initSparseMatrix(totalm.m_J_reg_smc, root["adam_J_regressor_big_smc"]);
    initSparseMatrix(totalm.m_small_coco_reg, root["small_coco_reg"]);
    totalm.m_small_coco_reg.makeCompressed();

    initMatrix(totalm.m_blendW, root["blendW"]);
    initMatrix(totalm.m_kintree_table, root["kintree_table"]);
	for (int idt = 0; idt < totalm.m_kintree_table.cols(); idt++) {
		totalm.m_id_to_col[totalm.m_kintree_table(1, idt)] = idt;
	}
	for (int idt = 1; idt < totalm.m_kintree_table.cols(); idt++) {
		totalm.m_parent[idt] = totalm.m_id_to_col[totalm.m_kintree_table(0, idt)];
	}

	initMatrix(totalm.m_correspond_adam2face70_face70Idx, root["correspond_adam2face70_face70Idx"]);
	initMatrix(totalm.m_correspond_adam2face70_adamIdx, root["correspond_adam2face70_adamIdx"]);
	initMatrix(totalm.m_correspond_adam2cocoear_cocobodyIdx, root["correspond_adam2cocoear_cocobodyIdx"]);
	initMatrix(totalm.m_correspond_adam2cocoear_adamIdx, root["correspond_adam2cocoear_adamIdx"]);
	initMatrix(totalm.m_correspond_adam2lHand_adamIdx, root["correspond_adam2lHand_adamIdx"]);
	initMatrix(totalm.m_correspond_adam2lHand_lHandIdx, root["correspond_adam2lHand_lHandIdx"]);
	initMatrix(totalm.m_correspond_adam2rHand_adamIdx, root["correspond_adam2rHand_adamIdx"]);
	initMatrix(totalm.m_correspond_adam2rHand_rHandIdx, root["correspond_adam2rHand_rHandIdx"]);

	totalm.m_indices_jointConst_adamIdx = Eigen::Matrix<int, Eigen::Dynamic, 1>(13, 1);
	totalm.m_indices_jointConst_smcIdx = Eigen::Matrix<int, Eigen::Dynamic, 1>(13, 1);
	totalm.m_indices_jointConst_adamIdx << 16, 18, 20, 1, 4, 7, 17, 19, 21, 2, 5, 8, 12;		
	totalm.m_indices_jointConst_smcIdx << 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0;		
	totalm.face_prior_A_exp.resize(TotalModel::NUM_EXP_BASIS_COEFFICIENTS);
	totalm.face_prior_A_exp << 1.0000,    1.3884,    1.4322,    3.2194,    4.0273,    4.5258,    4.9396,    5.1066,    5.6521,    5.8661,    6.6474,    7.5848,    7.8018,    7.9318,    8.4214,    9.0600,    9.3434,   10.5993,   10.6766,   11.0619,   11.3347,   12.8620,   14.4040,   15.4403,   15.9183,   16.8945,   17.1970,   18.0435,   18.8597,   19.4450,   19.8396,   20.3699,   20.6630,   21.8482,   23.2284,   23.5336,   24.1947,   25.7601,   26.5978,   27.8819,   29.3783,   29.6195,   30.9762,   31.9264,   32.8898,   34.0769,   34.6534,   35.5318,   37.0082,   38.3323,   38.7301,   39.6270,   43.3004,   45.4749,   47.4281,   49.3030,   49.9038,   51.4549,   52.1341,   53.1723,   53.6358,   54.2716,   55.6179,   56.4990,   57.4234,   58.0243,   59.3404,   60.8487,   62.4063,   62.6375,   64.4185,   65.4798,   66.1404,   67.0013,   67.4438,   68.8301,   70.4146,   70.9421,   72.7690,   74.5522,   76.4981,   77.6108,   79.6063,   79.9627,   81.4692,   82.0128,   82.2424,   84.3532,   86.8927,   87.9610,   88.0665,   89.4892,   90.5118,   90.9908,   92.7930,   94.3903,   95.7852,   96.7678,   97.3032,   99.8454,  100.4756,  101.5921,  102.1015,  103.8437,  105.1683,  105.4616,  107.7326,  109.0088,  109.1023,  111.6924,  113.5024,  115.0624,  116.4892,  117.5802,  119.9704,  120.8436,  122.6132,  123.3027,  125.0063,  125.4102,  126.2347,  126.3537,  129.1989,  129.5823,  130.2158,  131.5268,  133.4373,  135.1778,  137.9440,  140.7205,  141.5721,  142.4590,  143.7006,  144.8883,  146.3983,  147.1215,  147.8103,  149.5836,  150.6851,  150.6961,  153.1289,  154.3134,  155.3207,  156.5810,  157.8858,  158.8535,  160.9939,  161.8322,  163.7226,  165.7372,  166.1939,  168.1920,  168.5944,  168.9809,  170.9044,  172.3045,  172.7526,  176.3592,  177.1201,  177.2328,  179.7210,  180.7331,  180.9280,  183.4400,  183.7548,  184.5876,  185.7965,  187.5969,  189.1555,  190.4965,  192.2034,  193.3294,  195.0099,  198.5997,  200.0230,  200.7327,  202.6141,  205.1104,  206.0522,  207.0078,  207.6073,  209.0094,  211.3078,  212.3590,  214.3154,  215.2212,  215.9995,  218.3660,  220.2393,  221.0927,  222.2768,  224.2556,  225.6657,  228.3417,  228.9960,  232.4002,  232.4943,  234.7318,  235.8495,  236.6455;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U_exp_;
	initRowMajorMatrix(U_exp_, root["U_exp"]);

	file.close();

	// load PCA
	printf("Loading from: %s\n", pca_path.c_str());
	file.open(pca_path.c_str(), std::ifstream::in);
	file >> pca_root;

	initMatrix(totalm.m_meanshape, pca_root["mu"]);
	initMatrix(totalm.m_shapespace_Ds, pca_root["Ds"]);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> temp;
	initMatrix(temp, pca_root["Uw1"]); // Uw1 is reduced to (:, :30)
	totalm.m_shapespace_u = temp.block(0, 0, 3*TotalModel::NUM_VERTICES, TotalModel::NUM_SHAPE_COEFFICIENTS);

	totalm.J_mu_ = totalm.m_J_reg * totalm.m_meanshape;
	totalm.dJdc_ = totalm.m_J_reg * totalm.m_shapespace_u;

	// load correspondence
	printf("Loading from: %s\n", correspondence_path.c_str());
	totalm.m_vertexCorresSources.clear();
	std::ifstream fin(correspondence_path.c_str());
	if (fin.is_open() == false)
	{
		printf("## Error: Failed in opening total correspondence file: %s\n", correspondence_path.c_str());
		return;
	}
	while (fin.eof() == false)
	{
		int target_vertex_index;	//index of vertex in total mesh (these are all in order)
		int source_mesh;		//body=0, face=1, rhand=2, lhand=3
		int source_v[3];	// vertex indices of corresponding triangle in the mesh given by "source_mesh"
		double source_w[3]; // barycentric weights for each of the 3 vertices
		fin >> target_vertex_index >> source_mesh >> source_v[0] >> source_v[1] >> source_v[2];			//indices are 0-based. 
		fin >> source_w[0] >> source_w[1] >> source_w[2];			//weight are zeros

		if (fin.eof())
			break;

		if(source_mesh ==0 )
			totalm.m_vertexCorresSources.push_back(TotalModelCorresUnit(CMeshModelInstance::MESH_TYPE_SMPL));
		else if (source_mesh == 1)
			totalm.m_vertexCorresSources.push_back(TotalModelCorresUnit(CMeshModelInstance::MESH_TYPE_FACE));
		else if (source_mesh == 2)
			totalm.m_vertexCorresSources.push_back(TotalModelCorresUnit(CMeshModelInstance::MESH_TYPE_RHAND));
		else if (source_mesh == 3)
			totalm.m_vertexCorresSources.push_back(TotalModelCorresUnit(CMeshModelInstance::MESH_TYPE_LHAND));
		else
		{
			printf("## ERROR:: Unknown type: %d\n", source_mesh);
			fin.close();
			return;
		}

		for(int i=0;i<3;++i)
		{
			if (source_w[i] >= 0.001)
			{
				totalm.m_vertexCorresSources.back().m_corresWeight.push_back(std::make_pair(source_v[i], source_w[i]));
			}
		}
	}

	const int NUM_FACE_POINTS = 11510;
	totalm.m_C_face2total.resize(TotalModel::NUM_VERTICES*3, NUM_FACE_POINTS * 3);
	totalm.m_C_face2total.setZero();

	std::vector<Eigen::Triplet<double> > A_IJV;
	A_IJV.reserve(NUM_FACE_POINTS * 3);

	for (auto v = 0u; v < totalm.m_vertexCorresSources.size(); ++v)
	{
		if (totalm.m_vertexCorresSources[v].m_sourceMeshType != CMeshModelInstance::MESH_TYPE_FACE)
			continue;

		for (auto j = 0u; j < totalm.m_vertexCorresSources[v].m_corresWeight.size(); ++j)
		{
			int sourceV_idx = totalm.m_vertexCorresSources[v].m_corresWeight[j].first;
			double sourceV_weight = totalm.m_vertexCorresSources[v].m_corresWeight[j].second;
			int totalModelV_idx = v;

			A_IJV.push_back(Eigen::Triplet<double>(3 * totalModelV_idx +0, 3 * sourceV_idx +0, sourceV_weight));
			A_IJV.push_back(Eigen::Triplet<double>(3 * totalModelV_idx +1, 3 * sourceV_idx +1, sourceV_weight));
			A_IJV.push_back(Eigen::Triplet<double>(3 * totalModelV_idx +2, 3 * sourceV_idx +2, sourceV_weight));
		}
		//totalMeshOut.m_vertices[v] = totalMeshOut.m_vertices[v] + pos * 10;
	}
	totalm.m_C_face2total.setFromTriplets(A_IJV.begin(), A_IJV.end());
	totalm.m_C_face2total = totalm.m_C_face2total* 100.0; //100.0 to adjust scales

	totalm.m_dVdFaceEx = totalm.m_C_face2total * U_exp_;

	printf("## Total Body model has been loaded\n");
}

void adam_reconstruct_Eulers(const TotalModel& totalm,
	const double *parm_coeffs,
	const double *parm_pose_eulers,
	const double *parm_faceEx_coeffs,
	double *outVerts,
	Eigen::VectorXd &transforms)
{
	using namespace smpl;
	using namespace Eigen;
	Map< const Matrix<double, Dynamic, 1> > c(parm_coeffs, TotalModel::NUM_SHAPE_COEFFICIENTS);

	Matrix<double, Dynamic, Dynamic, RowMajor> Vt(TotalModel::NUM_VERTICES, 3);
	Map< Matrix<double, Dynamic, 1> > Vt_vec(Vt.data(), 3 * TotalModel::NUM_VERTICES);

	Map< Matrix<double, Dynamic, Dynamic, RowMajor> > outV(outVerts, TotalModel::NUM_VERTICES, 3);

	Vt_vec = totalm.m_meanshape + totalm.m_shapespace_u*c;

	Matrix<double, TotalModel::NUM_JOINTS, 3, RowMajor> J;
	Map< Matrix<double, Dynamic, 1> > J_vec(J.data(), TotalModel::NUM_JOINTS * 3);
	J_vec = totalm.m_J_reg * Vt_vec;

	transforms.resize(3 * TotalModel::NUM_JOINTS * 4);
	VectorXd transforms_joint(3 * TotalModel::NUM_JOINTS * 4 + 3 * TotalModel::NUM_JOINTS); // the first part is transform, the second part is outJoint

	const int num_t = (TotalModel::NUM_JOINTS) * 3 * 4;
	Matrix<double, Dynamic, 3 * TotalModel::NUM_JOINTS, RowMajor> dTrdP((TotalModel::NUM_JOINTS) * 3 * 5, 3 * TotalModel::NUM_JOINTS);
	Matrix<double, Dynamic, 3 * TotalModel::NUM_JOINTS, RowMajor> dTrdJ((TotalModel::NUM_JOINTS) * 3 * 5, 3 * TotalModel::NUM_JOINTS);

	ceres::AutoDiffCostFunction<PoseToTransformsNoLR_Eulers_adamModel,
		(TotalModel::NUM_JOINTS) * 3 * 5,
		(TotalModel::NUM_JOINTS) * 3,
		(TotalModel::NUM_JOINTS) * 3> p2t(new PoseToTransformsNoLR_Eulers_adamModel(totalm));
	const double * parameters[2] = { parm_pose_eulers, J.data() };
	double * residuals = transforms_joint.data();
	p2t.Evaluate(parameters, residuals, nullptr);		//automatically compute residuals and jacobians (dTdP and dTdJ)

	transforms.block(0, 0, num_t, 1) = transforms_joint.block(0, 0, num_t, 1);

	Map< const Matrix<double, Dynamic, 1> > c_faceEx(parm_faceEx_coeffs, TotalModel::NUM_EXP_BASIS_COEFFICIENTS);
	Vt_vec = Vt_vec + totalm.m_dVdFaceEx * c_faceEx;		// m_C_face2total*facem.U_exp_
	
	adam_lbs(totalm, Vt_vec.data(), transforms, outVerts);
}

void adam_lbs(const TotalModel &totalm,
    const double *verts,
    const MatrixXdr& T,
    double *outVerts)
{
    const Eigen::Map< const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >
        Vs(verts, TotalModel::NUM_VERTICES, 3);
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >
        outV(outVerts, TotalModel::NUM_VERTICES, 3);

    const Eigen::Map<const Eigen::VectorXd> Tv(T.data(), T.rows()*T.cols());
    const auto* T_data = T.data();
    #pragma omp parallel
    {
        #pragma omp for
        for (int idv = 0; idv < TotalModel::NUM_VERTICES; idv++)
        {
            auto* outVrow_data = &outVerts[3*idv];
            outVrow_data[0] = 0; // outV(idv, 0)
            outVrow_data[1] = 0; // outV(idv, 1)
            outVrow_data[2] = 0; // outV(idv, 2)
            const auto* const Vsrow_data = &verts[3*idv];
            for (int idj = 0; idj < TotalModel::NUM_JOINTS; idj++)
            {
                const double w = totalm.m_blendW(idv, idj);
                if (w)
                {
                    const auto baseIndex = idj * 3 * 4;
                    const auto* const Trow_data = &T_data[baseIndex];
                    outVrow_data[0] += w*(Vsrow_data[0]*Trow_data[0]
                                          + Vsrow_data[1]*Trow_data[1]
                                          + Vsrow_data[2]*Trow_data[2]
                                          + Trow_data[3]);
                    outVrow_data[1] += w*(Vsrow_data[0]*Trow_data[4]
                                          + Vsrow_data[1]*Trow_data[5]
                                          + Vsrow_data[2]*Trow_data[6]
                                          + Trow_data[7]);
                    outVrow_data[2] += w*(Vsrow_data[0]*Trow_data[8]
                                          + Vsrow_data[1]*Trow_data[9]
                                          + Vsrow_data[2]*Trow_data[10]
                                          + Trow_data[11]);
                    // Original code
                    // for (int idd = 0; idd < 3; idd++)
                    // {
                    //     outV(idv, idd) += w*Vs(idv, 0)*Tv(idj * 3 * 4 + idd * 4 + 0);
                    //     outV(idv, idd) += w*Vs(idv, 1)*Tv(idj * 3 * 4 + idd * 4 + 1);
                    //     outV(idv, idd) += w*Vs(idv, 2)*Tv(idj * 3 * 4 + idd * 4 + 2);
                    //     outV(idv, idd) += w*Tv(idj * 3 * 4 + idd * 4 + 3);
                    // }
                }
            }
        }
    }
}

void adam_lbs(const TotalModel &smpl,
	const double *verts,
	const MatrixXdr& T,
	double *outVerts,		//output
	const MatrixXdr &dVsdc,
	const MatrixXdr &dTdP,
	const MatrixXdr &dTdc,
	MatrixXdr &dVdc,	//output
	MatrixXdr &dVdP)	//output
{
	using namespace Eigen;
	Map< const Matrix<double, Dynamic, Dynamic, RowMajor> >
		Vs(verts, TotalModel::NUM_VERTICES, 3);
	Map< Matrix<double, Dynamic, Dynamic, RowMajor> >
		outV(outVerts, TotalModel::NUM_VERTICES, 3);

	dVdP.resize(TotalModel::NUM_VERTICES * 3, TotalModel::NUM_JOINTS * 3);
	dVdc.resize(TotalModel::NUM_VERTICES * 3, TotalModel::NUM_SHAPE_COEFFICIENTS);
	dVdP.setZero();
	dVdc.setZero();

	Map< const VectorXd > Tv(T.data(), T.rows()*T.cols());
	#pragma omp parallel num_threads(12)
	{
		#pragma omp for
		for (int idv = 0; idv<TotalModel::NUM_VERTICES; idv++) {
			outV(idv, 0) = 0;
			outV(idv, 1) = 0;
			outV(idv, 2) = 0;
			for (int idj = 0; idj<TotalModel::NUM_JOINTS; idj++) {
				if (smpl.m_blendW(idv, idj)) {
					double w = smpl.m_blendW(idv, idj);
					for (int idd = 0; idd<3; idd++) {
						outV(idv, idd) += w*Vs(idv, 0)*Tv(idj * 3 * 4 + idd * 4 + 0);
						outV(idv, idd) += w*Vs(idv, 1)*Tv(idj * 3 * 4 + idd * 4 + 1);
						outV(idv, idd) += w*Vs(idv, 2)*Tv(idj * 3 * 4 + idd * 4 + 2);
						outV(idv, idd) += w*Tv(idj * 3 * 4 + idd * 4 + 3);

						// SMPLModel::NUM_JOINTS
						// The joint transforms only depend on their parents, not vice-versa.
						// (meaning dTdP is block lower-triangular).
						for (int idp = 0; idp<(idj + 1) * 3; idp++) {
							dVdP(idv * 3 + idd, idp) += w*Vs(idv, 0)*dTdP(idj * 3 * 4 + idd * 4 + 0, idp);
							dVdP(idv * 3 + idd, idp) += w*Vs(idv, 1)*dTdP(idj * 3 * 4 + idd * 4 + 1, idp);
							dVdP(idv * 3 + idd, idp) += w*Vs(idv, 2)*dTdP(idj * 3 * 4 + idd * 4 + 2, idp);
							dVdP(idv * 3 + idd, idp) += w*dTdP(idj * 3 * 4 + idd * 4 + 3, idp);
						}
					
						for (int idc = 0; idc<TotalModel::NUM_SHAPE_COEFFICIENTS; idc++) {
							dVdc(idv * 3 + idd, idc) += w*dVsdc(idv * 3 + 0, idc)*Tv(idj * 3 * 4 + idd * 4 + 0);
							dVdc(idv * 3 + idd, idc) += w*dVsdc(idv * 3 + 1, idc)*Tv(idj * 3 * 4 + idd * 4 + 1);
							dVdc(idv * 3 + idd, idc) += w*dVsdc(idv * 3 + 2, idc)*Tv(idj * 3 * 4 + idd * 4 + 2);
							dVdc(idv * 3 + idd, idc) += w*dTdc(idj * 3 * 4 + idd * 4 + 3, idc);
						}
					}
				}
			}
		}
		// std::cout << "max threads " << omp_get_max_threads() << std::endl;
		// std::cout << "num threads " << omp_get_num_threads() << std::endl;
	}
}

void LoadTotalModelFromObj(TotalModel &totalm, const std::string &path)
{
	using namespace std;
	using namespace cv;

	ifstream fin(path.c_str());
	
	if (fin.is_open() == false)
	{
		printf("## Error: Failed in opening total body model: %s\n", path.c_str());
		return;
	}
	char bufChar[512];
	totalm.m_vertices.resize(TotalModel::NUM_VERTICES, 3);
	totalm.m_uvs.resize(TotalModel::NUM_VERTICES, 2);		//note that we assume  NUM_UVS==NUM_VERTICES
	totalm.m_normals.resize(TotalModel::NUM_VERTICES, 3);
	totalm.m_faces.resize(TotalModel::NUM_FACES, 3);


	//In this obj file, UV and normals are in arbitrary order
	//Need to align this to vertices ordering by chekcing face
	vector<Point2d> uvs;
	vector<Point3d> normals;
	vector<Point3d> vertices;
	normals.reserve(TotalModel::NUM_VERTICES);
	uvs.reserve(19531);

	totalm.m_uvs.setZero();
	
	int verCnt = 0;
	// int uvCnt = 0;
	// int normalCnt = 0;
	int faceCnt = 0;
	while(fin.eof()==false)
	{
		fin >> bufChar;
		if (fin.eof())
			break;
		if (strcmp(bufChar, "v") == 0)
		{
			/*Point3d temp;
			fin >> temp.x >> temp.y >> temp.z;
			vertices.push_back(temp);*/
			fin >> totalm.m_vertices(verCnt, 0) >> totalm.m_vertices(verCnt, 1) >> totalm.m_vertices(verCnt, 2);
			verCnt++;
		}
		else if(strcmp(bufChar,"vt")==0)
		{
			Point2d temp;
			fin >> temp.x >> temp.y;
			uvs.push_back(temp);
			//fin >> totalm.m_uvs(uvCnt, 0) >> totalm.m_uvs(uvCnt, 1);
			//uvCnt++;
		}
		else if(strcmp(bufChar, "vn") == 0)
		{
			Point3d temp;
			fin >> temp.x >> temp.y >> temp.z;
			normals.push_back(temp);
			//fin >> totalm.m_normals(normalCnt, 0) >> totalm.m_normals(normalCnt, 1) >> totalm.m_normals(normalCnt, 2);
			//normalCnt++;
		}
		else if (strcmp(bufChar, "f") == 0)
		{
			for(int i=0;i<3;++i)
			{
				int vertexId, textureId, normalIdx;
				fin >> vertexId >>textureId >> normalIdx;		//v1/vt1/vn1
				//faces.push_back(temp);*/
				//if(i==0)
					//fin >> totalm.m_faces(faceCnt, 0) >> totalm.m_faces(faceCnt, 1) >> totalm.m_faces(faceCnt, 2);
				vertexId--;
				totalm.m_faces(faceCnt, i) = vertexId;			//from 1-based to 0 based

				textureId--; //from 1-based to 0 based
				totalm.m_uvs(vertexId, 0) = uvs[textureId].x;		//overlaid if there are multiple uvs for a vertex
				totalm.m_uvs(vertexId, 1) = uvs[textureId].y;
							
				normalIdx--; //from 1-based to 0 based
				totalm.m_normals(vertexId, 0) = normals[normalIdx].x;
				totalm.m_normals(vertexId, 1) = normals[normalIdx].y;
				totalm.m_normals(vertexId, 2) = normals[normalIdx].z;
			}
			faceCnt++;
		}
		else
		{
			printf("## Warning: unknown option: %s\n", bufChar);
			continue;
		}
	}
	totalm.m_bInit = true;
	fin.close();
	printf("Finishing loading Total Model\n");
}

void LoadCocoplusRegressor(TotalModel &totalm, const std::string &path)
{
	printf("Loading from: %s\n", path.c_str());
	std::ifstream file(path.c_str(), std::ifstream::in);
    Json::Value root;
    file >> root;
    file.close();
    initSparseMatrix(totalm.m_cocoplus_reg, root["cocoplus_regressor"]);
    totalm.m_cocoplus_reg.makeCompressed();
}

Eigen::VectorXd adam_reconstruct_withDerivative_eulers(const TotalModel &totalm,
	const double *parm_coeffs,
	const double *parm_pose_eulers,		//Euler pose (but the first joint's param is still angle axis to avoid gimbal lock)
	const double *parm_faceEx_coeffs,
	double *outVerts,
	MatrixXdr &dVdc,
	MatrixXdr &dVdP,
	MatrixXdr &dVdfc,
	MatrixXdr &dTJdc,
	MatrixXdr &dTJdP,
	bool joint_only,
	bool fit_face)
{
	using namespace smpl;
	using namespace Eigen;
	Map< const Matrix<double, Dynamic, 1> > c_bodyshape(parm_coeffs, TotalModel::NUM_SHAPE_COEFFICIENTS);

	Matrix<double, Dynamic, Dynamic, RowMajor> Vt(TotalModel::NUM_VERTICES, 3);
	Map< Matrix<double, Dynamic, 1> > Vt_vec(Vt.data(), 3 * TotalModel::NUM_VERTICES);
	Map< Matrix<double, Dynamic, Dynamic, RowMajor> > outV(outVerts, TotalModel::NUM_VERTICES, 3);

	Vt_vec = totalm.m_meanshape + totalm.m_shapespace_u*c_bodyshape;

	Matrix<double, TotalModel::NUM_JOINTS, 3, RowMajor> J;
	Map< Matrix<double, Dynamic, 1> > J_vec(J.data(), TotalModel::NUM_JOINTS * 3);
	J_vec = totalm.J_mu_ + totalm.dJdc_*c_bodyshape;

	const int num_t = (TotalModel::NUM_JOINTS) * 3 * 5;
	Matrix<double, Dynamic, 3 * TotalModel::NUM_JOINTS, RowMajor> dTrdP(num_t, 3 * TotalModel::NUM_JOINTS); // Tr consists of T and TJ
	Matrix<double, Dynamic, 3 * TotalModel::NUM_JOINTS, RowMajor> dTrdJ(num_t, 3 * TotalModel::NUM_JOINTS);
	VectorXd transforms_joint(3 * TotalModel::NUM_JOINTS * 4 + 3 * TotalModel::NUM_JOINTS); // the first part is transform, the second part is outJoint

	ceres::AutoDiffCostFunction<PoseToTransformsNoLR_Eulers_adamModel,
		(TotalModel::NUM_JOINTS) * 3 * 4 + 3 * TotalModel::NUM_JOINTS,
		(TotalModel::NUM_JOINTS) * 3,
		(TotalModel::NUM_JOINTS) * 3> p2t(new PoseToTransformsNoLR_Eulers_adamModel(totalm));
	const double * parameters[2] = { parm_pose_eulers, J.data() };
	double* residuals = transforms_joint.data();
	double* jacobians[2] = { dTrdP.data(), dTrdJ.data() };
	p2t.Evaluate(parameters, residuals, jacobians);		//automatically compute residuals and jacobians (dTdP and dTdJ)

	Matrix<double, Dynamic, 3 * TotalModel::NUM_JOINTS, RowMajor> dTdP(3 * TotalModel::NUM_JOINTS * 4, 3 * TotalModel::NUM_JOINTS);
	Matrix<double, Dynamic, 3 * TotalModel::NUM_JOINTS, RowMajor> dTdJ(3 * TotalModel::NUM_JOINTS * 4, 3 * TotalModel::NUM_JOINTS);
	dTdP = dTrdP.block(0, 0, 3 * TotalModel::NUM_JOINTS * 4, 3 * TotalModel::NUM_JOINTS);
	dTdJ = dTrdJ.block(0, 0, 3 * TotalModel::NUM_JOINTS * 4, 3 * TotalModel::NUM_JOINTS);

	dTJdP = dTrdP.block(3 * TotalModel::NUM_JOINTS * 4, 0, 3 * TotalModel::NUM_JOINTS, 3 * TotalModel::NUM_JOINTS);
	Matrix<double, Dynamic, 3 * TotalModel::NUM_JOINTS, RowMajor> dTJdJ =
		dTrdP.block(3 * TotalModel::NUM_JOINTS * 4, 0, 3 * TotalModel::NUM_JOINTS, 3 * TotalModel::NUM_JOINTS);
	dTJdc = dTJdJ * totalm.dJdc_;

	//Apply Face here
	if (fit_face)
	{
		Map< const Matrix<double, Dynamic, 1> > c_faceEx(parm_faceEx_coeffs, TotalModel::NUM_EXP_BASIS_COEFFICIENTS);
		Vt_vec = Vt_vec + totalm.m_dVdFaceEx * c_faceEx;		// m_C_face2total*facem.U_exp_
	}

	VectorXd transforms = transforms_joint.block(0, 0, 3 * TotalModel::NUM_JOINTS * 4, 1);
	VectorXd outJoint = transforms_joint.block(3 * TotalModel::NUM_JOINTS * 4, 0, 3 * TotalModel::NUM_JOINTS, 1);

	Matrix<double, Dynamic, TotalModel::NUM_SHAPE_COEFFICIENTS, RowMajor> dTdc = dTdJ * totalm.dJdc_;
	if (!joint_only)
	{
		adam_lbs(totalm, Vt_vec.data(), transforms, outVerts, totalm.m_shapespace_u, dTdP, dTdc, dVdc, dVdP);

		if (fit_face)
		{
			//Apply transformation for face vertices
			#pragma omp parallel for
			for (int idv = 0; idv < TotalModel::NUM_VERTICES; idv++) 
			{
				int idj = 15;
				{
					for (int idd = 0; idd < 3; idd++) 
					{
						for (int idc = 0; idc < TotalModel::NUM_EXP_BASIS_COEFFICIENTS; idc++) 
						{
							dVdfc(idv * 3 + idd, idc) = totalm.m_dVdFaceEx(idv * 3 + 0, idc) * transforms(idj * 3 * 4 + idd * 4 + 0);
							dVdfc(idv * 3 + idd, idc) += totalm.m_dVdFaceEx(idv * 3 + 1, idc) * transforms(idj * 3 * 4 + idd * 4 + 1);
							dVdfc(idv * 3 + idd, idc) += totalm.m_dVdFaceEx(idv * 3 + 2, idc) * transforms(idj * 3 * 4 + idd * 4 + 2);
						}
					}
				}
			}
		}
	}
	return outJoint;
}

void adam_reconstruct_Eulers_Fast(const TotalModel& totalm,
	const Eigen::Matrix<double, Eigen::Dynamic, 1> &Vt_vec,
	const Eigen::Matrix<double, Eigen::Dynamic, 1> &J0_vec,
	const double *parm_pose_eulers,
	const double *parm_faceEx_coeffs,
	double *outVerts,
	Eigen::VectorXd &transforms)
{
// const auto start1 = std::chrono::high_resolution_clock::now();
    using namespace smpl;

    transforms.resize(3 * TotalModel::NUM_JOINTS * 4);
    Eigen::VectorXd transforms_joint(3 * TotalModel::NUM_JOINTS * 4 + 3 * TotalModel::NUM_JOINTS); // the first part is transform, the second part is outJoint

// const auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start1).count();
// const auto start2 = std::chrono::high_resolution_clock::now();
    const int num_t = (TotalModel::NUM_JOINTS) * 3 * 4;
    Eigen::Matrix<double, Eigen::Dynamic, 3 * TotalModel::NUM_JOINTS, Eigen::RowMajor> dTrdP((TotalModel::NUM_JOINTS) * 3 * 5, 3 * TotalModel::NUM_JOINTS);
    Eigen::Matrix<double, Eigen::Dynamic, 3 * TotalModel::NUM_JOINTS, Eigen::RowMajor> dTrdJ((TotalModel::NUM_JOINTS) * 3 * 5, 3 * TotalModel::NUM_JOINTS);

// const auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start2).count();
// const auto start3 = std::chrono::high_resolution_clock::now();
    ceres::AutoDiffCostFunction<PoseToTransformsNoLR_Eulers_adamModel,
        (TotalModel::NUM_JOINTS) * 3 * 5,
        (TotalModel::NUM_JOINTS) * 3,
        (TotalModel::NUM_JOINTS) * 3> p2t(new PoseToTransformsNoLR_Eulers_adamModel(totalm));
    const double * parameters[2] = { parm_pose_eulers, J0_vec.data() };
    double * residuals = transforms_joint.data();
// const auto duration3 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start3).count();
// const auto start4 = std::chrono::high_resolution_clock::now();
    p2t.Evaluate(parameters, residuals, nullptr);       //automatically compute residuals and jacobians (dTdP and dTdJ)

// const auto duration4 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start4).count();
// const auto start5 = std::chrono::high_resolution_clock::now();
    transforms.block(0, 0, num_t, 1) = transforms_joint.block(0, 0, num_t, 1);

// const auto duration5 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start5).count();
// const auto start6 = std::chrono::high_resolution_clock::now();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Vt_with_face(TotalModel::NUM_VERTICES, 3);
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, 1> > Vt_vec_with_face(Vt_with_face.data(), 3 * TotalModel::NUM_VERTICES);
    Eigen::Map< const Eigen::Matrix<double, Eigen::Dynamic, 1> > c_faceEx(parm_faceEx_coeffs, TotalModel::NUM_EXP_BASIS_COEFFICIENTS);
// const auto duration6 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start6).count();
// const auto start7 = std::chrono::high_resolution_clock::now();
    Vt_vec_with_face = Vt_vec + totalm.m_dVdFaceEx * c_faceEx;      // m_C_face2total*facem.U_exp_

// const auto duration7 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start7).count();
// const auto start8 = std::chrono::high_resolution_clock::now();
    adam_lbs(totalm, Vt_vec_with_face.data(), transforms, outVerts);
// const auto duration8 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start8).count();
// std::cout << __FILE__ << " " << duration1 * 1e-6 << " 1\n"
//           << __FILE__ << " " << duration2 * 1e-6 << " 2\n"
//           << __FILE__ << " " << duration3 * 1e-6 << " 3\n"
//           << __FILE__ << " " << duration4 * 1e-6 << " 4\n"
//           << __FILE__ << " " << duration5 * 1e-6 << " 5\n"
//           << __FILE__ << " " << duration6 * 1e-6 << " 6\n"
//           << __FILE__ << " " << duration7 * 1e-6 << " 7\n"
//           << __FILE__ << " " << duration8 * 1e-6 << " 8\n" << std::endl;
}
