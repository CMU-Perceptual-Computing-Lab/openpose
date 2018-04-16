#include "KinematicModel.h"
#include <stdexcept>
#define SMPL_VIS_SCALING 100.0f
#include <igl/per_vertex_normals.h>
#include <unsupported/Eigen/KroneckerProduct>

int map_adam_to_measure[] = {12, 15, 0, 16, 18, 20, 1, 4, 7, 17, 19, 21, 2, 5, 8, 15, 15, 15, 15,
							 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
							 21, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61};

							 
void GenerateMesh(CMeshModelInstance& returnMesh, double* resultJoint, smpl::SMPLParams& targetParam, smpl::HandModel& g_handl_model)
{
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out_v(smpl::HandModel::NUM_VERTICES, 3);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> outJ(smpl::HandModel::NUM_JOINTS, 3);
	MatrixXdr dJdc;
	MatrixXdr dJdP;

	smpl::HandModel* pHandm = NULL;

	pHandm = &g_handl_model;
	reconstruct_joints_mesh(g_handl_model, targetParam.handl_t.data(), targetParam.hand_coeffs.data(), targetParam.handl_pose.data(), outJ.data(), out_v.data(), dJdc, dJdP);

	// std::cout << targetParam.handl_t << std::endl;
	// std::cout << "here" << std::endl;
	// std::cout << targetParam.hand_coeffs << std::endl;
	// std::cout << "here" << std::endl;
	// std::cout << targetParam.handl_pose << std::endl;
	// std::cout << "here" << std::endl;
	returnMesh.m_meshType = CMeshModelInstance::MESH_TYPE_LHAND;
	returnMesh.m_vertices.reserve(pHandm->V_.rows());
	returnMesh.m_colors.reserve(pHandm->V_.rows());
	returnMesh.m_normals.reserve(pHandm->V_.rows());
	returnMesh.m_uvs.reserve(pHandm->V_.rows());

	for (int r = 0; r < pHandm->V_.rows(); r++)  //Vertices
	{
		returnMesh.m_vertices.push_back(cv::Point3f(out_v(r, 0) * SMPL_VIS_SCALING, out_v(r, 1) * SMPL_VIS_SCALING, out_v(r, 2) * SMPL_VIS_SCALING));
		returnMesh.m_colors.push_back(cv::Point3f(1, 1, 1));
		returnMesh.m_uvs.push_back(cv::Point2f(pHandm->uv(r, 0), pHandm->uv(r, 1)));		//random normal
	}
	for (int r = 0; r < pHandm->F_.rows(); ++r)		//Faces
	{
		// int idxvect[3];
		for (int c = 0; c < pHandm->F_.cols(); ++c)
		{
			const int idx = pHandm->F_(r, c);  //0based (applied -1 on loading)
			returnMesh.m_face_vertexIndices.push_back(idx);
			// idxvect[c] = idx;
		}
	}
	//Compute Normal
	Eigen::MatrixXd NV;
	igl::per_vertex_normals(Eigen::MatrixXd(out_v), pHandm->F_, NV);
	returnMesh.m_normals.reserve(NV.rows());
	for (int r = 0; r < NV.rows(); ++r)
	{
		returnMesh.m_normals.push_back(cv::Point3f(NV(r, 0), NV(r, 1), NV(r, 2)));
	}

	outJ = outJ * SMPL_VIS_SCALING;
	std::copy(outJ.data(), outJ.data()+63, resultJoint);
}


void GenerateMesh(CMeshModelInstance& returnMesh, double* resultJoint, smpl::SMPLParams& targetParam, TotalModel& g_total_model)
{
	Eigen::Matrix<double, Eigen::Dynamic, 1> outV(TotalModel::NUM_VERTICES * 3);
	Eigen::VectorXd transforms;
	adam_reconstruct_Eulers(g_total_model,
		targetParam.m_adam_coeffs.data(),
		targetParam.m_adam_pose.data(),
		targetParam.m_adam_facecoeffs_exp.data(),
		outV.data(),
		transforms);
	Eigen::SparseMatrix<double, Eigen::ColMajor> eye3(3, 3); eye3.setIdentity();
	Eigen::SparseMatrix<double, Eigen::ColMajor> dVdt = Eigen::kroneckerProduct(Eigen::VectorXd::Ones(TotalModel::NUM_VERTICES), eye3);
	outV += dVdt * targetParam.m_adam_t; //translation is applied at the end
	returnMesh.m_meshType = CMeshModelInstance::MESH_TYPE_ADAM;
	returnMesh.m_vertices.resize(TotalModel::NUM_VERTICES);
	returnMesh.m_colors.resize(TotalModel::NUM_VERTICES);
	returnMesh.m_uvs.resize(TotalModel::NUM_VERTICES);

	for (int idx = 0; idx < TotalModel::NUM_VERTICES; idx++)  //Vertices
	{
		returnMesh.m_vertices[idx] = cv::Point3f(outV(3 * idx), outV(3 * idx + 1), outV(3 * idx + 2));
		returnMesh.m_colors[idx] = cv::Point3f(1.0, 1.0, 0.9);			//Adam color
	}

	//Visualize SMPL Skeletons
	Eigen::Matrix<double, Eigen::Dynamic, 1> joints = g_total_model.m_J_reg * outV;
	// int jointIdx = 0;
	for (int r = 0; r < joints.rows(); r += 3)		//Faces
	{
		returnMesh.m_joints.push_back(cv::Point3f(joints(r), joints(r + 1), joints(r + 2)));
		// jointIdx++;
	}
	Eigen::Matrix<double, Eigen::Dynamic, 1> joints_smc = g_total_model.m_J_reg_smc * outV;
	for (int r = 0; r < joints.rows(); r += 3)		//Faces
	{
		returnMesh.m_joints_regress.push_back(cv::Point3f(joints_smc(r), joints_smc(r + 1), joints_smc(r + 2)));
	}
	returnMesh.m_face_vertexIndices.reserve(g_total_model.m_faces.size() * 3);
	for (int r = 0; r < g_total_model.m_faces.rows(); ++r)		//Faces
	{
		int idxvect[3];
		for (int c = 0; c < g_total_model.m_faces.cols(); ++c)
		{
			int idx = g_total_model.m_faces(r, c);
			idxvect[c] = idx;
		}
		returnMesh.m_face_vertexIndices.push_back(idxvect[0]);
		returnMesh.m_face_vertexIndices.push_back(idxvect[1]);
		returnMesh.m_face_vertexIndices.push_back(idxvect[2]);
	}
	returnMesh.RecomputeNomral((void*)(&g_total_model));
	for (int i = 0; i < 61; i++)
	{
		resultJoint[3 * i] = joints_smc(3* map_adam_to_measure[i]);
		resultJoint[3 * i + 1] = joints_smc(3 * map_adam_to_measure[i] + 1);
		resultJoint[3 * i + 2] = joints_smc(3 * map_adam_to_measure[i] + 2);
	}
}

void GenerateMesh_Fast(CMeshModelInstance& returnMesh, double* resultJoint, smpl::SMPLParams& targetParam, TotalModel& g_total_model,
	const Eigen::Matrix<double, Eigen::Dynamic, 1> &Vt_vec, const Eigen::Matrix<double, Eigen::Dynamic, 1> &J0_vec)
{
	Eigen::Matrix<double, Eigen::Dynamic, 1> outV(TotalModel::NUM_VERTICES * 3);
	Eigen::VectorXd transforms;
	adam_reconstruct_Eulers_Fast(g_total_model,
		Vt_vec,
		J0_vec,
		targetParam.m_adam_pose.data(),
		targetParam.m_adam_facecoeffs_exp.data(),
		outV.data(),
		transforms);
	Eigen::SparseMatrix<double, Eigen::ColMajor> eye3(3, 3); eye3.setIdentity();
	Eigen::SparseMatrix<double, Eigen::ColMajor> dVdt = Eigen::kroneckerProduct(Eigen::VectorXd::Ones(TotalModel::NUM_VERTICES), eye3);
	outV += dVdt * targetParam.m_adam_t; //translation is applied at the end
	returnMesh.m_meshType = CMeshModelInstance::MESH_TYPE_ADAM;
	returnMesh.m_vertices.resize(TotalModel::NUM_VERTICES);
	returnMesh.m_colors.resize(TotalModel::NUM_VERTICES);
	returnMesh.m_uvs.resize(TotalModel::NUM_VERTICES);

	for (int idx = 0; idx < TotalModel::NUM_VERTICES; idx++)  //Vertices
	{
		returnMesh.m_vertices[idx] = cv::Point3f(outV(3 * idx), outV(3 * idx + 1), outV(3 * idx + 2));
		returnMesh.m_colors[idx] = cv::Point3f(1.0, 1.0, 0.9);			//Adam color
	}

	//Visualize SMPL Skeletons
	Eigen::Matrix<double, Eigen::Dynamic, 1> joints = g_total_model.m_J_reg * outV;
	// int jointIdx = 0;
	for (int r = 0; r < joints.rows(); r += 3)		//Faces
	{
		returnMesh.m_joints.push_back(cv::Point3f(joints(r), joints(r + 1), joints(r + 2)));
		// jointIdx++;
	}
	Eigen::Matrix<double, Eigen::Dynamic, 1> joints_smc = g_total_model.m_J_reg_smc * outV;
	for (int r = 0; r < joints.rows(); r += 3)		//Faces
	{
		returnMesh.m_joints_regress.push_back(cv::Point3f(joints_smc(r), joints_smc(r + 1), joints_smc(r + 2)));
	}
	returnMesh.m_face_vertexIndices.reserve(g_total_model.m_faces.size() * 3);
	for (int r = 0; r < g_total_model.m_faces.rows(); ++r)		//Faces
	{
		int idxvect[3];
		for (int c = 0; c < g_total_model.m_faces.cols(); ++c)
		{
			int idx = g_total_model.m_faces(r, c);
			idxvect[c] = idx;
		}
		returnMesh.m_face_vertexIndices.push_back(idxvect[0]);
		returnMesh.m_face_vertexIndices.push_back(idxvect[1]);
		returnMesh.m_face_vertexIndices.push_back(idxvect[2]);
	}
	returnMesh.RecomputeNomral((void*)(&g_total_model));
	for (int i = 0; i < 61; i++)
	{
		resultJoint[3 * i] = joints_smc(3* map_adam_to_measure[i]);
		resultJoint[3 * i + 1] = joints_smc(3 * map_adam_to_measure[i] + 1);
		resultJoint[3 * i + 2] = joints_smc(3 * map_adam_to_measure[i] + 2);
	}
}

void CopyMesh(CMeshModelInstance& mesh, VisualizedData& g_visData)
{
	int vertexCnt = g_visData.m_meshVertices.size();
	g_visData.m_meshVertices.insert(g_visData.m_meshVertices.end(), mesh.m_vertices.begin(), mesh.m_vertices.end());
	g_visData.m_meshVerticesColor.insert(g_visData.m_meshVerticesColor.end(), mesh.m_colors.begin(), mesh.m_colors.end());
	g_visData.m_meshVerticesNormal.insert(g_visData.m_meshVerticesNormal.end(), mesh.m_normals.begin(), mesh.m_normals.end());
	g_visData.m_meshVerticesUV.insert(g_visData.m_meshVerticesUV.end(), mesh.m_uvs.begin(), mesh.m_uvs.end());

	g_visData.m_meshIndices.reserve(g_visData.m_meshVertices.size());
	for (auto i = 0u; i < mesh.m_face_vertexIndices.size(); i++)
	{
		g_visData.m_meshIndices.push_back(mesh.m_face_vertexIndices[i] + vertexCnt);
	}
}