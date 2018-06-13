#include "KinematicModel.h"
#include <stdexcept>
#define SMPL_VIS_SCALING 100.0f
#include <igl/per_vertex_normals.h>
#include <unsupported/Eigen/KroneckerProduct>

std::array<int, 62> map_adam_to_measure = {12, 15, 0, 16, 18, 20, 1, 4, 7, 17, 19, 21, 2, 5, 8, 15, 15, 15, 15, 15,
							 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
							 21, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61};
std::array<int, 20> map_cocoreg_to_measure = {12, 14, 12, 9, 10, 11, 3, 4, 5, 8, 7, 6, 2, 1, 0, 15, 17, 16, 18, 13};

							 
void GenerateMesh(CMeshModelInstance& returnMesh, double* resultJoint, const smpl::SMPLParams& targetParam, const smpl::HandModel& g_handl_model, const int regressor_type)
{
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out_v(smpl::HandModel::NUM_VERTICES, 3);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> outJ(smpl::HandModel::NUM_JOINTS, 3);
	MatrixXdr dJdc;
	MatrixXdr dJdP;

	const smpl::HandModel* const pHandm = &g_handl_model;
	reconstruct_joints_mesh(g_handl_model, targetParam.handl_t.data(), targetParam.hand_coeffs.data(), targetParam.handl_pose.data(), outJ.data(), out_v.data(), dJdc, dJdP, regressor_type);

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


void GenerateMesh(CMeshModelInstance& returnMesh, double* resultJoint, const smpl::SMPLParams& targetParam, const TotalModel& g_total_model, const int regressor_type)
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
	for (int r = 0; r < joints.rows(); r += 3)		//Faces
	{
		returnMesh.m_joints.push_back(cv::Point3f(joints(r), joints(r + 1), joints(r + 2)));
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
	returnMesh.RecomputeNormal(g_total_model);
	if (regressor_type == 0)
	{
		for (int i = 0; i < 62; i++)
		{
			resultJoint[3 * i] = joints(3* map_adam_to_measure[i]);
			resultJoint[3 * i + 1] = joints(3 * map_adam_to_measure[i] + 1);
			resultJoint[3 * i + 2] = joints(3 * map_adam_to_measure[i] + 2);
		}
		std::copy(outV.data() + 3 * 8130, outV.data() + 3 * 8131, resultJoint + 3 * 1); // nose
		std::copy(outV.data() + 3 * 6731, outV.data() + 3 * 6732, resultJoint + 3 * 15); // leye
		std::copy(outV.data() + 3 * 6970, outV.data() + 3 * 6971, resultJoint + 3 * 16); // lear
		std::copy(outV.data() + 3 * 4131, outV.data() + 3 * 4132, resultJoint + 3 * 17); // reye
		std::copy(outV.data() + 3 * 10088, outV.data() + 3 * 10089, resultJoint + 3 * 18); // rear
	}
	else if (regressor_type == 1)
	{
		Eigen::Map<MatrixXdr> m_outV(outV.data(), TotalModel::NUM_VERTICES, 3);
		MatrixXdr J_coco = g_total_model.m_cocoplus_reg * m_outV;
		for (int i = 0; i < 20; i++)
		{
			resultJoint[3 * i] = J_coco(map_cocoreg_to_measure[i], 0);
			resultJoint[3 * i + 1] = J_coco(map_cocoreg_to_measure[i], 1);
			resultJoint[3 * i + 2] = J_coco(map_cocoreg_to_measure[i], 2);
		}
		resultJoint[3 * 2 + 0] = 0.5 * (resultJoint[3 * 6 + 0] + resultJoint[3 * 12 + 0]);
		resultJoint[3 * 2 + 1] = 0.5 * (resultJoint[3 * 6 + 1] + resultJoint[3 * 12 + 1]);
		resultJoint[3 * 2 + 2] = 0.5 * (resultJoint[3 * 6 + 2] + resultJoint[3 * 12 + 2]);
		for (int i = 20; i < 61; i++)
		{
			resultJoint[3 * i] = joints(3 * map_adam_to_measure[i]);
			resultJoint[3 * i + 1] = joints(3 * map_adam_to_measure[i] + 1);
			resultJoint[3 * i + 2] = joints(3 * map_adam_to_measure[i] + 2);
		}
	}
	else
	{
		assert(regressor_type == 2);
		std::cout << "Reconstruct mesh using regressor_type 2" << std::endl;
		Eigen::Map<MatrixXdr> m_outV(outV.data(), TotalModel::NUM_VERTICES, 3);
		MatrixXdr J_coco = g_total_model.m_small_coco_reg * m_outV;
		for (int i = 0; i < 20; i++)
		{
			resultJoint[3 * i] = J_coco(map_cocoreg_to_measure[i], 0);
			resultJoint[3 * i + 1] = J_coco(map_cocoreg_to_measure[i], 1);
			resultJoint[3 * i + 2] = J_coco(map_cocoreg_to_measure[i], 2);
		}
		resultJoint[3 * 2 + 0] = 0.5 * (resultJoint[3 * 6 + 0] + resultJoint[3 * 12 + 0]);
		resultJoint[3 * 2 + 1] = 0.5 * (resultJoint[3 * 6 + 1] + resultJoint[3 * 12 + 1]);
		resultJoint[3 * 2 + 2] = 0.5 * (resultJoint[3 * 6 + 2] + resultJoint[3 * 12 + 2]);
		for (int i = 20; i < 62; i++)
		{
			resultJoint[3 * i] = joints(3 * map_adam_to_measure[i]);
			resultJoint[3 * i + 1] = joints(3 * map_adam_to_measure[i] + 1);
			resultJoint[3 * i + 2] = joints(3 * map_adam_to_measure[i] + 2);
		}
		resultJoint[3 * 20 + 0] = J_coco(11, 0);
		resultJoint[3 * 20 + 1] = J_coco(11, 1);
		resultJoint[3 * 20 + 2] = J_coco(11, 2);
		resultJoint[3 * 41 + 0] = J_coco(6, 0);
		resultJoint[3 * 41 + 1] = J_coco(6, 1);
		resultJoint[3 * 41 + 2] = J_coco(6, 2);
	}
}

void GenerateMesh_Fast(CMeshModelInstance& returnMesh, double* resultJoint, const TotalModel& g_total_model,
	const Eigen::Matrix<double, Eigen::Dynamic, 1> &Vt_vec, const Eigen::Matrix<double, Eigen::Dynamic, 1> &J0_vec,
	const double* const ptr_adam_pose, const double* const ptr_adam_facecoeffs_exp, const Eigen::Vector3d& adam_t)
{
// const auto start = std::chrono::high_resolution_clock::now();
// const auto start1 = std::chrono::high_resolution_clock::now();
	Eigen::Matrix<double, Eigen::Dynamic, 1> outV(TotalModel::NUM_VERTICES * 3);
	Eigen::VectorXd transforms;
// const auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start1).count();
// const auto start2 = std::chrono::high_resolution_clock::now();
	adam_reconstruct_Eulers_Fast(g_total_model,
		Vt_vec,
		J0_vec,
		ptr_adam_pose,
		ptr_adam_facecoeffs_exp,
		outV.data(),
		transforms);
// const auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start2).count();
// const auto start3 = std::chrono::high_resolution_clock::now();
    Eigen::SparseMatrix<double, Eigen::ColMajor> eye3(3, 3);
    eye3.setIdentity();
    Eigen::SparseMatrix<double, Eigen::ColMajor> dVdt = Eigen::kroneckerProduct(Eigen::VectorXd::Ones(TotalModel::NUM_VERTICES), eye3);
// const auto duration3 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start3).count();
// const auto start4 = std::chrono::high_resolution_clock::now();
	outV += dVdt * adam_t; //translation is applied at the end
// const auto duration4 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start4).count();
// const auto start5 = std::chrono::high_resolution_clock::now();
	returnMesh.m_meshType = CMeshModelInstance::MESH_TYPE_ADAM;
	returnMesh.m_vertices.resize(TotalModel::NUM_VERTICES);
	returnMesh.m_colors.resize(TotalModel::NUM_VERTICES, cv::Point3f(1.0, 1.0, 0.9)); // Adam color
	returnMesh.m_uvs.resize(TotalModel::NUM_VERTICES);
// const auto duration5 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start5).count();
// const auto start6 = std::chrono::high_resolution_clock::now();

    const auto* outV_data = outV.data();
	for (int idx = 0; idx < TotalModel::NUM_VERTICES; idx++)  //Vertices
	{
        const auto* outVrow_data = &outV_data[3*idx];
        returnMesh.m_vertices[idx] = cv::Point3f(outVrow_data[0], outVrow_data[1], outVrow_data[2]);
        // Slow equivalent
        // returnMesh.m_vertices[idx] = cv::Point3f(outV(3 * idx), outV(3 * idx + 1), outV(3 * idx + 2));
	}
// const auto duration6 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start6).count();
// const auto start7 = std::chrono::high_resolution_clock::now();

	//Visualize SMPL Skeletons
	Eigen::Matrix<double, Eigen::Dynamic, 1> joints = g_total_model.m_J_reg * outV;
// const auto duration7 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start7).count();
// const auto start8 = std::chrono::high_resolution_clock::now();
	// int jointIdx = 0;
	for (int r = 0; r < joints.rows(); r += 3)		//Faces
	{
		returnMesh.m_joints.push_back(cv::Point3f(joints(r), joints(r + 1), joints(r + 2)));
		// jointIdx++;
	}
// const auto duration8 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start8).count();
// const auto start9 = std::chrono::high_resolution_clock::now();
	Eigen::Matrix<double, Eigen::Dynamic, 1> joints_smc = g_total_model.m_J_reg_smc * outV;
// const auto duration9 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start9).count();
// const auto start10 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < joints.rows(); r += 3)		//Faces
    {
        returnMesh.m_joints_regress.push_back(cv::Point3f(joints_smc(r), joints_smc(r + 1), joints_smc(r + 2)));
    }
// const auto duration10 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start10).count();
// const auto start11 = std::chrono::high_resolution_clock::now();
	returnMesh.m_face_vertexIndices.reserve(g_total_model.m_faces.size() * 3);
// const auto duration11 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start11).count();
// const auto start12 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < g_total_model.m_faces.rows(); ++r)		//Faces
    {
        const auto* facesPtr = &g_total_model.m_faces.data()[3*r];
        returnMesh.m_face_vertexIndices.push_back(facesPtr[0]);
        returnMesh.m_face_vertexIndices.push_back(facesPtr[1]);
        returnMesh.m_face_vertexIndices.push_back(facesPtr[2]);
        // Slow equivalent
        // returnMesh.m_face_vertexIndices.push_back(g_total_model.m_faces(r, 0));
        // returnMesh.m_face_vertexIndices.push_back(g_total_model.m_faces(r, 1));
        // returnMesh.m_face_vertexIndices.push_back(g_total_model.m_faces(r, 2));
    }
// const auto duration12 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start12).count();
// const auto start13 = std::chrono::high_resolution_clock::now();
	returnMesh.RecomputeNormal(g_total_model);
// const auto duration13 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start13).count();
// const auto start14 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 62; i++)
	{
		resultJoint[3 * i] = joints(3* map_adam_to_measure[i]);
		resultJoint[3 * i + 1] = joints(3 * map_adam_to_measure[i] + 1);
		resultJoint[3 * i + 2] = joints(3 * map_adam_to_measure[i] + 2);
	}
	std::copy(outV.data() + 3 * 8130, outV.data() + 3 * 8131, resultJoint + 3 * 1); // nose
	std::copy(outV.data() + 3 * 6731, outV.data() + 3 * 6732, resultJoint + 3 * 15); // leye
	std::copy(outV.data() + 3 * 6970, outV.data() + 3 * 6971, resultJoint + 3 * 16); // lear
	std::copy(outV.data() + 3 * 4131, outV.data() + 3 * 4132, resultJoint + 3 * 17); // reye
	std::copy(outV.data() + 3 * 10088, outV.data() + 3 * 10089, resultJoint + 3 * 18); // rear
// const auto duration14 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start14).count();
// const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
// std::cout << __FILE__ << " " << duration1 * 1e-6 << " 1\n"
//           << __FILE__ << " " << duration2 * 1e-6 << " 2\n"
//           << __FILE__ << " " << duration3 * 1e-6 << " 3\n"
//           << __FILE__ << " " << duration4 * 1e-6 << " 4\n"
//           << __FILE__ << " " << duration5 * 1e-6 << " 5\n"
//           << __FILE__ << " " << duration6 * 1e-6 << " 6\n"
//           << __FILE__ << " " << duration7 * 1e-6 << " 7\n"
//           << __FILE__ << " " << duration8 * 1e-6 << " 8\n"
//           << __FILE__ << " " << duration9 * 1e-6 << " 9\n"
//           << __FILE__ << " " << duration10 * 1e-6 << " 10\n"
//           << __FILE__ << " " << duration11 * 1e-6 << " 11\n"
//           << __FILE__ << " " << duration12 * 1e-6 << " 12\n"
//           << __FILE__ << " " << duration13 * 1e-6 << " 13\n"
//           << __FILE__ << " " << duration14 * 1e-6 << " 14\n"
//           << __FILE__ << "T: " << duration* 1e-6 << " \n" << std::endl;
}

void CopyMesh(const CMeshModelInstance& mesh, VisualizedData& g_visData)
{
	const int vertexCnt = g_visData.m_meshVertices.size();
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