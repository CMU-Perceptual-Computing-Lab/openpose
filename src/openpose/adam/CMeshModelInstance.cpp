#include "CMeshModelInstance.h"
#include <Eigen/Dense>
#include <igl/per_vertex_normals.h>
#include <assert.h>
#include "totalmodel.h"

void CMeshModelInstance::RecomputeNomral(void* ptr_model)
{
	//Compute Normal
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> V_3(m_vertices.size(), 3);
	
	for (int r = 0; r < V_3.rows(); ++r)
	{
		V_3(r, 0) = m_vertices[r].x;
		V_3(r, 1) = m_vertices[r].y;
		V_3(r, 2) = m_vertices[r].z;
	}
	Eigen::MatrixXd NV;

	if(m_meshType==MESH_TYPE_SMPL)
	{
		std::string errorMessage("Not supporting MESH_TYPE_SMPL currently");
		throw std::runtime_error(errorMessage);
		// igl::per_vertex_normals(V_3, g_smpl.faces_, NV);
	}
	if (m_meshType == MESH_TYPE_TOTAL || m_meshType == MESH_TYPE_ADAM)
	{
		assert(ptr_model != NULL);
		TotalModel* ptr_totalmodel = (TotalModel*)(ptr_model);
		igl::per_vertex_normals(V_3, ptr_totalmodel->m_faces, NV);
	}
	m_normals.clear();
	m_normals.resize(NV.rows());
	for (int r = 0; r < NV.rows(); ++r)
	{
		m_normals[r] = cv::Point3f(NV(r, 0), NV(r, 1), NV(r, 2));
	}
}