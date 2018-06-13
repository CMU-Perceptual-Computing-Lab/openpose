#include "handm.h"
#include "totalmodel.h"
#include <VisualizedData.h>

#ifndef KINEMATICMODEL
#define KINEMATICMODEL

void GenerateMesh(CMeshModelInstance& returnMesh, double* resultJoint, const smpl::SMPLParams& targetParam, const smpl::HandModel& g_handl_model, const int regressor_type=0);
void GenerateMesh(CMeshModelInstance& returnMesh, double* resultJoint, const smpl::SMPLParams& targetParam, const TotalModel& g_total_model, const int regressor_type=0);
void GenerateMesh_Fast(CMeshModelInstance& returnMesh, double* resultJoint, const TotalModel& g_total_model,
	const Eigen::Matrix<double, Eigen::Dynamic, 1> &Vt_vec, const Eigen::Matrix<double, Eigen::Dynamic, 1> &J0_Vec,
	const double* const m_adam_pose, const double* const m_adam_facecoeffs_exp, const Eigen::Vector3d& m_adam_t);

void CopyMesh(const CMeshModelInstance& mesh, VisualizedData& g_vis_data);

#endif