#include "handm.h"
#include "totalmodel.h"
#include <VisualizedData.h>

#ifndef KINEMATICMODEL
#define KINEMATICMODEL

void GenerateMesh(CMeshModelInstance& returnMesh, double* resultJoint, smpl::SMPLParams& targetParam, smpl::HandModel& g_handl_model, const int regressor_type=0);
void GenerateMesh(CMeshModelInstance& returnMesh, double* resultJoint, smpl::SMPLParams& targetParam, TotalModel& g_total_model, const int regressor_type=0);
void GenerateMesh_Fast(CMeshModelInstance& returnMesh, double* resultJoint, smpl::SMPLParams& targetParam, TotalModel& g_total_model, 
	const Eigen::Matrix<double, Eigen::Dynamic, 1> &Vt_vec, const Eigen::Matrix<double, Eigen::Dynamic, 1> &J0_Vec);

void CopyMesh(const CMeshModelInstance& mesh, VisualizedData& g_vis_data);

#endif