#include "handm.h"
#include "totalmodel.h"
#include "ceres/ceres.h"

void SetSolverOptions(ceres::Solver::Options *options);

void FitToHandCeres_Right_Naive(
	smpl::HandModel &handr_model,
	Eigen::MatrixXd &Joints,
	Eigen::Vector3d& parm_handr_t,
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& parm_hand_pose,
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& parm_hand_coeffs,
	int regressor_type=0);

void FitToProjectionCeres(
	smpl::HandModel &handr_model,
	Eigen::MatrixXd &Joints2d,
	const double* K,
	Eigen::MatrixXd &PAF,
	Eigen::Vector3d& parm_handr_t,
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& parm_hand_pose,
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& parm_hand_coeffs,
	int regressor_type=0);

void Adam_FitTotalBodyCeres( TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,	   //2points
	Eigen::MatrixXd &lFoot,		//2points
	Eigen::MatrixXd &rHandJoints,	   //
	Eigen::MatrixXd &lHandJoints,		//
	Eigen::MatrixXd &faceJoints);

void Adam_FitTotalBodyCeres2d( TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints2d,
	Eigen::MatrixXd &rFoot2d,	   //2points
	Eigen::MatrixXd &lFoot2d,		//2points
	Eigen::MatrixXd &rHandJoints2d,	   //
	Eigen::MatrixXd &lHandJoints2d,		//
	Eigen::MatrixXd &faceJoints2d,
	double* calibK);

void Adam_FitTotalBodyCeres3d2d( TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,	   //2points
	Eigen::MatrixXd &lFoot,		//2points
	Eigen::MatrixXd &rHandJoints,	   //
	Eigen::MatrixXd &lHandJoints,		//
	Eigen::MatrixXd &faceJoints,
	double* calibK);

void Adam_FastFit_Initialize(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,
	Eigen::MatrixXd &lFoot,
	Eigen::MatrixXd &rHandJoints,
	Eigen::MatrixXd &lHandJoints,
	Eigen::MatrixXd &faceJoints);

void Adam_FastFit(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,
	Eigen::MatrixXd &lFoot,
	Eigen::MatrixXd &rHandJoints,
	Eigen::MatrixXd &lHandJoints,
	Eigen::MatrixXd &faceJoints,
	bool verbose=false);

void Adam_Fit_PAF(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,
	Eigen::MatrixXd &lFoot,
	Eigen::MatrixXd &rHandJoints,
	Eigen::MatrixXd &lHandJoints,
	Eigen::MatrixXd &faceJoints,
	Eigen::MatrixXd &PAF,
	double* K=nullptr,
	uint regressor_type=0u,
	bool quan=false,
	bool fitPAFfirst=false);

void Adam_Fit_H36M(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints);

void Adam_skeletal_refit(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,
	Eigen::MatrixXd &lFoot,
	Eigen::MatrixXd &rHandJoints,
	Eigen::MatrixXd &lHandJoints,
	Eigen::MatrixXd &faceJoints,
	Eigen::MatrixXd &PAF,
	uint regressor_type=0u);

void Adam_skeletal_init(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,
	Eigen::MatrixXd &lFoot,
	Eigen::MatrixXd &rHandJoints,
	Eigen::MatrixXd &lHandJoints,
	Eigen::MatrixXd &faceJoints,
	Eigen::MatrixXd &PAF,
	uint regressor_type=0u);