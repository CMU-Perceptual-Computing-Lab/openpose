#include "FitToBody.h"
#include "ceres/normal_prior.h"
#include <iostream>
#include "FitCost.h"
#include "AdamFastCost.h"

void FreezeJoint(ceres::Problem& problem, double* dataPtr, int index)
{
	problem.SetParameterLowerBound(dataPtr, index, -0.00001);
	problem.SetParameterUpperBound(dataPtr, index, 0.00001);
}

void SetSolverOptions(ceres::Solver::Options *options) {
	CHECK(StringToLinearSolverType("sparse_normal_cholesky",
		&options->linear_solver_type));
	CHECK(StringToPreconditionerType("jacobi",
		&options->preconditioner_type));
	options->num_linear_solver_threads = 4;
	options->max_num_iterations = 15;
	options->num_threads = 10;
	options->dynamic_sparsity = true;
	options->use_nonmonotonic_steps = true;
	CHECK(StringToTrustRegionStrategyType("levenberg_marquardt",
		&options->trust_region_strategy_type));
}


void FitToHandCeres_Right_Naive(
	smpl::HandModel &hand_model,
	Eigen::MatrixXd &Joints,
	Eigen::Vector3d& parm_handl_t,
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& parm_hand_pose,
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& parm_hand_coeffs,
	MatrixXdr &outV)
{
	using namespace Eigen;
	Map< VectorXd > V_vec(outV.data(), outV.size());

	ceres::Problem problem;
	ceres::Solver::Options options;

	ceres::CostFunction* fit_cost_analytic_ha =
		new ceres::AutoDiffCostFunction<Hand3DCostPose, smpl::HandModel::NUM_JOINTS * 3, 3, smpl::HandModel::NUM_JOINTS*3, smpl::HandModel::NUM_SHAPE_COEFFICIENTS>
		(new Hand3DCostPose(hand_model, 0, Joints) ) ;
	problem.AddResidualBlock(fit_cost_analytic_ha,
		NULL,
		parm_handl_t.data(),
		parm_hand_pose.data(),
		parm_hand_coeffs.data());

	// Regularization
	ceres::CostFunction *hand_pose_reg = new ceres::AutoDiffCostFunction
		<CoeffsParameterNorm,
		smpl::HandModel::NUM_JOINTS * 3,
		smpl::HandModel::NUM_JOINTS * 3>(new CoeffsParameterNorm(smpl::HandModel::NUM_JOINTS * 3));
	ceres::LossFunction* hand_pose_reg_loss = new ceres::ScaledLoss(NULL,
		1e-9 * 5,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(hand_pose_reg,
		hand_pose_reg_loss,
		parm_hand_pose.data());

	SetSolverOptions(&options);
	options.update_state_every_iteration = true;
	options.max_num_iterations = 100;
	options.max_solver_time_in_seconds = 8200;
	options.use_nonmonotonic_steps = true;
	options.dynamic_sparsity = true;
	options.min_lm_diagonal = 2e7;

	CHECK(StringToLinearSolverType("sparse_normal_cholesky",
		&options.linear_solver_type));
	// std::cout << "Before: coeff" << parm_hand_coeffs << std::endl;
	// std::cout << "Before: pose" << parm_hand_pose << std::endl;
	// std::cout << "Before: trans" << parm_handr_t << std::endl;
	ceres::Solver::Summary summary;
	problem.SetParameterBlockConstant(parm_hand_coeffs.data());
	// problem.SetParameterBlockConstant(parm_hand_pose.data());
	ceres::Solve(options, &problem, &summary);
	// std::cout << summary.FullReport() << "\n";

	problem.SetParameterBlockVariable(parm_hand_coeffs.data());
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";
	// parm_hand_coeffs.col(1) = parm_hand_coeffs.col(0);
	// parm_hand_coeffs.col(2) = parm_hand_coeffs.col(0);
	std::cout << "After: coeff" << parm_hand_coeffs << std::endl;
	std::cout << "After: pose" << parm_hand_pose << std::endl;
	std::cout << "After: trans" << parm_handl_t << std::endl;

	printf("FitToHandCeres_Right_Naive: Done\n");
}

void FitToProjectionCeres(
	smpl::HandModel &hand_model,
	Eigen::MatrixXd &Joints2d,
	Eigen::MatrixXd &K,
	Eigen::Vector3d& parm_handl_t,
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& parm_hand_pose,
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& parm_hand_coeffs
)
{
	using namespace Eigen;
	ceres::Problem problem;
	ceres::Solver::Options options;
	// define the reprojection error

	ceres::CostFunction* fit_cost_analytic_ha =
		new ceres::AutoDiffCostFunction<Hand2DProjectionCost, smpl::HandModel::NUM_JOINTS * 2, 3, smpl::HandModel::NUM_JOINTS*3, smpl::HandModel::NUM_SHAPE_COEFFICIENTS>
		(new Hand2DProjectionCost(hand_model, Joints2d, K) ) ;
	problem.AddResidualBlock(fit_cost_analytic_ha,
		NULL,
		parm_handl_t.data(),
		parm_hand_pose.data(),
		parm_hand_coeffs.data());

	// Regularization
	ceres::CostFunction *hand_pose_reg = new ceres::AutoDiffCostFunction
		<CoeffsParameterNorm,
		smpl::HandModel::NUM_JOINTS * 3,
		smpl::HandModel::NUM_JOINTS * 3>(new CoeffsParameterNorm(smpl::HandModel::NUM_JOINTS * 3));
	ceres::LossFunction* hand_pose_reg_loss = new ceres::ScaledLoss(NULL,
		1e-6 * 5,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(hand_pose_reg,
		hand_pose_reg_loss,
		parm_hand_pose.data());
	ceres::CostFunction *hand_coeff_reg = new ceres::AutoDiffCostFunction
		<CoeffsParameterLogNorm,
		smpl::HandModel::NUM_JOINTS * 3,
		smpl::HandModel::NUM_JOINTS * 3>(new CoeffsParameterLogNorm(smpl::HandModel::NUM_JOINTS * 3));
	ceres::LossFunction* hand_coeff_reg_loss = new ceres::ScaledLoss(NULL,
		1e-3 * 1,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(hand_coeff_reg,
		hand_coeff_reg_loss,
		parm_hand_coeffs.data());

	SetSolverOptions(&options);
	options.update_state_every_iteration = true;
	options.max_num_iterations = 100;
	options.max_solver_time_in_seconds = 8200;
	options.use_nonmonotonic_steps = true;
	options.dynamic_sparsity = true;
	options.min_lm_diagonal = 2e7;

	CHECK(StringToLinearSolverType("sparse_normal_cholesky",
		&options.linear_solver_type));
	ceres::Solver::Summary summary;
	problem.SetParameterBlockConstant(parm_hand_coeffs.data());
	problem.SetParameterBlockConstant(parm_hand_pose.data());
	ceres::Solve(options, &problem, &summary);
	problem.SetParameterBlockVariable(parm_hand_pose.data());
	// problem.SetParameterBlockVariable(parm_hand_coeffs.data());
	ceres::Solve(options, &problem, &summary);
	// std::cout << summary.FullReport() << "\n";
	// problem.SetParameterBlockVariable(parm_hand_coeffs.data());
	// ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";
	std::cout << "After: coeff" << parm_hand_coeffs << std::endl;
	std::cout << "After: pose" << parm_hand_pose << std::endl;
	std::cout << "After: trans" << parm_handl_t << std::endl;

	printf("FitToProjectionCeres: Done\n");
}


void Adam_FitTotalBodyCeres(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,	   //2points
	Eigen::MatrixXd &lFoot,		//2points
	Eigen::MatrixXd &rHandJoints,	   //
	Eigen::MatrixXd &lHandJoints,		//
	Eigen::MatrixXd &faceJoints)
{
	using namespace Eigen;
	ceres::Problem problem_init;
	CostFunc_Adam_keypoints_withFoot *cost_body_keypoints_init = new CostFunc_Adam_keypoints_withFoot(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints);

	problem_init.AddResidualBlock(cost_body_keypoints_init,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());	

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_coeffs_init = new ceres::AutoDiffCostFunction
		<CoeffsParameterNorm,
		TotalModel::NUM_SHAPE_COEFFICIENTS,
		TotalModel::NUM_SHAPE_COEFFICIENTS>(new CoeffsParameterNorm(TotalModel::NUM_SHAPE_COEFFICIENTS));
	ceres::LossFunction* loss_weight_prior_body_coeffs_init = new ceres::ScaledLoss(NULL,
		1e-4,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(cost_prior_body_coeffs_init,
		loss_weight_prior_body_coeffs_init,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_pose_init = new ceres::AutoDiffCostFunction
		<AdamBodyPoseParamPrior,
		TotalModel::NUM_POSE_PARAMETERS,
		TotalModel::NUM_POSE_PARAMETERS>(new AdamBodyPoseParamPrior(TotalModel::NUM_POSE_PARAMETERS));

	ceres::LossFunction* loss_weight_prior_body_pose_init = new ceres::ScaledLoss(NULL,
		500.0f,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(cost_prior_body_pose_init,
		loss_weight_prior_body_pose_init,
		frame_param.m_adam_pose.data());

	ceres::Solver::Options options_init;
	ceres::Solver::Summary summary;
	SetSolverOptions(&options_init);
	options_init.max_num_iterations = 10;
	options_init.use_nonmonotonic_steps = false;
	options_init.num_linear_solver_threads = 10;
	options_init.minimizer_progress_to_stdout = true;
	cost_body_keypoints_init->toggle_activate(false, false);
	cost_body_keypoints_init->joint_only = true;
	// problem_init.SetParameterBlockConstant(frame_param.m_adam_pose.data());
	problem_init.SetParameterBlockConstant(frame_param.m_adam_coeffs.data());
	// ceres::Solve(options, &problem, &summary);
	// std::cout << "Finish translation" << std::endl;
	// problem_init.SetParameterBlockVariable(frame_param.m_adam_pose.data());
	ceres::Solve(options_init, &problem_init, &summary);
	std::cout << summary.FullReport() << std::endl;

	ceres::Problem problem;
	CostFunc_Adam_keypoints_withFoot *cost_body_keypoints = new CostFunc_Adam_keypoints_withFoot(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints);

	problem.AddResidualBlock(cost_body_keypoints,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());	

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_coeffs = new ceres::AutoDiffCostFunction
		<CoeffsParameterNorm,
		TotalModel::NUM_SHAPE_COEFFICIENTS,
		TotalModel::NUM_SHAPE_COEFFICIENTS>(new CoeffsParameterNorm(TotalModel::NUM_SHAPE_COEFFICIENTS));
	ceres::LossFunction* loss_weight_prior_body_coeffs = new ceres::ScaledLoss(NULL,
		1e-4,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_coeffs,
		loss_weight_prior_body_coeffs,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_pose = new ceres::AutoDiffCostFunction
		<AdamBodyPoseParamPrior,
		TotalModel::NUM_POSE_PARAMETERS,
		TotalModel::NUM_POSE_PARAMETERS>(new AdamBodyPoseParamPrior(TotalModel::NUM_POSE_PARAMETERS));

	ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_pose,
		loss_weight_prior_body_pose,
		frame_param.m_adam_pose.data());

	ceres::Solver::Options options;
	// ceres::Solver::Summary summary;
	SetSolverOptions(&options);
	options.function_tolerance = 1e-4;
	options.max_num_iterations = 20;
	options.use_nonmonotonic_steps = false;
	options.num_linear_solver_threads = 10;
	options.minimizer_progress_to_stdout = true;
	cost_body_keypoints->joint_only = true;
	cost_body_keypoints->toggle_activate(true, false);
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;

	// cost_body_keypoints->joint_only = false;
	cost_body_keypoints->toggle_activate(true, true);
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;

	// options.max_num_iterations = 5;
	// cost_body_keypoints->joint_only = false;
	// ceres::Solve(options, &problem, &summary);
	// std::cout << summary.FullReport() << std::endl;
}

void Adam_FitTotalBodyCeres2d(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints2d,
	Eigen::MatrixXd &rFoot2d,	   //2points
	Eigen::MatrixXd &lFoot2d,		//2points
	Eigen::MatrixXd &rHandJoints2d,	   //
	Eigen::MatrixXd &lHandJoints2d,		//
	Eigen::MatrixXd &faceJoints2d,
	double* calibK)
{
	using namespace Eigen;

	ceres::Problem problem_init;
	CostFunc_Adam_keypoints_withFoot *cost_body_keypoints_init = new CostFunc_Adam_keypoints_withFoot(adam, BodyJoints2d, rFoot2d, lFoot2d, faceJoints2d,
																								 lHandJoints2d, rHandJoints2d, calibK, false, 1u);

	problem_init.AddResidualBlock(cost_body_keypoints_init,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());	

	ceres::Solver::Options options_init;
	ceres::Solver::Summary summary_init;
	SetSolverOptions(&options_init);
	options_init.function_tolerance = 1e-4;
	options_init.max_num_iterations = 20;
	options_init.use_nonmonotonic_steps = false;
	options_init.num_linear_solver_threads = 10;
	options_init.minimizer_progress_to_stdout = true;
	cost_body_keypoints_init->joint_only = true;
	// Pure Translation, fit body only!
	cost_body_keypoints_init->toggle_activate(false, false);
	problem_init.SetParameterBlockConstant(frame_param.m_adam_pose.data());
	problem_init.SetParameterBlockConstant(frame_param.m_adam_coeffs.data());

	ceres::Solve(options_init, &problem_init, &summary_init);
	std::cout << summary_init.FullReport() << std::endl;

	// Tranlation, pose and shape, add regularization, use body and hand
	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_coeffs_init = new ceres::AutoDiffCostFunction
		<CoeffsParameterNorm,
		TotalModel::NUM_SHAPE_COEFFICIENTS,
		TotalModel::NUM_SHAPE_COEFFICIENTS>(new CoeffsParameterNorm(TotalModel::NUM_SHAPE_COEFFICIENTS));
	ceres::LossFunction* loss_weight_prior_body_coeffs_init = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(cost_prior_body_coeffs_init,
		loss_weight_prior_body_coeffs_init,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_pose_init = new ceres::AutoDiffCostFunction
		<AdamBodyPoseParamPrior,
		TotalModel::NUM_POSE_PARAMETERS,
		TotalModel::NUM_POSE_PARAMETERS>(new AdamBodyPoseParamPrior(TotalModel::NUM_POSE_PARAMETERS));

	ceres::LossFunction* loss_weight_prior_body_pose_init = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(cost_prior_body_pose_init,
		loss_weight_prior_body_pose_init,
		frame_param.m_adam_pose.data());

	cost_body_keypoints_init->toggle_activate(true, true);
	problem_init.SetParameterBlockVariable(frame_param.m_adam_pose.data());
	problem_init.SetParameterBlockVariable(frame_param.m_adam_coeffs.data());
	ceres::Solve(options_init, &problem_init, &summary_init);
	std::cout << summary_init.FullReport() << std::endl;

	ceres::Problem problem;
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;
	CostFunc_Adam_keypoints_withFoot *cost_body_keypoints = new CostFunc_Adam_keypoints_withFoot(adam, BodyJoints2d, rFoot2d, lFoot2d, faceJoints2d,
																								 lHandJoints2d, rHandJoints2d, calibK, true, 1u);

	problem.AddResidualBlock(cost_body_keypoints,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data(),
		frame_param.m_adam_facecoeffs_exp.data());	

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_coeffs = new ceres::AutoDiffCostFunction
		<CoeffsParameterNorm,
		TotalModel::NUM_SHAPE_COEFFICIENTS,
		TotalModel::NUM_SHAPE_COEFFICIENTS>(new CoeffsParameterNorm(TotalModel::NUM_SHAPE_COEFFICIENTS));
	ceres::LossFunction* loss_weight_prior_body_coeffs = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_coeffs,
		loss_weight_prior_body_coeffs,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_pose = new ceres::AutoDiffCostFunction
		<AdamBodyPoseParamPrior,
		TotalModel::NUM_POSE_PARAMETERS,
		TotalModel::NUM_POSE_PARAMETERS>(new AdamBodyPoseParamPrior(TotalModel::NUM_POSE_PARAMETERS));

	ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_pose,
		loss_weight_prior_body_pose,
		frame_param.m_adam_pose.data());

	// ceres::CostFunction *cost_prior_face_exp = new ceres::NormalPrior(facem.face_prior_A_exp, facem.face_prior_mu_exp);
	ceres::CostFunction *cost_prior_face_exp = new ceres::AutoDiffCostFunction
		<CoeffsParameterNorm,
		TotalModel::NUM_EXP_BASIS_COEFFICIENTS,
		TotalModel::NUM_EXP_BASIS_COEFFICIENTS>(new CoeffsParameterNorm(TotalModel::NUM_EXP_BASIS_COEFFICIENTS));
	ceres::LossFunction *loss_weight_prior_face_exp = new ceres::ScaledLoss(NULL,
		10000,		//original
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_face_exp,
		loss_weight_prior_face_exp,
		frame_param.m_adam_facecoeffs_exp.data());
	
	// problem.SetParameterBlockConstant(frame_param.m_adam_facecoeffs_exp.data());
	options.max_num_iterations = 10;
	options.use_nonmonotonic_steps = false;
	options.num_linear_solver_threads = 10;
	options.minimizer_progress_to_stdout = true;
	cost_body_keypoints->joint_only = false;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;
}

void Adam_FitTotalBodyCeres3d2d(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,	   //2points
	Eigen::MatrixXd &lFoot,		//2points
	Eigen::MatrixXd &rHandJoints,	   //
	Eigen::MatrixXd &lHandJoints,		//
	Eigen::MatrixXd &faceJoints,
	double* calibK)
{
	using namespace Eigen;

	int weight2d = 1.0f;

	ceres::Problem problem_init;
	CostFunc_Adam_keypoints_withFoot *cost_body_keypoints_init = new CostFunc_Adam_keypoints_withFoot(adam, BodyJoints, rFoot, lFoot, faceJoints,
																								 lHandJoints, rHandJoints, calibK, false, 2u);

	problem_init.AddResidualBlock(cost_body_keypoints_init,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());

	ceres::Solver::Options options_init;
	ceres::Solver::Summary summary_init;
	SetSolverOptions(&options_init);
	options_init.function_tolerance = 1e-4;
	options_init.max_num_iterations = 20;
	options_init.use_nonmonotonic_steps = false;
	options_init.num_linear_solver_threads = 10;
	options_init.minimizer_progress_to_stdout = true;
	cost_body_keypoints_init->joint_only = true;
	// Pure Translation, fit body only!
	cost_body_keypoints_init->toggle_activate(true, false);
	cost_body_keypoints_init->weight2d = weight2d;
	problem_init.SetParameterBlockConstant(frame_param.m_adam_pose.data());
	problem_init.SetParameterBlockConstant(frame_param.m_adam_coeffs.data());

	ceres::Solve(options_init, &problem_init, &summary_init);
	std::cout << summary_init.FullReport() << std::endl;

	// Tranlation, pose and shape, add regularization, use body and hand
	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_coeffs_init = new ceres::AutoDiffCostFunction
		<CoeffsParameterNorm,
		TotalModel::NUM_SHAPE_COEFFICIENTS,
		TotalModel::NUM_SHAPE_COEFFICIENTS>(new CoeffsParameterNorm(TotalModel::NUM_SHAPE_COEFFICIENTS));
	ceres::LossFunction* loss_weight_prior_body_coeffs_init = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(cost_prior_body_coeffs_init,
		loss_weight_prior_body_coeffs_init,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_pose_init = new ceres::AutoDiffCostFunction
		<AdamBodyPoseParamPrior,
		TotalModel::NUM_POSE_PARAMETERS,
		TotalModel::NUM_POSE_PARAMETERS>(new AdamBodyPoseParamPrior(TotalModel::NUM_POSE_PARAMETERS));

	ceres::LossFunction* loss_weight_prior_body_pose_init = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(cost_prior_body_pose_init,
		loss_weight_prior_body_pose_init,
		frame_param.m_adam_pose.data());

	cost_body_keypoints_init->toggle_activate(true, true);
	problem_init.SetParameterBlockVariable(frame_param.m_adam_pose.data());
	problem_init.SetParameterBlockVariable(frame_param.m_adam_coeffs.data());
	ceres::Solve(options_init, &problem_init, &summary_init);
	std::cout << summary_init.FullReport() << std::endl;

	ceres::Problem problem;
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;
	CostFunc_Adam_keypoints_withFoot *cost_body_keypoints = new CostFunc_Adam_keypoints_withFoot(adam, BodyJoints, rFoot, lFoot, faceJoints,
																								 lHandJoints, rHandJoints, calibK, true, 2u);

	problem.AddResidualBlock(cost_body_keypoints,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data(),
		frame_param.m_adam_facecoeffs_exp.data());	

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_coeffs = new ceres::AutoDiffCostFunction
		<CoeffsParameterNorm,
		TotalModel::NUM_SHAPE_COEFFICIENTS,
		TotalModel::NUM_SHAPE_COEFFICIENTS>(new CoeffsParameterNorm(TotalModel::NUM_SHAPE_COEFFICIENTS));
	ceres::LossFunction* loss_weight_prior_body_coeffs = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_coeffs,
		loss_weight_prior_body_coeffs,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_pose = new ceres::AutoDiffCostFunction
		<AdamBodyPoseParamPrior,
		TotalModel::NUM_POSE_PARAMETERS,
		TotalModel::NUM_POSE_PARAMETERS>(new AdamBodyPoseParamPrior(TotalModel::NUM_POSE_PARAMETERS));

	ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_pose,
		loss_weight_prior_body_pose,
		frame_param.m_adam_pose.data());

	// ceres::CostFunction *cost_prior_face_exp = new ceres::NormalPrior(facem.face_prior_A_exp, facem.face_prior_mu_exp);
	ceres::CostFunction *cost_prior_face_exp = new ceres::AutoDiffCostFunction
		<CoeffsParameterNorm,
		TotalModel::NUM_EXP_BASIS_COEFFICIENTS,
		TotalModel::NUM_EXP_BASIS_COEFFICIENTS>(new CoeffsParameterNorm(TotalModel::NUM_EXP_BASIS_COEFFICIENTS));
	ceres::LossFunction *loss_weight_prior_face_exp = new ceres::ScaledLoss(NULL,
		10000,		//original
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_face_exp,
		loss_weight_prior_face_exp,
		frame_param.m_adam_facecoeffs_exp.data());
	
	// problem.SetParameterBlockConstant(frame_param.m_adam_facecoeffs_exp.data());
	options.max_num_iterations = 10;
	options.use_nonmonotonic_steps = false;
	options.num_linear_solver_threads = 10;
	options.minimizer_progress_to_stdout = true;
	cost_body_keypoints->joint_only = false;
	cost_body_keypoints->weight2d = weight2d;
	// ceres::Solve(options, &problem, &summary);
	// std::cout << summary.FullReport() << std::endl;
}

ceres::Problem g_problem;
smpl::SMPLParams g_params;
AdamFastCost* g_cost_body_keypoints = NULL;
void Adam_FastFit(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,	   //2points
	Eigen::MatrixXd &lFoot,		//2points
	Eigen::MatrixXd &rHandJoints,	   //
	Eigen::MatrixXd &lHandJoints,		//
	Eigen::MatrixXd &faceJoints)
{
// const auto start1 = std::chrono::high_resolution_clock::now();
	// use the existing shape coeff in frame_param, fit the pose and trans fast
	using namespace Eigen;
	if (g_cost_body_keypoints == NULL)
	{
		g_cost_body_keypoints = new AdamFastCost(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, frame_param.m_adam_coeffs.data());
		g_problem.AddResidualBlock(g_cost_body_keypoints,
			NULL,
			g_params.m_adam_t.data(),
			g_params.m_adam_pose.data()
		);	

		//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
		ceres::CostFunction *cost_prior_body_pose = new ceres::AutoDiffCostFunction
			<AdamBodyPoseParamPrior,
			TotalModel::NUM_POSE_PARAMETERS,
			TotalModel::NUM_POSE_PARAMETERS>(new AdamBodyPoseParamPrior(TotalModel::NUM_POSE_PARAMETERS));

		ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
			1e-2,
			ceres::TAKE_OWNERSHIP);
		g_problem.AddResidualBlock(cost_prior_body_pose,
			loss_weight_prior_body_pose,
			g_params.m_adam_pose.data());

		std::copy(frame_param.m_adam_coeffs.data(), frame_param.m_adam_coeffs.data() + 30, g_params.m_adam_coeffs.data());  // always use the shape coeff of first frame
	}
	else g_cost_body_keypoints->UpdateJoints(BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints);
// const auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start1).count();
// const auto start2 = std::chrono::high_resolution_clock::now();

	std::copy(frame_param.m_adam_t.data(), frame_param.m_adam_t.data() + 3, g_params.m_adam_t.data());
	std::copy(frame_param.m_adam_pose.data(), frame_param.m_adam_pose.data() + 62 * 3, g_params.m_adam_pose.data());
// const auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start2).count();
// const auto start3 = std::chrono::high_resolution_clock::now();

	ceres::Solver::Options options;
	SetSolverOptions(&options);
	options.max_num_iterations = 20;
	options.use_nonmonotonic_steps = false;
	options.num_linear_solver_threads = 10;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;

	// g_problem.SetParameterBlockConstant(g_params.m_adam_t.data());
	// g_problem.SetParameterBlockConstant(g_params.m_adam_pose.data());

	// g_cost_body_keypoints->toggle_activate(false, false);
	// ceres::Solve(options, &g_problem, &summary);
	// std::cout << summary.FullReport() << std::endl;

	// g_cost_body_keypoints->toggle_activate(true, false);
	// ceres::Solve(options, &g_problem, &summary);
	// std::cout << summary.FullReport() << std::endl;

	g_cost_body_keypoints->toggle_activate(true, true);
// const auto duration3 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start3).count();
// const auto start4 = std::chrono::high_resolution_clock::now();
	ceres::Solve(options, &g_problem, &summary);
// const auto duration4 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start4).count();
// const auto start5 = std::chrono::high_resolution_clock::now();
	std::cout << summary.FullReport() << std::endl;

	std::copy(g_params.m_adam_t.data(), g_params.m_adam_t.data() + 3, frame_param.m_adam_t.data());
	std::copy(g_params.m_adam_pose.data(), g_params.m_adam_pose.data() + 62 * 3, frame_param.m_adam_pose.data());
// const auto duration5 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start5).count();
// std::cout << __FILE__ << " " << duration1 * 1e-6 << "\n"
// 		  << __FILE__ << " " << duration2 * 1e-6 << "\n"
// 		  << __FILE__ << " " << duration3 * 1e-6 << "\n"
// 		  << __FILE__ << " " << duration4 * 1e-6 << "\n"
// 		  << __FILE__ << " " << duration5 * 1e-6 << "\n" << std::endl;
}
