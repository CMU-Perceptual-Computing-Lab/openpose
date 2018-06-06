#include "FitToBody.h"
#include "ceres/normal_prior.h"
#include <iostream>
#include "FitCost.h"
#include "AdamFastCost.h"
#include <chrono>
#include "HandFastCost.h"

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
	int regressor_type)
{
	using namespace Eigen;

	ceres::Problem problem;
	ceres::Solver::Options options;

	// Eigen::MatrixXd Joints_part = Joints.block(0, 0, 3, 21);
	// ceres::CostFunction* fit_cost_analytic_ha =
	// 	new ceres::AutoDiffCostFunction<Hand3DCostPose_LBS, smpl::HandModel::NUM_JOINTS * 3, 3, smpl::HandModel::NUM_JOINTS * 3, smpl::HandModel::NUM_SHAPE_COEFFICIENTS>
	// 	(new Hand3DCostPose_LBS(hand_model, 0, Joints_part)) ;
	Eigen::MatrixXd PAF(3, 20); PAF.setZero();
	HandFastCost* fit_cost_analytic_ha = new HandFastCost(hand_model, Joints, PAF, true, false, false, nullptr, regressor_type);
	problem.AddResidualBlock(fit_cost_analytic_ha,
		NULL,
		parm_handl_t.data(),
		parm_hand_pose.data(),
		parm_hand_coeffs.data());

	ceres::CostFunction *hand_pose_reg = new ceres::AutoDiffCostFunction
		<CoeffsParameterNorm,
		smpl::HandModel::NUM_JOINTS * 3,
		smpl::HandModel::NUM_JOINTS * 3>(new CoeffsParameterNorm(smpl::HandModel::NUM_JOINTS * 3));
	ceres::LossFunction* hand_pose_reg_loss = new ceres::ScaledLoss(NULL,
		1e-7 * 1,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(hand_pose_reg,
		hand_pose_reg_loss,
		parm_hand_pose.data());

	// // Regularization
	// ceres::CostFunction *hand_pose_reg = new ceres::AutoDiffCostFunction
	// 	<HandPoseParameterNorm,
	// 	smpl::HandModel::NUM_JOINTS * 3,
	// 	smpl::HandModel::NUM_JOINTS * 3>(new HandPoseParameterNorm(smpl::HandModel::NUM_JOINTS * 3, hand_model));
	// ceres::LossFunction* hand_pose_reg_loss = new ceres::ScaledLoss(NULL,
	// 	1e-8,
	// 	ceres::TAKE_OWNERSHIP);
	// problem.AddResidualBlock(hand_pose_reg,
	// 	hand_pose_reg_loss,
	// 	parm_hand_pose.data());

	// Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cov(smpl::HandModel::NUM_JOINTS * 3, smpl::HandModel::NUM_JOINTS * 3);
	// cov.setIdentity();
	// Eigen::Matrix<double, Eigen::Dynamic, 1> ones(smpl::HandModel::NUM_JOINTS * 3, 1);
	// ones.setOnes();
	// ceres::CostFunction *hand_coeff_reg = new ceres::NormalPrior(cov, ones);
	// ceres::LossFunction *hand_coeff_reg_loss = new ceres::ScaledLoss(NULL,
	// 	1e-9,
	// 	ceres::TAKE_OWNERSHIP);
	// problem.AddResidualBlock(hand_coeff_reg,
	// 	hand_coeff_reg_loss,
	// 	parm_hand_coeffs.data());
	// ceres::CostFunction *hand_coeff_reg = new ceres::AutoDiffCostFunction
	// 	<CoeffsParameterNorm,
	// 	smpl::HandModel::NUM_JOINTS * 3,
	// 	smpl::HandModel::NUM_JOINTS * 3>(new CoeffsParameterNorm(smpl::HandModel::NUM_JOINTS * 3));
	// ceres::LossFunction* hand_coeff_reg_loss = new ceres::ScaledLoss(NULL,
	// 	1e-4,
	// 	ceres::TAKE_OWNERSHIP);
	// problem.AddResidualBlock(hand_coeff_reg,
	// 	hand_coeff_reg_loss,
	// 	parm_hand_coeffs.data());

	for (int i = 0; i < smpl::HandModel::NUM_JOINTS; ++i)
	{
		problem.SetParameterLowerBound(parm_hand_coeffs.data(), i * 3 + 0, 0.5);
		problem.SetParameterLowerBound(parm_hand_coeffs.data(), i * 3 + 1, 0.5);
		problem.SetParameterLowerBound(parm_hand_coeffs.data(), i * 3 + 2, 0.5);
	}

	SetSolverOptions(&options);
	// options.function_tolerance = 1e-8;
	// options.max_num_iterations = 50;
	// options.use_nonmonotonic_steps = false;
	// options.num_linear_solver_threads = 10;
	// options.minimizer_progress_to_stdout = true;
	options.update_state_every_iteration = true;
	options.max_num_iterations = 30;
	options.max_solver_time_in_seconds = 8200;
	options.use_nonmonotonic_steps = true;
	options.dynamic_sparsity = true;
	options.min_lm_diagonal = 2e7;
	options.minimizer_progress_to_stdout = true;

	CHECK(StringToLinearSolverType("sparse_normal_cholesky",
		&options.linear_solver_type));
	// std::cout << "Before: coeff:\n" << parm_hand_coeffs << std::endl;
	// std::cout << "Before: pose:\n" << parm_hand_pose << std::endl;
	// std::cout << "Before: trans:\n" << parm_handl_t << std::endl;
	ceres::Solver::Summary summary;

	problem.SetParameterBlockConstant(parm_hand_coeffs.data());
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";
	problem.SetParameterBlockVariable(parm_hand_coeffs.data());
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";
	// std::cout << "After: coeff:\n" << parm_hand_coeffs << std::endl;
	// std::cout << "After: pose:\n" << parm_hand_pose << std::endl;
	// std::cout << "After: trans:\n" << parm_handl_t << std::endl;

	printf("FitToHandCeres_Right_Naive: Done\n");
}

void FitToProjectionCeres(
	smpl::HandModel &hand_model,
	Eigen::MatrixXd &Joints2d,
	const double* K,
	Eigen::MatrixXd &PAF,
	Eigen::Vector3d& parm_handl_t,
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& parm_hand_pose,
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& parm_hand_coeffs,
	int regressor_type
)
{
	using namespace Eigen;
	ceres::Problem problem_init;
	ceres::Solver::Options options_init;
	// define the reprojection error
	HandFastCost* fit_cost_analytic_ha_init = new HandFastCost(hand_model, Joints2d, PAF, false, false, true, nullptr, regressor_type);
	problem_init.AddResidualBlock(fit_cost_analytic_ha_init,
		NULL,
		parm_handl_t.data(),
		parm_hand_pose.data(),
		parm_hand_coeffs.data());

	// Regularization
	ceres::CostFunction *hand_pose_reg_init = new ceres::AutoDiffCostFunction
		<HandPoseParameterNorm,
		smpl::HandModel::NUM_JOINTS * 3,
		smpl::HandModel::NUM_JOINTS * 3>(new HandPoseParameterNorm(smpl::HandModel::NUM_JOINTS * 3, hand_model));
	ceres::LossFunction* hand_pose_reg_loss_init = new ceres::ScaledLoss(NULL,
		1e-5,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(hand_pose_reg_init,
		hand_pose_reg_loss_init,
		parm_hand_pose.data());
	// ceres::CostFunction *hand_coeff_reg_init = new ceres::AutoDiffCostFunction
	// 	<CoeffsParameterLogNorm,
	// 	smpl::HandModel::NUM_JOINTS * 3,
	// 	smpl::HandModel::NUM_JOINTS * 3>(new CoeffsParameterLogNorm(smpl::HandModel::NUM_JOINTS * 3));
	// ceres::LossFunction* hand_coeff_reg_loss_init = new ceres::ScaledLoss(NULL,
	// 	1e-5,
	// 	ceres::TAKE_OWNERSHIP);
	// problem_init.AddResidualBlock(hand_coeff_reg_init,
	// 	hand_coeff_reg_loss_init,
	// 	parm_hand_coeffs.data());
	Eigen::MatrixXd A(smpl::HandModel::NUM_JOINTS * 3, smpl::HandModel::NUM_JOINTS * 3);
	A.setIdentity();
	Eigen::VectorXd b(smpl::HandModel::NUM_JOINTS * 3);
	b.setOnes();
	ceres::CostFunction *hand_coeff_reg_init = new ceres::NormalPrior(A, b);
	ceres::LossFunction* hand_coeff_reg_loss_init = new ceres::ScaledLoss(NULL,
		1000,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(hand_coeff_reg_init,
		hand_coeff_reg_loss_init,
		parm_hand_coeffs.data());


	for (int i = 0; i < smpl::HandModel::NUM_JOINTS; ++i)
	{
		problem_init.SetParameterLowerBound(parm_hand_coeffs.data(), i * 3 + 0, 0.5);
		problem_init.SetParameterLowerBound(parm_hand_coeffs.data(), i * 3 + 1, 0.5);
		problem_init.SetParameterLowerBound(parm_hand_coeffs.data(), i * 3 + 2, 0.5);
		problem_init.SetParameterUpperBound(parm_hand_coeffs.data(), i * 3 + 0, 2);
		problem_init.SetParameterUpperBound(parm_hand_coeffs.data(), i * 3 + 1, 2);
		problem_init.SetParameterUpperBound(parm_hand_coeffs.data(), i * 3 + 2, 2);
	}

	SetSolverOptions(&options_init);
	options_init.function_tolerance = 1e-8;
	options_init.max_num_iterations = 30;
	options_init.use_nonmonotonic_steps = true;
	options_init.num_linear_solver_threads = 10;
	options_init.minimizer_progress_to_stdout = true;

	problem_init.SetParameterBlockConstant(parm_hand_coeffs.data());
	CHECK(StringToLinearSolverType("sparse_normal_cholesky",
		&options_init.linear_solver_type));
	ceres::Solver::Summary summary_init;
	ceres::Solve(options_init, &problem_init, &summary_init);
	std::cout << summary_init.FullReport() << "\n";
	problem_init.SetParameterBlockVariable(parm_hand_coeffs.data());

	ceres::Problem problem;
	ceres::Solver::Options options;
	HandFastCost* fit_cost_analytic_ha = new HandFastCost(hand_model, Joints2d, PAF, false, true, true, K, regressor_type);
	fit_cost_analytic_ha->weight_PAF = 50.0f;
	problem.AddResidualBlock(fit_cost_analytic_ha,
		NULL,
		parm_handl_t.data(),
		parm_hand_pose.data(),
		parm_hand_coeffs.data());

	// Regularization
	ceres::CostFunction *hand_pose_reg = new ceres::AutoDiffCostFunction
		<HandPoseParameterNorm,
		smpl::HandModel::NUM_JOINTS * 3,
		smpl::HandModel::NUM_JOINTS * 3>(new HandPoseParameterNorm(smpl::HandModel::NUM_JOINTS * 3, hand_model));
	ceres::LossFunction* hand_pose_reg_loss = new ceres::ScaledLoss(NULL,
		1e-5,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(hand_pose_reg,
		hand_pose_reg_loss,
		parm_hand_pose.data());
	// ceres::CostFunction *hand_coeff_reg = new ceres::AutoDiffCostFunction
	// 	<CoeffsParameterLogNorm,
	// 	smpl::HandModel::NUM_JOINTS * 3,
	// 	smpl::HandModel::NUM_JOINTS * 3>(new CoeffsParameterLogNorm(smpl::HandModel::NUM_JOINTS * 3));
	// ceres::LossFunction* hand_coeff_reg_loss = new ceres::ScaledLoss(NULL,
	// Eigen::MatrixXd A(smpl::HandModel::NUM_JOINTS * 3, smpl::HandModel::NUM_JOINTS * 3);
	// A.setIdentity();
	// Eigen::VectorXd b(smpl::HandModel::NUM_JOINTS * 3);
	// b.setOnes();
	ceres::CostFunction *hand_coeff_reg = new ceres::NormalPrior(A, b);
	ceres::LossFunction* hand_coeff_reg_loss = new ceres::ScaledLoss(NULL,
		1000,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(hand_coeff_reg,
		hand_coeff_reg_loss,
		parm_hand_coeffs.data());

	for (int i = 0; i < smpl::HandModel::NUM_JOINTS; ++i)
	{
		problem.SetParameterLowerBound(parm_hand_coeffs.data(), i * 3 + 0, 0.5);
		problem.SetParameterLowerBound(parm_hand_coeffs.data(), i * 3 + 1, 0.5);
		problem.SetParameterLowerBound(parm_hand_coeffs.data(), i * 3 + 2, 0.5);
		problem.SetParameterUpperBound(parm_hand_coeffs.data(), i * 3 + 0, 2);
		problem.SetParameterUpperBound(parm_hand_coeffs.data(), i * 3 + 1, 2);
		problem.SetParameterUpperBound(parm_hand_coeffs.data(), i * 3 + 2, 2);
	}

	SetSolverOptions(&options);
	options.function_tolerance = 1e-8;
	options.max_num_iterations = 30;
	options.use_nonmonotonic_steps = true;
	options.num_linear_solver_threads = 10;
	options.minimizer_progress_to_stdout = true;

	CHECK(StringToLinearSolverType("sparse_normal_cholesky",
		&options.linear_solver_type));
	ceres::Solver::Summary summary;
	problem.SetParameterBlockConstant(parm_hand_coeffs.data());
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";
	problem.SetParameterBlockVariable(parm_hand_coeffs.data());
	ceres::Solve(options, &problem, &summary);
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
	MatrixXd PAF(3, 54);
	// std::fill(PAF.data(), PAF.data() + PAF.size(), 0);
	const AdamFitData data(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, PAF, true);
	ceres::Problem problem_init;
	AdamFullCost* adam_cost = new AdamFullCost(data);

	problem_init.AddResidualBlock(adam_cost,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());	

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	CoeffsParameterNormDiff* cost_prior_body_coeffs_init = new CoeffsParameterNormDiff(TotalModel::NUM_SHAPE_COEFFICIENTS);
	ceres::LossFunction* loss_weight_prior_body_coeffs_init = new ceres::ScaledLoss(NULL,
		1e-4,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(cost_prior_body_coeffs_init,
		loss_weight_prior_body_coeffs_init,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	AdamBodyPoseParamPriorDiff* cost_prior_body_pose_init = new AdamBodyPoseParamPriorDiff(TotalModel::NUM_POSE_PARAMETERS);
	ceres::LossFunction* loss_weight_prior_body_pose_init = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(cost_prior_body_pose_init,
		loss_weight_prior_body_pose_init,
		frame_param.m_adam_pose.data());

	ceres::Solver::Options options_init;
	ceres::Solver::Summary summary;
	SetSolverOptions(&options_init);
	options_init.max_num_iterations = 20;
	options_init.use_nonmonotonic_steps = false;
	options_init.num_linear_solver_threads = 10;
	options_init.minimizer_progress_to_stdout = true;
	adam_cost->toggle_activate(false, false, false);
	adam_cost->toggle_rigid_body(true);
	ceres::Solve(options_init, &problem_init, &summary);
	std::cout << summary.FullReport() << std::endl;

	adam_cost->toggle_rigid_body(false);
	adam_cost->toggle_activate(true, false, false);
	ceres::Solve(options_init, &problem_init, &summary);
	std::cout << summary.FullReport() << std::endl;

	adam_cost->toggle_activate(true, true, true);
	ceres::Solve(options_init, &problem_init, &summary);
	std::cout << summary.FullReport() << std::endl;
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

void Adam_FastFit_Initialize(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,	   //2points
	Eigen::MatrixXd &lFoot,		//2points
	Eigen::MatrixXd &rHandJoints,	   //
	Eigen::MatrixXd &lHandJoints,		//
	Eigen::MatrixXd &faceJoints)
{
	using namespace Eigen;
	MatrixXd PAF(3, 54);
	// std::fill(PAF.data(), PAF.data() + PAF.size(), 0);
	const AdamFitData data(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, PAF, true);
	ceres::Problem problem_init;
	AdamFullCost* adam_cost = new AdamFullCost(data);

	problem_init.AddResidualBlock(adam_cost,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());	

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	CoeffsParameterNormDiff* cost_prior_body_coeffs_init = new CoeffsParameterNormDiff(TotalModel::NUM_SHAPE_COEFFICIENTS);
	ceres::LossFunction* loss_weight_prior_body_coeffs_init = new ceres::ScaledLoss(NULL,
		1e-4,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(cost_prior_body_coeffs_init,
		loss_weight_prior_body_coeffs_init,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	AdamBodyPoseParamPriorDiff* cost_prior_body_pose_init = new AdamBodyPoseParamPriorDiff(TotalModel::NUM_POSE_PARAMETERS);
	ceres::LossFunction* loss_weight_prior_body_pose_init = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(cost_prior_body_pose_init,
		loss_weight_prior_body_pose_init,
		frame_param.m_adam_pose.data());

	adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 0] = 
	adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 1] = 
	adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 2] = 1.0;
	adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 2] = 0;
	ceres::Solver::Options options_init;
	ceres::Solver::Summary summary;
	SetSolverOptions(&options_init);
	options_init.max_num_iterations = 20;
	options_init.use_nonmonotonic_steps = false;
	options_init.num_linear_solver_threads = 10;
	options_init.minimizer_progress_to_stdout = true;
	adam_cost->toggle_activate(false, false, false);
	adam_cost->toggle_rigid_body(true);
	ceres::Solve(options_init, &problem_init, &summary);
	std::cout << summary.FullReport() << std::endl;

	adam_cost->toggle_rigid_body(false);
	adam_cost->toggle_activate(true, false, false);
	ceres::Solve(options_init, &problem_init, &summary);
	std::cout << summary.FullReport() << std::endl;

	adam_cost->toggle_activate(true, true, true);
	ceres::Solve(options_init, &problem_init, &summary);
	std::cout << summary.FullReport() << std::endl;
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
		// ceres::CostFunction *cost_prior_body_pose = new ceres::AutoDiffCostFunction
		// 	<AdamBodyPoseParamPrior,
		// 	TotalModel::NUM_POSE_PARAMETERS,
		// 	TotalModel::NUM_POSE_PARAMETERS>(new AdamBodyPoseParamPrior(TotalModel::NUM_POSE_PARAMETERS));
		AdamBodyPoseParamPriorDiff *cost_prior_body_pose = new AdamBodyPoseParamPriorDiff(TotalModel::NUM_POSE_PARAMETERS);

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

void Adam_Fit_PAF(TotalModel &adam, smpl::SMPLParams &frame_param, Eigen::MatrixXd &BodyJoints, Eigen::MatrixXd &rFoot, Eigen::MatrixXd &lFoot, Eigen::MatrixXd &rHandJoints,
				  Eigen::MatrixXd &lHandJoints, Eigen::MatrixXd &faceJoints, Eigen::MatrixXd &PAF, double* calibK, uint regressor_type, bool quan, bool fitPAFfirst)
{
	using namespace Eigen;	
const auto start = std::chrono::high_resolution_clock::now();

	if (fitPAFfirst)  // if true, fit onto only PAF first
	{
		std::cout << "Fitting to 3D skeleton as the first step" << std::endl;
		ceres::Problem init_problem;

		AdamFitData data(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, PAF, false, false, nullptr, true);
		AdamFullCost* adam_cost;
		adam_cost = new AdamFullCost(data, regressor_type);

		init_problem.AddResidualBlock(adam_cost,
			NULL,
			frame_param.m_adam_t.data(),
			frame_param.m_adam_pose.data(),
			frame_param.m_adam_coeffs.data());

		// for (int i = 0; i < TotalModel::NUM_POSE_PARAMETERS; i++)
		// {
		// 	init_problem.SetParameterLowerBound(frame_param.m_adam_pose.data(), i, -180);
		// 	init_problem.SetParameterUpperBound(frame_param.m_adam_pose.data(), i, 180);
		// }

		ceres::Solver::Options init_options;
		ceres::Solver::Summary init_summary;
		SetSolverOptions(&init_options);
		init_options.function_tolerance = 1e-4;
		init_options.max_num_iterations = 20;
		init_options.use_nonmonotonic_steps = true;
		init_options.num_linear_solver_threads = 10;
		init_options.minimizer_progress_to_stdout = true;
		// if (quan) init_problem.SetParameterBlockConstant(frame_param.m_adam_coeffs.data());
		adam_cost->toggle_activate(false, false, false);
		adam_cost->toggle_rigid_body(true);

const auto start_solve = std::chrono::high_resolution_clock::now();
		ceres::Solve(init_options, &init_problem, &init_summary);
		std::cout << init_summary.FullReport() << std::endl;

		//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
		CoeffsParameterNormDiff* cost_prior_body_coeffs = new CoeffsParameterNormDiff(TotalModel::NUM_SHAPE_COEFFICIENTS);
		ceres::LossFunction* loss_weight_prior_body_coeffs = new ceres::ScaledLoss(NULL,
			quan? 1e-2 : 1e-2,
			ceres::TAKE_OWNERSHIP);
		init_problem.AddResidualBlock(cost_prior_body_coeffs,
			loss_weight_prior_body_coeffs,
			frame_param.m_adam_coeffs.data());

		//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
		AdamBodyPoseParamPriorDiff* cost_prior_body_pose = new AdamBodyPoseParamPriorDiff(TotalModel::NUM_POSE_PARAMETERS);
		ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
			quan? 1e-2 : 1e-2,
			// 1e-2,
			ceres::TAKE_OWNERSHIP);
		init_problem.AddResidualBlock(cost_prior_body_pose,
			loss_weight_prior_body_pose,
			frame_param.m_adam_pose.data());

		init_options.function_tolerance = 1e-4;
		adam_cost->toggle_activate(true, true, true);
		adam_cost->toggle_rigid_body(false);
		ceres::Solve(init_options, &init_problem, &init_summary);
		std::cout << init_summary.FullReport() << std::endl;

const auto duration_solve = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_solve).count();
std::cout << "3D solve time: " << duration_solve * 1e-6 << "\n";
		frame_param.m_adam_t[2] = 200.0; // for fitting onto projection
	}

	std::cout << "Fitting to 2D skeleton Projection" << std::endl;
	ceres::Problem problem;
	AdamFitData data(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, PAF, false, true, calibK, true);
	AdamFullCost* adam_cost;
	adam_cost = new AdamFullCost(data, regressor_type);

	problem.AddResidualBlock(adam_cost,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());

	ceres::Solver::Options options;
	ceres::Solver::Summary summary;
	SetSolverOptions(&options);
	options.function_tolerance = 1e-4;
	options.max_num_iterations = 30;
	options.use_nonmonotonic_steps = true;
	options.num_linear_solver_threads = 10;
	options.minimizer_progress_to_stdout = true;
	adam_cost->toggle_activate(false, false, false);
	adam_cost->toggle_rigid_body(true);
	if(!fitPAFfirst && !quan)  // if fitPAFfirst, should be the first frame in video, allow shape change
	{
		problem.SetParameterBlockConstant(frame_param.m_adam_coeffs.data());
	}

const auto start_solve = std::chrono::high_resolution_clock::now();
	if(!quan) // for quantitative, don't solve this time, especially for failure cases
	{
		ceres::Solve(options, &problem, &summary);
		std::cout << summary.FullReport() << std::endl;
	}

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	CoeffsParameterNormDiff* cost_prior_body_coeffs = new CoeffsParameterNormDiff(TotalModel::NUM_SHAPE_COEFFICIENTS);
	ceres::LossFunction* loss_weight_prior_body_coeffs = new ceres::ScaledLoss(NULL,
		quan? 1e-5 : 1e-2,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_coeffs,
		loss_weight_prior_body_coeffs,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	AdamBodyPoseParamPriorDiff* cost_prior_body_pose = new AdamBodyPoseParamPriorDiff(TotalModel::NUM_POSE_PARAMETERS);
	ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
		quan? 1e-2 : 1e-2,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_pose,
		loss_weight_prior_body_pose,
		frame_param.m_adam_pose.data());

	if (quan)
	{
		for (int i = 0; i < 12; i++) adam_cost->PAF_weight[i] = 50;
		std::fill(cost_prior_body_pose->weight.data() + 36, cost_prior_body_pose->weight.data() + TotalModel::NUM_POSE_PARAMETERS, 2.0);
	}
	else
	{
		if (regressor_type == 0)
		{
			// setting for make videos
			adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 0] =
			adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 1] =
			adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 2] =
			adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 4] =
			adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 5] =
			adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 6] =
			adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 7] = 1.0;
			for (auto i = 0; i < adam.m_correspond_adam2face70_adamIdx.rows(); i++)  // face starts from 8
				adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 8] = 1.0;
		}
		else if (regressor_type == 2)
		{
			// set weight for all vertices
			for (auto i = 0; i < adam_cost->m_nCorrespond_adam2pts; ++i) adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + i] = 1.0;
		}
		adam_cost->toggle_activate(true, false, false);
		adam_cost->toggle_rigid_body(false);
		ceres::Solve(options, &problem, &summary);
		std::cout << summary.FullReport() << std::endl;
	}

	adam_cost->toggle_activate(true, true, true);
	adam_cost->toggle_rigid_body(false);
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;
const auto duration_solve = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_solve).count();
std::cout << "2D solve time: " << duration_solve * 1e-6 << "\n";

const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
std::cout << "Total fitting time: " << duration * 1e-6 << "\n";
}

void Adam_Fit_H36M(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints)
{
	ceres::Problem init_problem;
	Eigen::MatrixXd faceJoints(5, 70);
	Eigen::MatrixXd lHandJoints(5, 20);
	Eigen::MatrixXd rHandJoints(5, 20);
	Eigen::MatrixXd lFoot(5, 3);
	Eigen::MatrixXd rFoot(5, 3);
	Eigen::MatrixXd PAF(3, 14);
	faceJoints.setZero();
	lHandJoints.setZero();
	rHandJoints.setZero();
	lFoot.setZero();
	rFoot.setZero();
	PAF.setZero();
	AdamFitData data(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, PAF, true, false, nullptr, false);
	AdamFullCost* adam_cost;
	adam_cost = new AdamFullCost(data, 1);

	init_problem.AddResidualBlock(adam_cost,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());

	ceres::Solver::Options init_options;
	ceres::Solver::Summary init_summary;
	SetSolverOptions(&init_options);
	init_options.function_tolerance = 1e-4;
	init_options.max_num_iterations = 30;
	init_options.use_nonmonotonic_steps = true;
	init_options.num_linear_solver_threads = 10;
	init_options.minimizer_progress_to_stdout = true;
	adam_cost->toggle_activate(false, false, false);
	adam_cost->toggle_rigid_body(true);

	ceres::Solve(init_options, &init_problem, &init_summary);
	std::cout << init_summary.FullReport() << std::endl;

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	CoeffsParameterNormDiff* cost_prior_body_coeffs = new CoeffsParameterNormDiff(TotalModel::NUM_SHAPE_COEFFICIENTS);
	ceres::LossFunction* loss_weight_prior_body_coeffs = new ceres::ScaledLoss(NULL,
		1e-5,
		ceres::TAKE_OWNERSHIP);
	init_problem.AddResidualBlock(cost_prior_body_coeffs,
		loss_weight_prior_body_coeffs,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	AdamBodyPoseParamPriorDiff* cost_prior_body_pose = new AdamBodyPoseParamPriorDiff(TotalModel::NUM_POSE_PARAMETERS);
	ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	init_problem.AddResidualBlock(cost_prior_body_pose,
		loss_weight_prior_body_pose,
		frame_param.m_adam_pose.data());

	init_options.function_tolerance = 1e-4;
	adam_cost->toggle_activate(true, false, false);
	adam_cost->toggle_rigid_body(false);
	ceres::Solve(init_options, &init_problem, &init_summary);
	std::cout << init_summary.FullReport() << std::endl;
}

void Adam_skeletal_refit(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,
	Eigen::MatrixXd &lFoot,
	Eigen::MatrixXd &rHandJoints,
	Eigen::MatrixXd &lHandJoints,
	Eigen::MatrixXd &faceJoints,
	Eigen::MatrixXd &PAF,
	uint regressor_type)
{
	std::cout << "3D skeletal refitting" << std::endl;
	AdamFitData data(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, PAF, true);  // fit 3D only
	AdamFullCost* adam_cost;
	adam_cost = new AdamFullCost(data, regressor_type);

	ceres::Problem problem;
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;
	problem.AddResidualBlock(adam_cost,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());

	SetSolverOptions(&options);
	options.function_tolerance = 1e-4;
	options.max_num_iterations = 30;
	options.use_nonmonotonic_steps = true;
	options.num_linear_solver_threads = 10;
	options.minimizer_progress_to_stdout = true;

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	CoeffsParameterNormDiff* cost_prior_body_coeffs = new CoeffsParameterNormDiff(TotalModel::NUM_SHAPE_COEFFICIENTS);
	ceres::LossFunction* loss_weight_prior_body_coeffs = new ceres::ScaledLoss(NULL,
		1e-5,
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

	adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 0] =
	adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 1] =
	adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 2] =
	adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 3] =
	adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 4] =
	adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 5] = 1.0;  // foot vertices

	// problem.SetParameterBlockConstant(frame_param.m_adam_coeffs.data());
	adam_cost->toggle_activate(false, false, false);
	adam_cost->toggle_rigid_body(true);
	ceres::Solve(options, &problem, &summary);
	adam_cost->toggle_activate(true, false, false);
	adam_cost->toggle_rigid_body(false);
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;

	AdamFitData data_new(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, PAF, true);  // fit 3D only
	AdamFullCost* adam_cost_new;
	adam_cost_new = new AdamFullCost(data_new, regressor_type);
	ceres::Problem problem_new;
	ceres::Solver::Options options_new;
	ceres::Solver::Summary summary_new;
	problem_new.AddResidualBlock(adam_cost_new,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());

	adam_cost_new->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 0] =
	adam_cost_new->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 1] =
	adam_cost_new->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 2] =
	adam_cost_new->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 3] =
	adam_cost_new->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 4] =
	adam_cost_new->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 5] = 1.0;  // foot vertices

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	CoeffsParameterNormDiff* cost_prior_body_coeffs_new = new CoeffsParameterNormDiff(TotalModel::NUM_SHAPE_COEFFICIENTS);
	ceres::LossFunction* loss_weight_prior_body_coeffs_new = new ceres::ScaledLoss(NULL,
		1e-5,
		ceres::TAKE_OWNERSHIP);
	problem_new.AddResidualBlock(cost_prior_body_coeffs_new,
		loss_weight_prior_body_coeffs_new,
		frame_param.m_adam_coeffs.data());
	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_pose_new = new ceres::AutoDiffCostFunction
		<AdamBodyPoseParamPrior,
		TotalModel::NUM_POSE_PARAMETERS,
		TotalModel::NUM_POSE_PARAMETERS>(new AdamBodyPoseParamPrior(TotalModel::NUM_POSE_PARAMETERS));
	ceres::LossFunction* loss_weight_prior_body_pose_new = new ceres::ScaledLoss(NULL,
		1e-5,
		ceres::TAKE_OWNERSHIP);
	problem_new.AddResidualBlock(cost_prior_body_pose_new,
		loss_weight_prior_body_pose_new,
		frame_param.m_adam_pose.data());

	SetSolverOptions(&options_new);
	options_new.function_tolerance = 1e-4;
	options_new.max_num_iterations = 30;
	options_new.use_nonmonotonic_steps = true;
	options_new.num_linear_solver_threads = 10;
	options_new.minimizer_progress_to_stdout = true;
	adam_cost_new->toggle_activate(true, true, false);
	ceres::Solve(options_new, &problem_new, &summary_new);
	std::cout << summary_new.FullReport() << std::endl;
	adam_cost_new->toggle_activate(true, true, true);
	ceres::Solve(options_new, &problem_new, &summary_new);
	std::cout << summary_new.FullReport() << std::endl;
}

void Adam_skeletal_init(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,
	Eigen::MatrixXd &lFoot,
	Eigen::MatrixXd &rHandJoints,
	Eigen::MatrixXd &lHandJoints,
	Eigen::MatrixXd &faceJoints,
	Eigen::MatrixXd &PAF,
	uint regressor_type)
{
	std::cout << "3D skeletal refitting" << std::endl;
	AdamFitData data(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, PAF, true);  // fit 3D only
	AdamFullCost* adam_cost;
	adam_cost = new AdamFullCost(data, regressor_type);

	ceres::Problem problem;
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;
	problem.AddResidualBlock(adam_cost,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());

	SetSolverOptions(&options);
	options.function_tolerance = 1e-4;
	options.max_num_iterations = 30;
	options.use_nonmonotonic_steps = false;
	options.num_linear_solver_threads = 10;
	options.minimizer_progress_to_stdout = true;

	adam_cost->toggle_activate(false, false, false);
	adam_cost->toggle_rigid_body(true);
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	CoeffsParameterNormDiff* cost_prior_body_coeffs = new CoeffsParameterNormDiff(TotalModel::NUM_SHAPE_COEFFICIENTS);
	ceres::LossFunction* loss_weight_prior_body_coeffs = new ceres::ScaledLoss(NULL,
		1e-1,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_coeffs,
		loss_weight_prior_body_coeffs,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	AdamBodyPoseParamPriorDiff* cost_prior_body_pose = new AdamBodyPoseParamPriorDiff(TotalModel::NUM_POSE_PARAMETERS);
	ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
		1e-1,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_pose,
		loss_weight_prior_body_pose,
		frame_param.m_adam_pose.data());

	adam_cost->toggle_activate(true, false, false);
	adam_cost->toggle_rigid_body(false);
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;

	// std::fill(adam_cost->m_targetPts_weight.data() + 19, adam_cost->m_targetPts_weight.data() + 59, 5);
	adam_cost->toggle_activate(true, true, false);
	adam_cost->toggle_rigid_body(false);
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;
}