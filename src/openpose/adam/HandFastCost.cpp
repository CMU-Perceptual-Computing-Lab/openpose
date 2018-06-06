#include "HandFastCost.h"
#include "FKDerivative.h"
#define SMPL_VIS_SCALING 100.0f

bool HandFastCost::Evaluate(double const* const* parameters,
    double* residuals,
    double** jacobians) const
{
	using namespace Eigen;
	typedef double T;
	const double* t = parameters[0];
	const double* p_euler = parameters[1];
	const double* c = parameters[2];

	MatrixXdr outJ(smpl::HandModel::NUM_JOINTS, 3);
	MatrixXdr dJdP(smpl::HandModel::NUM_JOINTS * 3, smpl::HandModel::NUM_JOINTS * 3);
	MatrixXdr dJdc(smpl::HandModel::NUM_JOINTS * 3, smpl::HandModel::NUM_JOINTS * 3);
	Map< const Vector3d > t_vec(t);

	// First Step: Compute Current Joint;
	ForwardKinematics(p_euler, c, outJ.data(), jacobians? dJdP.data():nullptr, jacobians? dJdc.data():nullptr);
	outJ.rowwise() += t_vec.transpose();
	MatrixXdr outProj(21, 3);
	const Eigen::Map<const Eigen::Matrix<double, 3, 3>> Km(K_);  // map to a Column Major matrix (equivalent to transpose already)
	if (fit2d_) outProj = (SMPL_VIS_SCALING * outJ) * Km;

	// Second step: Set the residual
	const double* ptarget = HandJoints_.data();
	Map< VectorXd > res(residuals, m_nResiduals);
	for (int i = 0; i < smpl::HandModel::NUM_JOINTS; i++)
	{
		const int idj = handm_.m_jointmap_pm2model(i);
		if (fit3d_)
		{
			if(ptarget[5 * i + 0] == 0.0 && ptarget[5 * i + 1] == 0.0 && ptarget[5 * i + 2] == 0.0)
			{
				residuals[res_dim * idj + 0] = 0.0;
				residuals[res_dim * idj + 1] = 0.0;
				residuals[res_dim * idj + 2] = 0.0;
			}
			else
			{
				residuals[res_dim * idj + 0] = weight_joints[i] * (outJ(idj, 0) - ptarget[5 * i + 0]);
				residuals[res_dim * idj + 1] = weight_joints[i] * (outJ(idj, 1) - ptarget[5 * i + 1]);
				residuals[res_dim * idj + 2] = weight_joints[i] * (outJ(idj, 2) - ptarget[5 * i + 2]);
			}
		}

		if (fit2d_)
		{
			if(ptarget[5 * i + 3] == 0.0 && ptarget[5 * i + 4] == 0.0)
			{
				residuals[res_dim * idj + start_2d_dim + 0] = 0.0;
				residuals[res_dim * idj + start_2d_dim + 1] = 0.0;
			}
			else
			{
				residuals[res_dim * idj + start_2d_dim + 0] = weight_2d * weight_joints[i] * ((outProj(idj, 0) / outProj(idj, 2) - ptarget[5 * i + 3]));
				residuals[res_dim * idj + start_2d_dim + 1] = weight_2d * weight_joints[i] * ((outProj(idj, 1) / outProj(idj, 2) - ptarget[5 * i + 4]));
			}
		}
	}

	if (fitPAF_)
	{
		const int offset = start_PAF;
		for (auto i = 0; i < num_PAF_constraint; i++)
		{
			if (PAF_.col(i).isZero(0))
			{
				residuals[offset + 3 * i + 0] = residuals[offset + 3 * i + 1] = residuals[offset + 3 * i + 2] = 0.0;
                continue;
			}
            const std::array<double, 3> AB{{
                outJ.data()[3 * handm_.m_jointmap_pm2model(PAF_connection[i][3]) + 0] - outJ.data()[3 * handm_.m_jointmap_pm2model(PAF_connection[i][1]) + 0], 
                outJ.data()[3 * handm_.m_jointmap_pm2model(PAF_connection[i][3]) + 1] - outJ.data()[3 * handm_.m_jointmap_pm2model(PAF_connection[i][1]) + 1], 
                outJ.data()[3 * handm_.m_jointmap_pm2model(PAF_connection[i][3]) + 2] - outJ.data()[3 * handm_.m_jointmap_pm2model(PAF_connection[i][1]) + 2], 
            }};
            const auto length = sqrt(AB[0] * AB[0] + AB[1] * AB[1] + AB[2] * AB[2]);
            residuals[offset + 3 * i + 0] = weight_PAF * (AB[0] / length - PAF_(0, i));
            residuals[offset + 3 * i + 1] = weight_PAF * (AB[1] / length - PAF_(1, i));
            residuals[offset + 3 * i + 2] = weight_PAF * (AB[2] / length - PAF_(2, i));
		}
	}

	if (jacobians)
	{
		if (jacobians[0])
		{
			Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdt(jacobians[0], m_nResiduals, 3);
			Matrix<double, Dynamic, Dynamic, RowMajor> dJdt(3, 3);
	        dJdt.setIdentity();
			for (int i = 0; i < smpl::HandModel::NUM_JOINTS; i++)
			{
				const int idj = handm_.m_jointmap_pm2model(i);
				if (fit3d_)
				{
					if(ptarget[5 * i + 0] == 0.0 && ptarget[5 * i + 1] == 0.0 && ptarget[5 * i + 2] == 0.0)	drdt.block(res_dim * idj, 0, 3, 3).setZero();
					else
					{
						drdt.block(res_dim * idj, 0, 3, 3) = weight_joints[i] * Eigen::Matrix<double, 3, 3, RowMajor>::Identity();
					}
				}

				if (fit2d_)
				{
					if(ptarget[5 * i + 3] == 0.0 && ptarget[5 * i + 4] == 0.0) drdt.block(res_dim * idj + start_2d_dim, 0, 2, 3).setZero();
					else projection_Derivative(drdt.data(), dJdt.data(), drdt.cols(), (double*)(outJ.data() + 3 * i), K_, res_dim * idj + start_2d_dim, 0, weight_2d * weight_joints[i]);
				}
			}

            if (fitPAF_)
            {
                std::fill(jacobians[0] + 3 * start_PAF, jacobians[0] + 3 * start_PAF + 9 * num_PAF_constraint, 0);
            }
		}

		if (jacobians[1])
		{
			Map< Matrix<double, Dynamic, Dynamic, RowMajor> > dr_dPose(jacobians[1], m_nResiduals, smpl::HandModel::NUM_JOINTS * 3);
			for (int i = 0; i < smpl::HandModel::NUM_JOINTS; i++)
			{
				const int idj = handm_.m_jointmap_pm2model(i);
				if (fit3d_)
				{
					if(ptarget[5 * i + 0] == 0.0 && ptarget[5 * i + 1] == 0.0 && ptarget[5 * i + 2] == 0.0)	dr_dPose.block(res_dim * idj, 0, 3, smpl::HandModel::NUM_JOINTS * 3).setZero();
					else
					{
						dr_dPose.block(res_dim * idj, 0, 3, smpl::HandModel::NUM_JOINTS * 3) = weight_joints[i] * dJdP.block(3 * idj, 0, 3, smpl::HandModel::NUM_JOINTS * 3);
					}
				}
				if (fit2d_)
				{
					if(ptarget[5 * i + 3] == 0.0 && ptarget[5 * i + 4] == 0.0) dr_dPose.block(res_dim * idj + start_2d_dim, 0, 2, 3 * smpl::HandModel::NUM_JOINTS).setZero();
					else projection_Derivative(dr_dPose.data(), dJdP.data(), dr_dPose.cols(), (double*)(outJ.data() + 3 * idj), K_, res_dim * idj + start_2d_dim, 3 * idj, weight_2d * weight_joints[i]);
				}
			}

			if (fitPAF_)
			{
				const int offset = start_PAF;
                for (auto i = 0; i < num_PAF_constraint; i++)
                {
                    if (PAF_.col(i).isZero(0))
                    {
                        std::fill(dr_dPose.data() + (offset + 3 * i) * smpl::HandModel::NUM_POSE_PARAMETERS,
                                  dr_dPose.data() + (offset + 3 * i + 3) * smpl::HandModel::NUM_POSE_PARAMETERS, 0);
                        continue;
                    }
                    const std::array<double, 3> AB{{
		                outJ.data()[3 * handm_.m_jointmap_pm2model(PAF_connection[i][3]) + 0] - outJ.data()[3 * handm_.m_jointmap_pm2model(PAF_connection[i][1]) + 0], 
		                outJ.data()[3 * handm_.m_jointmap_pm2model(PAF_connection[i][3]) + 1] - outJ.data()[3 * handm_.m_jointmap_pm2model(PAF_connection[i][1]) + 1], 
		                outJ.data()[3 * handm_.m_jointmap_pm2model(PAF_connection[i][3]) + 2] - outJ.data()[3 * handm_.m_jointmap_pm2model(PAF_connection[i][1]) + 2], 
		            }};
		            const auto length2 = AB[0] * AB[0] + AB[1] * AB[1] + AB[2] * AB[2];
		            const auto length = sqrt(AB[0] * AB[0] + AB[1] * AB[1] + AB[2] * AB[2]);
		            const Eigen::Map< const Matrix<double, 3, 1> > AB_vec(AB.data());
                    const Eigen::Matrix<double, 3, 3, RowMajor> dudJ = Eigen::Matrix<double, 3, 3>::Identity() / length - AB_vec * AB_vec.transpose() / length2 / length;

                    std::fill(dr_dPose.data() + (offset + 3 * i) * smpl::HandModel::NUM_POSE_PARAMETERS, dr_dPose.data() + (offset + 3 * i + 3) * smpl::HandModel::NUM_POSE_PARAMETERS, 0);
                    const double* dudJ_data = dudJ.data();
                    double* drdp_row0 = dr_dPose.data() + (offset + 3 * i) * smpl::HandModel::NUM_POSE_PARAMETERS;
                    double* drdp_row1 = dr_dPose.data() + (offset + 3 * i + 1) * smpl::HandModel::NUM_POSE_PARAMETERS;
                    double* drdp_row2 = dr_dPose.data() + (offset + 3 * i + 2) * smpl::HandModel::NUM_POSE_PARAMETERS;
                    {
                        const double* dJdP_row0 = dJdP.data() + (3 * handm_.m_jointmap_pm2model(PAF_connection[i][3])) * smpl::HandModel::NUM_POSE_PARAMETERS;
                        const double* dJdP_row1 = dJdP.data() + (3 * handm_.m_jointmap_pm2model(PAF_connection[i][3]) + 1) * smpl::HandModel::NUM_POSE_PARAMETERS;
                        const double* dJdP_row2 = dJdP.data() + (3 * handm_.m_jointmap_pm2model(PAF_connection[i][3]) + 2) * smpl::HandModel::NUM_POSE_PARAMETERS;
                        for(auto& ipar: parentIndexes[handm_.m_jointmap_pm2model(PAF_connection[i][3])])
                        {
                            drdp_row0[3 * ipar + 0] += weight_PAF * (dudJ_data[0] * dJdP_row0[3 * ipar + 0] + dudJ_data[1] * dJdP_row1[3 * ipar + 0] + dudJ_data[2] * dJdP_row2[3 * ipar + 0]);
                            drdp_row0[3 * ipar + 1] += weight_PAF * (dudJ_data[0] * dJdP_row0[3 * ipar + 1] + dudJ_data[1] * dJdP_row1[3 * ipar + 1] + dudJ_data[2] * dJdP_row2[3 * ipar + 1]);
                            drdp_row0[3 * ipar + 2] += weight_PAF * (dudJ_data[0] * dJdP_row0[3 * ipar + 2] + dudJ_data[1] * dJdP_row1[3 * ipar + 2] + dudJ_data[2] * dJdP_row2[3 * ipar + 2]);
                            drdp_row1[3 * ipar + 0] += weight_PAF * (dudJ_data[3] * dJdP_row0[3 * ipar + 0] + dudJ_data[4] * dJdP_row1[3 * ipar + 0] + dudJ_data[5] * dJdP_row2[3 * ipar + 0]);
                            drdp_row1[3 * ipar + 1] += weight_PAF * (dudJ_data[3] * dJdP_row0[3 * ipar + 1] + dudJ_data[4] * dJdP_row1[3 * ipar + 1] + dudJ_data[5] * dJdP_row2[3 * ipar + 1]);
                            drdp_row1[3 * ipar + 2] += weight_PAF * (dudJ_data[3] * dJdP_row0[3 * ipar + 2] + dudJ_data[4] * dJdP_row1[3 * ipar + 2] + dudJ_data[5] * dJdP_row2[3 * ipar + 2]);
                            drdp_row2[3 * ipar + 0] += weight_PAF * (dudJ_data[6] * dJdP_row0[3 * ipar + 0] + dudJ_data[7] * dJdP_row1[3 * ipar + 0] + dudJ_data[8] * dJdP_row2[3 * ipar + 0]);
                            drdp_row2[3 * ipar + 1] += weight_PAF * (dudJ_data[6] * dJdP_row0[3 * ipar + 1] + dudJ_data[7] * dJdP_row1[3 * ipar + 1] + dudJ_data[8] * dJdP_row2[3 * ipar + 1]);
                            drdp_row2[3 * ipar + 2] += weight_PAF * (dudJ_data[6] * dJdP_row0[3 * ipar + 2] + dudJ_data[7] * dJdP_row1[3 * ipar + 2] + dudJ_data[8] * dJdP_row2[3 * ipar + 2]);
                        }
                    }
                    {
                        const double* dJdP_row0 = dJdP.data() + (3 * handm_.m_jointmap_pm2model(PAF_connection[i][1])) * smpl::HandModel::NUM_POSE_PARAMETERS;
                        const double* dJdP_row1 = dJdP.data() + (3 * handm_.m_jointmap_pm2model(PAF_connection[i][1]) + 1) * smpl::HandModel::NUM_POSE_PARAMETERS;
                        const double* dJdP_row2 = dJdP.data() + (3 * handm_.m_jointmap_pm2model(PAF_connection[i][1]) + 2) * smpl::HandModel::NUM_POSE_PARAMETERS;
                        for(auto& ipar: parentIndexes[handm_.m_jointmap_pm2model(PAF_connection[i][1])])
                        {
                            drdp_row0[3 * ipar + 0] -= weight_PAF * (dudJ_data[0] * dJdP_row0[3 * ipar + 0] + dudJ_data[1] * dJdP_row1[3 * ipar + 0] + dudJ_data[2] * dJdP_row2[3 * ipar + 0]);
                            drdp_row0[3 * ipar + 1] -= weight_PAF * (dudJ_data[0] * dJdP_row0[3 * ipar + 1] + dudJ_data[1] * dJdP_row1[3 * ipar + 1] + dudJ_data[2] * dJdP_row2[3 * ipar + 1]);
                            drdp_row0[3 * ipar + 2] -= weight_PAF * (dudJ_data[0] * dJdP_row0[3 * ipar + 2] + dudJ_data[1] * dJdP_row1[3 * ipar + 2] + dudJ_data[2] * dJdP_row2[3 * ipar + 2]);
                            drdp_row1[3 * ipar + 0] -= weight_PAF * (dudJ_data[3] * dJdP_row0[3 * ipar + 0] + dudJ_data[4] * dJdP_row1[3 * ipar + 0] + dudJ_data[5] * dJdP_row2[3 * ipar + 0]);
                            drdp_row1[3 * ipar + 1] -= weight_PAF * (dudJ_data[3] * dJdP_row0[3 * ipar + 1] + dudJ_data[4] * dJdP_row1[3 * ipar + 1] + dudJ_data[5] * dJdP_row2[3 * ipar + 1]);
                            drdp_row1[3 * ipar + 2] -= weight_PAF * (dudJ_data[3] * dJdP_row0[3 * ipar + 2] + dudJ_data[4] * dJdP_row1[3 * ipar + 2] + dudJ_data[5] * dJdP_row2[3 * ipar + 2]);
                            drdp_row2[3 * ipar + 0] -= weight_PAF * (dudJ_data[6] * dJdP_row0[3 * ipar + 0] + dudJ_data[7] * dJdP_row1[3 * ipar + 0] + dudJ_data[8] * dJdP_row2[3 * ipar + 0]);
                            drdp_row2[3 * ipar + 1] -= weight_PAF * (dudJ_data[6] * dJdP_row0[3 * ipar + 1] + dudJ_data[7] * dJdP_row1[3 * ipar + 1] + dudJ_data[8] * dJdP_row2[3 * ipar + 1]);
                            drdp_row2[3 * ipar + 2] -= weight_PAF * (dudJ_data[6] * dJdP_row0[3 * ipar + 2] + dudJ_data[7] * dJdP_row1[3 * ipar + 2] + dudJ_data[8] * dJdP_row2[3 * ipar + 2]);
                        }
                    }
                }
			}
		}

		if (jacobians[2])
		{
			Map< Matrix<double, Dynamic, Dynamic, RowMajor> > dr_dCoeff(jacobians[2], m_nResiduals, smpl::HandModel::NUM_JOINTS * 3);
			for (int i = 0; i < smpl::HandModel::NUM_JOINTS; i++)
			{
				const int idj = handm_.m_jointmap_pm2model(i);
				if (fit3d_)
				{
					if(ptarget[5 * i + 0] == 0.0 && ptarget[5 * i + 1] == 0.0 && ptarget[5 * i + 2] == 0.0)	dr_dCoeff.block(res_dim * idj, 0, 3, smpl::HandModel::NUM_JOINTS * 3).setZero();
					else
					{
						dr_dCoeff.block(res_dim * idj, 0, 3, smpl::HandModel::NUM_JOINTS * 3) = weight_joints[i] * dJdc.block(3 * idj, 0, 3, smpl::HandModel::NUM_JOINTS * 3);
					}
				}
				if (fit2d_)
				{
					if(ptarget[5 * i + 3] == 0.0 && ptarget[5 * i + 4] == 0.0) dr_dCoeff.block(res_dim * idj + start_2d_dim, 0, 2, 3 * smpl::HandModel::NUM_JOINTS).setZero();
					else projection_Derivative(dr_dCoeff.data(), dJdc.data(), dr_dCoeff.cols(), (double*)(outJ.data() + 3 * idj), K_, res_dim * idj + start_2d_dim, 3 * idj, weight_2d * weight_joints[i]);
				}
				if (fitPAF_)
				{
					const int offset = start_PAF;
	                for (auto i = 0; i < num_PAF_constraint; i++)
	                {
	                    if (PAF_.col(i).isZero(0))
	                    {
	                        std::fill(dr_dCoeff.data() + (offset + 3 * i) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS,
	                                  dr_dCoeff.data() + (offset + 3 * i + 3) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS, 0);
	                        continue;
	                    }
	                    const std::array<double, 3> AB{{
			                outJ.data()[3 * handm_.m_jointmap_pm2model(PAF_connection[i][3]) + 0] - outJ.data()[3 * handm_.m_jointmap_pm2model(PAF_connection[i][1]) + 0], 
			                outJ.data()[3 * handm_.m_jointmap_pm2model(PAF_connection[i][3]) + 1] - outJ.data()[3 * handm_.m_jointmap_pm2model(PAF_connection[i][1]) + 1], 
			                outJ.data()[3 * handm_.m_jointmap_pm2model(PAF_connection[i][3]) + 2] - outJ.data()[3 * handm_.m_jointmap_pm2model(PAF_connection[i][1]) + 2], 
			            }};
			            const auto length2 = AB[0] * AB[0] + AB[1] * AB[1] + AB[2] * AB[2];
			            const auto length = sqrt(AB[0] * AB[0] + AB[1] * AB[1] + AB[2] * AB[2]);
			            const Eigen::Map< const Matrix<double, 3, 1> > AB_vec(AB.data());
	                    const Eigen::Matrix<double, 3, 3, RowMajor> dudJ = Eigen::Matrix<double, 3, 3>::Identity() / length - AB_vec * AB_vec.transpose() / length2 / length;

	                    std::fill(dr_dCoeff.data() + (offset + 3 * i) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS, dr_dCoeff.data() + (offset + 3 * i + 3) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS, 0);
	                    const double* dudJ_data = dudJ.data();

	                    double* drdc_row0 = dr_dCoeff.data() + (offset + 3 * i) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS;
	                    double* drdc_row1 = dr_dCoeff.data() + (offset + 3 * i + 1) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS;
	                    double* drdc_row2 = dr_dCoeff.data() + (offset + 3 * i + 2) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS;
	                    {
	                        const double* dJdc_row0 = dJdc.data() + (3 * handm_.m_jointmap_pm2model(PAF_connection[i][3])) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS;
	                        const double* dJdc_row1 = dJdc.data() + (3 * handm_.m_jointmap_pm2model(PAF_connection[i][3]) + 1) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS;
	                        const double* dJdc_row2 = dJdc.data() + (3 * handm_.m_jointmap_pm2model(PAF_connection[i][3]) + 2) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS;
	                        for(auto& ipar: parentIndexes[handm_.m_jointmap_pm2model(PAF_connection[i][3])])
	                        {
	                            drdc_row0[3 * ipar + 0] += weight_PAF * (dudJ_data[0] * dJdc_row0[3 * ipar + 0] + dudJ_data[1] * dJdc_row1[3 * ipar + 0] + dudJ_data[2] * dJdc_row2[3 * ipar + 0]);
	                            drdc_row0[3 * ipar + 1] += weight_PAF * (dudJ_data[0] * dJdc_row0[3 * ipar + 1] + dudJ_data[1] * dJdc_row1[3 * ipar + 1] + dudJ_data[2] * dJdc_row2[3 * ipar + 1]);
	                            drdc_row0[3 * ipar + 2] += weight_PAF * (dudJ_data[0] * dJdc_row0[3 * ipar + 2] + dudJ_data[1] * dJdc_row1[3 * ipar + 2] + dudJ_data[2] * dJdc_row2[3 * ipar + 2]);
	                            drdc_row1[3 * ipar + 0] += weight_PAF * (dudJ_data[3] * dJdc_row0[3 * ipar + 0] + dudJ_data[4] * dJdc_row1[3 * ipar + 0] + dudJ_data[5] * dJdc_row2[3 * ipar + 0]);
	                            drdc_row1[3 * ipar + 1] += weight_PAF * (dudJ_data[3] * dJdc_row0[3 * ipar + 1] + dudJ_data[4] * dJdc_row1[3 * ipar + 1] + dudJ_data[5] * dJdc_row2[3 * ipar + 1]);
	                            drdc_row1[3 * ipar + 2] += weight_PAF * (dudJ_data[3] * dJdc_row0[3 * ipar + 2] + dudJ_data[4] * dJdc_row1[3 * ipar + 2] + dudJ_data[5] * dJdc_row2[3 * ipar + 2]);
	                            drdc_row2[3 * ipar + 0] += weight_PAF * (dudJ_data[6] * dJdc_row0[3 * ipar + 0] + dudJ_data[7] * dJdc_row1[3 * ipar + 0] + dudJ_data[8] * dJdc_row2[3 * ipar + 0]);
	                            drdc_row2[3 * ipar + 1] += weight_PAF * (dudJ_data[6] * dJdc_row0[3 * ipar + 1] + dudJ_data[7] * dJdc_row1[3 * ipar + 1] + dudJ_data[8] * dJdc_row2[3 * ipar + 1]);
	                            drdc_row2[3 * ipar + 2] += weight_PAF * (dudJ_data[6] * dJdc_row0[3 * ipar + 2] + dudJ_data[7] * dJdc_row1[3 * ipar + 2] + dudJ_data[8] * dJdc_row2[3 * ipar + 2]);
	                        }
	                    }
	                    {
	                        const double* dJdc_row0 = dJdc.data() + (3 * handm_.m_jointmap_pm2model(PAF_connection[i][3])) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS;
	                        const double* dJdc_row1 = dJdc.data() + (3 * handm_.m_jointmap_pm2model(PAF_connection[i][3]) + 1) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS;
	                        const double* dJdc_row2 = dJdc.data() + (3 * handm_.m_jointmap_pm2model(PAF_connection[i][3]) + 2) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS;
	                        for(auto& ipar: parentIndexes[handm_.m_jointmap_pm2model(PAF_connection[i][1])])
	                        {
	                            drdc_row0[3 * ipar + 0] -= weight_PAF * (dudJ_data[0] * dJdc_row0[3 * ipar + 0] + dudJ_data[1] * dJdc_row1[3 * ipar + 0] + dudJ_data[2] * dJdc_row2[3 * ipar + 0]);
	                            drdc_row0[3 * ipar + 1] -= weight_PAF * (dudJ_data[0] * dJdc_row0[3 * ipar + 1] + dudJ_data[1] * dJdc_row1[3 * ipar + 1] + dudJ_data[2] * dJdc_row2[3 * ipar + 1]);
	                            drdc_row0[3 * ipar + 2] -= weight_PAF * (dudJ_data[0] * dJdc_row0[3 * ipar + 2] + dudJ_data[1] * dJdc_row1[3 * ipar + 2] + dudJ_data[2] * dJdc_row2[3 * ipar + 2]);
	                            drdc_row1[3 * ipar + 0] -= weight_PAF * (dudJ_data[3] * dJdc_row0[3 * ipar + 0] + dudJ_data[4] * dJdc_row1[3 * ipar + 0] + dudJ_data[5] * dJdc_row2[3 * ipar + 0]);
	                            drdc_row1[3 * ipar + 1] -= weight_PAF * (dudJ_data[3] * dJdc_row0[3 * ipar + 1] + dudJ_data[4] * dJdc_row1[3 * ipar + 1] + dudJ_data[5] * dJdc_row2[3 * ipar + 1]);
	                            drdc_row1[3 * ipar + 2] -= weight_PAF * (dudJ_data[3] * dJdc_row0[3 * ipar + 2] + dudJ_data[4] * dJdc_row1[3 * ipar + 2] + dudJ_data[5] * dJdc_row2[3 * ipar + 2]);
	                            drdc_row2[3 * ipar + 0] -= weight_PAF * (dudJ_data[6] * dJdc_row0[3 * ipar + 0] + dudJ_data[7] * dJdc_row1[3 * ipar + 0] + dudJ_data[8] * dJdc_row2[3 * ipar + 0]);
	                            drdc_row2[3 * ipar + 1] -= weight_PAF * (dudJ_data[6] * dJdc_row0[3 * ipar + 1] + dudJ_data[7] * dJdc_row1[3 * ipar + 1] + dudJ_data[8] * dJdc_row2[3 * ipar + 1]);
	                            drdc_row2[3 * ipar + 2] -= weight_PAF * (dudJ_data[6] * dJdc_row0[3 * ipar + 2] + dudJ_data[7] * dJdc_row1[3 * ipar + 2] + dudJ_data[8] * dJdc_row2[3 * ipar + 2]);
	                        }
	                    }
	                }
	            }
			}
		}
	}
	return true;
}

void HandFastCost::ForwardKinematics(double const* p_euler, double const* c, double* J_data, double* dJdP_data, double* dJdc_data) const
{
	using namespace Eigen;
	Eigen::Map< Matrix<double, smpl::HandModel::NUM_JOINTS, 3, RowMajor> > outJ(J_data);
	Map< Matrix<double, smpl::HandModel::NUM_JOINTS * 3, smpl::HandModel::NUM_JOINTS * 3, RowMajor> > dJdP(dJdP_data);
	Map< Matrix<double, smpl::HandModel::NUM_JOINTS * 3, smpl::HandModel::NUM_JOINTS * 3, RowMajor> > dJdc(dJdc_data);
	Matrix<double, 3, 3, RowMajor> R1; // Interface with ceres (Angle Axis Output)
	Matrix<double, 3, 3, RowMajor> S; // Scaling diagonal
	Matrix<double, 3, 3, RowMajor> R2; // R1 * S
	Matrix<double, 3, 3, RowMajor> R3; // m_M_l2pl
	Matrix<double, 3, 3, RowMajor> R4; // R3 * R2;

	// some buffer
    Matrix<double, 9, 3 * smpl::HandModel::NUM_JOINTS, RowMajor> dR1dP(9, 3 * smpl::HandModel::NUM_JOINTS);
    Matrix<double, 9, 3 * smpl::HandModel::NUM_JOINTS, RowMajor> dR1dc(9, 3 * smpl::HandModel::NUM_JOINTS);
    Matrix<double, 9, 3 * smpl::HandModel::NUM_JOINTS, RowMajor> dR2dP(9, 3 * smpl::HandModel::NUM_JOINTS);
    Matrix<double, 9, 3 * smpl::HandModel::NUM_JOINTS, RowMajor> dR2dc(9, 3 * smpl::HandModel::NUM_JOINTS);
    Matrix<double, 9, 3 * smpl::HandModel::NUM_JOINTS, RowMajor> dR3dP(9, 3 * smpl::HandModel::NUM_JOINTS);
    Matrix<double, 9, 3 * smpl::HandModel::NUM_JOINTS, RowMajor> dR3dc(9, 3 * smpl::HandModel::NUM_JOINTS);
    Matrix<double, 9, 3 * smpl::HandModel::NUM_JOINTS, RowMajor> dR4dP(9, 3 * smpl::HandModel::NUM_JOINTS);
    Matrix<double, 9, 3 * smpl::HandModel::NUM_JOINTS, RowMajor> dR4dc(9, 3 * smpl::HandModel::NUM_JOINTS);
    Matrix<double, 9, 3 * smpl::HandModel::NUM_JOINTS, RowMajor> dSdP(9, 3 * smpl::HandModel::NUM_JOINTS);
    Matrix<double, 9, 3 * smpl::HandModel::NUM_JOINTS, RowMajor> dSdc(9, 3 * smpl::HandModel::NUM_JOINTS);
    Matrix<double, 3, 1> offset; // a buffer for 3D vector
    Matrix<double, 3, 3 * smpl::HandModel::NUM_JOINTS, Eigen::RowMajor> dtdP(3, 3 * smpl::HandModel::NUM_JOINTS); // a buffer for the derivative
    Matrix<double, 3, 3 * smpl::HandModel::NUM_JOINTS, Eigen::RowMajor> dtdc(3, 3 * smpl::HandModel::NUM_JOINTS); // a buffer for the derivative

    std::vector<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> MR(smpl::HandModel::NUM_JOINTS, Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(3, 3));
    std::vector<Eigen::Matrix<double, 3, 1>> Mt(smpl::HandModel::NUM_JOINTS, Eigen::Matrix<double, 3, 1>(3, 1));

    std::vector<Eigen::Matrix<double, 9, 3 * smpl::HandModel::NUM_JOINTS, Eigen::RowMajor>> dMRdP(smpl::HandModel::NUM_JOINTS, Eigen::Matrix<double, 9, 3 * smpl::HandModel::NUM_JOINTS, Eigen::RowMajor>(9, 3 * smpl::HandModel::NUM_JOINTS));
    std::vector<Eigen::Matrix<double, 9, 3 * smpl::HandModel::NUM_JOINTS, Eigen::RowMajor>> dMRdc(smpl::HandModel::NUM_JOINTS, Eigen::Matrix<double, 9, 3 * smpl::HandModel::NUM_JOINTS, Eigen::RowMajor>(9, 3 * smpl::HandModel::NUM_JOINTS));
    std::vector<Eigen::Matrix<double, 3, 3 * smpl::HandModel::NUM_JOINTS, Eigen::RowMajor>> dMtdP(smpl::HandModel::NUM_JOINTS, Eigen::Matrix<double, 3, 3 * smpl::HandModel::NUM_JOINTS, Eigen::RowMajor>(3, 3 * smpl::HandModel::NUM_JOINTS));
    std::vector<Eigen::Matrix<double, 3, 3 * smpl::HandModel::NUM_JOINTS, Eigen::RowMajor>> dMtdc(smpl::HandModel::NUM_JOINTS, Eigen::Matrix<double, 3, 3 * smpl::HandModel::NUM_JOINTS, Eigen::RowMajor>(3, 3 * smpl::HandModel::NUM_JOINTS));

    int idj = handm_.update_inds_(0);
    ceres::AngleAxisToRotationMatrix(p_euler + idj * 3, R1.data());
    // S = Eigen::DiagonalMatrix<double, 3>(exp(c[3 * idj]), exp(c[3 * idj]), exp(c[3 * idj]));
    S = Eigen::DiagonalMatrix<double, 3>(c[3 * idj], c[3 * idj], c[3 * idj]);
    R2 = R1 * S;
    R3 = handm_.m_M_l2pl.block(idj * 4, 0, 3, 3);
    offset = handm_.m_M_l2pl.block(idj * 4, 3, 3, 1);
    MR.at(idj) = R3 * R2;  // no parent
    Mt.at(idj) = offset;
    outJ.row(idj) = Mt.at(idj).transpose();

    if (dJdP_data && dJdc_data)
    {
	    AngleAxisToRotationMatrix_Derivative(p_euler + 3 * idj, dR1dP.data(), idj, 3 * smpl::HandModel::NUM_JOINTS);
	    std::fill(dR1dc.data(), dR1dc.data() + 9 * 3 * smpl::HandModel::NUM_JOINTS, 0.0);
	    std::fill(dSdP.data(), dSdP.data() + 9 * 3 * smpl::HandModel::NUM_JOINTS, 0.0);
	    std::fill(dSdc.data(), dSdc.data() + 9 * 3 * smpl::HandModel::NUM_JOINTS, 0.0);
	    // dSdc.data()[3 * idj] = exp(c[3 * idj]);
	    // dSdc.data()[3 * idj + 4 * 3 * smpl::HandModel::NUM_JOINTS] = exp(c[3 * idj]);
	    // dSdc.data()[3 * idj + 8 * 3 * smpl::HandModel::NUM_JOINTS] = exp(c[3 * idj]);
	    dSdc.data()[3 * idj] = 1;
	    dSdc.data()[3 * idj + 4 * 3 * smpl::HandModel::NUM_JOINTS] = 1;
	    dSdc.data()[3 * idj + 8 * 3 * smpl::HandModel::NUM_JOINTS] = 1;
	    SparseProductDerivative(R1.data(), dR1dP.data(), S.data(), dSdP.data(), idj, parentIndexes[idj], dR2dP.data(), 3 * smpl::HandModel::NUM_JOINTS);
	    SparseProductDerivative(R1.data(), dR1dc.data(), S.data(), dSdc.data(), idj, parentIndexes[idj], dR2dc.data(), 3 * smpl::HandModel::NUM_JOINTS);
	    std::fill(dR3dP.data(), dR3dP.data() + 9 * 3 * smpl::HandModel::NUM_JOINTS, 0.0);  // R3 is precomputed
	    std::fill(dR3dc.data(), dR3dc.data() + 9 * 3 * smpl::HandModel::NUM_JOINTS, 0.0);  // R3 is precomputed
	    SparseProductDerivative(R3.data(), dR3dP.data(), R2.data(), dR2dP.data(), idj, parentIndexes[idj], dMRdP[idj].data(), 3 * smpl::HandModel::NUM_JOINTS);
	    SparseProductDerivative(R3.data(), dR3dc.data(), R2.data(), dR2dc.data(), idj, parentIndexes[idj], dMRdc[idj].data(), 3 * smpl::HandModel::NUM_JOINTS);
	    std::fill(dMtdP[idj].data(), dMtdP[idj].data() + 3 * 3 * smpl::HandModel::NUM_JOINTS, 0.0);
	    std::fill(dMtdc[idj].data(), dMtdc[idj].data() + 3 * 3 * smpl::HandModel::NUM_JOINTS, 0.0);
	    std::copy(dMtdP[idj].data(), dMtdP[idj].data() + 3 * 3 * smpl::HandModel::NUM_JOINTS, dJdP_data + 3 * smpl::HandModel::NUM_JOINTS * 3 * idj);
	    std::copy(dMtdc[idj].data(), dMtdc[idj].data() + 3 * 3 * smpl::HandModel::NUM_JOINTS, dJdc_data + 3 * smpl::HandModel::NUM_JOINTS * 3 * idj);
	    // set back the dSdc value (no need to reset)
	    dSdc.data()[3 * idj] = 0;
	    dSdc.data()[3 * idj + 4 * 3 * smpl::HandModel::NUM_JOINTS] = 0;
	    dSdc.data()[3 * idj + 8 * 3 * smpl::HandModel::NUM_JOINTS] = 0;
	}

	for (int idji = 1; idji < handm_.NUM_JOINTS; idji++)
	{
		idj = handm_.update_inds_(idji);
		int ipar = handm_.parents_(idj);
		double angles[3];
		// if (idj == 4 || idj == 6 || idj == 8 || idj == 10)
		// {
		// 	angles[0] = 0;
		// 	angles[1] = p_euler[idj * 3 + 1];
		// 	angles[2] = p_euler[idj * 3 + 2];
		// }
		// else if (idj == 2 || idj == 12 || idj == 13 || idj == 14 || idj == 15 || idj == 16 || idj == 17 || idj == 18 || idj == 19)		//No twist, no side directional
		// {
		// 	angles[0] = 0;
		// 	angles[1] = 0;
		// 	angles[2] = p_euler[idj * 3 + 2];
		// }
		// else
		// {
		// 	angles[0] = p_euler[idj * 3 + 0];
		// 	angles[1] = p_euler[idj * 3 + 1];
		// 	angles[2] = p_euler[idj * 3 + 2];
		// }
		angles[0] = p_euler[idj * 3 + 0];
		angles[1] = p_euler[idj * 3 + 1];
		angles[2] = p_euler[idj * 3 + 2];
		ceres::EulerAnglesToRotationMatrix(angles, 3, R1.data());
		// S = Eigen::DiagonalMatrix<double, 3>(exp(c[3 * idj]), 1, 1);
		S = Eigen::DiagonalMatrix<double, 3>(c[3 * idj], 1, 1);
		R2 = R1 * S;
		R3 = handm_.m_M_l2pl.block(idj * 4, 0, 3, 3);
		offset = handm_.m_M_l2pl.block(idj * 4, 3, 3, 1);
		R4 = R3 * R2;
		MR.at(idj) = MR.at(ipar) * R4;
		Mt.at(idj) = MR.at(ipar) * offset + Mt.at(ipar);

	    if (dJdP_data && dJdc_data)
	    {
		    EulerAnglesToRotationMatrix_Derivative(angles, dR1dP.data(), idj, 3 * smpl::HandModel::NUM_JOINTS);
   //  		if (idj == 4 || idj == 6 || idj == 8 || idj == 10)
			// {
			// 	dR1dP.block(0, 3 * idj, 9, 1).setZero();
			// }
			// else if (idj == 2 || idj == 12 || idj == 13 || idj == 14 || idj == 15 || idj == 16 || idj == 17 || idj == 18 || idj == 19)		//No twist, no side directional
			// {
			// 	dR1dP.block(0, 3 * idj, 9, 2).setZero();
			// }
		    // no need to reset dR1dc
		    // no need to reset dSdP
		    // dSdc.data()[3 * idj] = exp(c[3 * idj]);
		    dSdc.data()[3 * idj] = 1;
		    SparseProductDerivative(R1.data(), dR1dP.data(), S.data(), dSdP.data(), idj, parentIndexes[idj], dR2dP.data(), 3 * smpl::HandModel::NUM_JOINTS);
		    SparseProductDerivative(R1.data(), dR1dc.data(), S.data(), dSdc.data(), idj, parentIndexes[idj], dR2dc.data(), 3 * smpl::HandModel::NUM_JOINTS);
		    // no need to reset dR3dP
		    // no need to reset dR3dc
		    SparseProductDerivative(R3.data(), dR3dP.data(), R2.data(), dR2dP.data(), idj, parentIndexes[idj], dR4dP.data(), 3 * smpl::HandModel::NUM_JOINTS);
		    SparseProductDerivative(R3.data(), dR3dc.data(), R2.data(), dR2dc.data(), idj, parentIndexes[idj], dR4dc.data(), 3 * smpl::HandModel::NUM_JOINTS);
		    SparseProductDerivative(MR.at(ipar).data(), dMRdP[ipar].data(), R4.data(), dR4dP.data(), idj, parentIndexes[idj], dMRdP[idj].data(), 3 * smpl::HandModel::NUM_JOINTS);
		    SparseProductDerivative(MR.at(ipar).data(), dMRdc[ipar].data(), R4.data(), dR4dc.data(), idj, parentIndexes[idj], dMRdc[idj].data(), 3 * smpl::HandModel::NUM_JOINTS);
		    SparseProductDerivative(dMRdP[ipar].data(), offset.data(), parentIndexes[idj], dMtdP[idj].data(), 3 * smpl::HandModel::NUM_JOINTS);
		    SparseProductDerivative(dMRdc[ipar].data(), offset.data(), parentIndexes[idj], dMtdc[idj].data(), 3 * smpl::HandModel::NUM_JOINTS);
		    SparseAdd(dMtdP[ipar].data(), parentIndexes[idj], dMtdP[idj].data(), 3 * smpl::HandModel::NUM_JOINTS);
		    SparseAdd(dMtdc[ipar].data(), parentIndexes[idj], dMtdc[idj].data(), 3 * smpl::HandModel::NUM_JOINTS);
		    // reset dSdc
		    dSdc.data()[3 * idj] = 0;

// if (idji == 20)
// {
// std::cout << "(0, 0)\n" << MR[idj](0, 0) << std::endl;
// std::cout << "(0, 0)\n" << dMRdP[idj].row(0) << std::endl;
// std::cout << "(0, 0)\n" << dMRdc[idj].row(0) << std::endl;
// std::cout << "(0, 1)\n" << MR[idj](0, 1) << std::endl;
// std::cout << "(0, 1)\n" << dMRdP[idj].row(1) << std::endl;
// std::cout << "(0, 1)\n" << dMRdc[idj].row(1) << std::endl;
// std::cout << "(0, 2)\n" << MR[idj](0, 2) << std::endl;
// std::cout << "(0, 2)\n" << dMRdP[idj].row(2) << std::endl;
// std::cout << "(0, 2)\n" << dMRdc[idj].row(2) << std::endl;
// std::cout << "(1, 0)\n" << MR[idj](1, 0) << std::endl;
// std::cout << "(1, 0)\n" << dMRdP[idj].row(3) << std::endl;
// std::cout << "(1, 0)\n" << dMRdc[idj].row(3) << std::endl;
// std::cout << "(1, 1)\n" << MR[idj](1, 1) << std::endl;
// std::cout << "(1, 1)\n" << dMRdP[idj].row(4) << std::endl;
// std::cout << "(1, 1)\n" << dMRdc[idj].row(4) << std::endl;
// std::cout << "(1, 2)\n" << MR[idj](1, 2) << std::endl;
// std::cout << "(1, 2)\n" << dMRdP[idj].row(5) << std::endl;
// std::cout << "(1, 2)\n" << dMRdc[idj].row(5) << std::endl;
// std::cout << "(2, 0)\n" << MR[idj](2, 0) << std::endl;
// std::cout << "(2, 0)\n" << dMRdP[idj].row(6) << std::endl;
// std::cout << "(2, 0)\n" << dMRdc[idj].row(6) << std::endl;
// std::cout << "(2, 1)\n" << MR[idj](2, 1) << std::endl;
// std::cout << "(2, 1)\n" << dMRdP[idj].row(7) << std::endl;
// std::cout << "(2, 1)\n" << dMRdc[idj].row(7) << std::endl;
// std::cout << "(2, 2)\n" << MR[idj](2, 2) << std::endl;
// std::cout << "(2, 2)\n" << dMRdP[idj].row(8) << std::endl;
// std::cout << "(2, 2)\n" << dMRdc[idj].row(8) << std::endl;
// }
		}
	}

	for (int idji = 1; idji < handm_.NUM_JOINTS; idji++)
	{
		idj = handm_.m_jointmap_pm2model(idji);  // joint_order vs update_inds_ // joint_order(SMC -> idj)
		outJ.row(idj) = Mt.at(idj).transpose();

		if (dJdP_data && dJdc_data)
		{
		    std::copy(dMtdP[idj].data(), dMtdP[idj].data() + 3 * 3 * smpl::HandModel::NUM_JOINTS, dJdP_data + 3 * smpl::HandModel::NUM_JOINTS * 3 * idj);
		    std::copy(dMtdc[idj].data(), dMtdc[idj].data() + 3 * 3 * smpl::HandModel::NUM_JOINTS, dJdc_data + 3 * smpl::HandModel::NUM_JOINTS * 3 * idj);
		}
	}

	if (regressor_type == 1)
	{
		// compute LBS for joints apply regressor regressor
		// First Compute output transform,
		// Original:
		// outT.block(idj * 3, 0, 3, 4) = Ms.block(idj * 4, 0, 3, 4)*mod_.m_M_w2l.block(idj * 4, 0, 4, 4).cast<T>();
		for (int idji = 0; idji < handm_.NUM_JOINTS; idji++)
		{
			const int idj = handm_.update_inds_(idji);
			offset = handm_.m_M_w2l.block(idj * 4, 3, 3, 1);
			R3 = handm_.m_M_w2l.block(idj * 4, 0, 3, 3); 
			// first compute the derivative, in case the original variable changes
			if (dJdP_data && dJdc_data)
			{
				std::copy(dMRdP[idj].data(), dMRdP[idj].data() + 9 * 3 * smpl::HandModel::NUM_JOINTS, dR2dP.data());  // copy the jacobian to dR2dP
				std::copy(dMRdc[idj].data(), dMRdc[idj].data() + 9 * 3 * smpl::HandModel::NUM_JOINTS, dR2dc.data());  // copy the jacobian to dR2dc
				// dR3dP, dR3dc has always been zero
			    SparseProductDerivative(dMRdP[idj].data(), offset.data(), parentIndexes[idj], dtdP.data(), 3 * smpl::HandModel::NUM_JOINTS);
			    SparseProductDerivative(dMRdc[idj].data(), offset.data(), parentIndexes[idj], dtdc.data(), 3 * smpl::HandModel::NUM_JOINTS);
			    SparseAdd(dtdP.data(), parentIndexes[idj], dMtdP[idj].data(), 3 * smpl::HandModel::NUM_JOINTS);
			    SparseAdd(dtdc.data(), parentIndexes[idj], dMtdc[idj].data(), 3 * smpl::HandModel::NUM_JOINTS);
				SparseProductDerivative(MR.at(idj).data(), dR2dP.data(), R3.data(), dR3dP.data(), idj, parentIndexes[idj], dMRdP[idj].data(), 3 * smpl::HandModel::NUM_JOINTS);
				SparseProductDerivative(MR.at(idj).data(), dR2dc.data(), R3.data(), dR3dc.data(), idj, parentIndexes[idj], dMRdc[idj].data(), 3 * smpl::HandModel::NUM_JOINTS);
			}
			Mt.at(idj) = MR.at(idj) * offset + Mt.at(idj);
			MR.at(idj) = MR.at(idj) * R3;
		}
// std::cout << Mt[1] << "\n";
// std::cout << "dMtdP[1]\n" << dMtdP[1].row(0) << "\n";
// std::cout << "dMtdc[1]\n" << dMtdc[1].row(0) << "\n";

		// Second Perform LBS given output transformation
		MatrixXdr outVert(total_vertex.size(), 3);
		MatrixXdr dVdP(total_vertex.size() * 3, smpl::HandModel::NUM_POSE_PARAMETERS);
		MatrixXdr dVdc(total_vertex.size() * 3, smpl::HandModel::NUM_SHAPE_COEFFICIENTS);
		select_lbs(MR, Mt, dMRdP, dMRdc, dMtdP, dMtdc, outVert, dVdP.data(), dVdc.data());

		// Third, perform Sparse Regression
		// The STB regressor only contains the wrist
		// MatrixXdr outJ(1, 3);
		// MatrixXdr dJdP(3, smpl::HandModel::NUM_POSE_PARAMETERS);
		// MatrixXdr dJdc(3, smpl::HandModel::NUM_SHAPE_COEFFICIENTS);
		const int idj = handm_.update_inds_(0);
		SparseRegress(handm_.STB_wrist_reg,
					  outVert.data(),
					  dJdP_data? dVdP.data() : nullptr,
					  dJdP_data? dVdc.data() : nullptr,
					  outJ.data() + 3 * idj,
					  dJdP_data? dJdP_data + 3 * idj * smpl::HandModel::NUM_POSE_PARAMETERS : nullptr,
					  dJdP_data? dJdc_data + 3 * idj * smpl::HandModel::NUM_SHAPE_COEFFICIENTS : nullptr);
	}
}

void HandFastCost::select_lbs(
    std::vector<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>& MR,
    std::vector<Eigen::Matrix<double, 3, 1>>& Mt,
	std::vector<Eigen::Matrix<double, 9, 3 * smpl::HandModel::NUM_JOINTS, Eigen::RowMajor>>& dMRdP,
	std::vector<Eigen::Matrix<double, 9, 3 * smpl::HandModel::NUM_JOINTS, Eigen::RowMajor>>& dMRdc,
	std::vector<Eigen::Matrix<double, 3, 3 * smpl::HandModel::NUM_JOINTS, Eigen::RowMajor>>& dMtdP,
	std::vector<Eigen::Matrix<double, 3, 3 * smpl::HandModel::NUM_JOINTS, Eigen::RowMajor>>& dMtdc,
	MatrixXdr &outVert, double* dVdP_data, double* dVdc_data) const
{
	using namespace Eigen;
	assert(outVert.rows()== total_vertex.size());
    std::fill(dVdc_data, dVdc_data + 3 * total_vertex.size() * smpl::HandModel::NUM_SHAPE_COEFFICIENTS, 0); // dVdc.setZero();
    std::fill(dVdP_data, dVdP_data + 3 * total_vertex.size() * smpl::HandModel::NUM_POSE_PARAMETERS, 0); // dVdP.setZero();

    for (auto i = 0u; i < total_vertex.size(); i++)
    {
        const int idv = total_vertex[i];
        const auto* v0_data = handm_.V_.data() + 3 * idv;
        auto* outVrow_data = outVert.data() + 3 * i;
        outVrow_data[0] = outVrow_data[1] = outVrow_data[2] = 0;
        for (int idj = 0; idj < smpl::HandModel::NUM_JOINTS; idj++)
        {
        	const double w = handm_.W_(idv, idj);
        	if (w)
        	{
                outVrow_data[0] += w * (MR[idj].data()[0] * v0_data[0] + MR[idj].data()[1] * v0_data[1] + MR[idj].data()[2] * v0_data[2] + Mt[idj].data()[0]);
                outVrow_data[1] += w * (MR[idj].data()[3] * v0_data[0] + MR[idj].data()[4] * v0_data[1] + MR[idj].data()[5] * v0_data[2] + Mt[idj].data()[1]);
                outVrow_data[2] += w * (MR[idj].data()[6] * v0_data[0] + MR[idj].data()[7] * v0_data[1] + MR[idj].data()[8] * v0_data[2] + Mt[idj].data()[2]);

                int ncol = smpl::HandModel::NUM_POSE_PARAMETERS;
                double* dVdP_row0 = dVdP_data + (i * 3 + 0) * smpl::HandModel::NUM_POSE_PARAMETERS;
                double* dVdP_row1 = dVdP_data + (i * 3 + 1) * smpl::HandModel::NUM_POSE_PARAMETERS;
                double* dVdP_row2 = dVdP_data + (i * 3 + 2) * smpl::HandModel::NUM_POSE_PARAMETERS;
                for (int j = 0; j < parentIndexes[idj].size(); j++)
                {
                    const int idp = parentIndexes[idj][j];
                    dVdP_row0[3 * idp + 0] += w * (v0_data[0] * dMRdP[idj].data()[(0 * 3 + 0) * ncol + 3 * idp + 0] + v0_data[1] * dMRdP[idj].data()[(0 * 3 + 1) * ncol + 3 * idp + 0] + v0_data[2] * dMRdP[idj].data()[(0 * 3 + 2) * ncol + 3 * idp + 0] + dMtdP[idj].data()[0 * ncol + 3 * idp + 0]);
                    dVdP_row1[3 * idp + 0] += w * (v0_data[0] * dMRdP[idj].data()[(1 * 3 + 0) * ncol + 3 * idp + 0] + v0_data[1] * dMRdP[idj].data()[(1 * 3 + 1) * ncol + 3 * idp + 0] + v0_data[2] * dMRdP[idj].data()[(1 * 3 + 2) * ncol + 3 * idp + 0] + dMtdP[idj].data()[1 * ncol + 3 * idp + 0]);
                    dVdP_row2[3 * idp + 0] += w * (v0_data[0] * dMRdP[idj].data()[(2 * 3 + 0) * ncol + 3 * idp + 0] + v0_data[1] * dMRdP[idj].data()[(2 * 3 + 1) * ncol + 3 * idp + 0] + v0_data[2] * dMRdP[idj].data()[(2 * 3 + 2) * ncol + 3 * idp + 0] + dMtdP[idj].data()[2 * ncol + 3 * idp + 0]);
                    dVdP_row0[3 * idp + 1] += w * (v0_data[0] * dMRdP[idj].data()[(0 * 3 + 0) * ncol + 3 * idp + 1] + v0_data[1] * dMRdP[idj].data()[(0 * 3 + 1) * ncol + 3 * idp + 1] + v0_data[2] * dMRdP[idj].data()[(0 * 3 + 2) * ncol + 3 * idp + 1] + dMtdP[idj].data()[0 * ncol + 3 * idp + 1]);
                    dVdP_row1[3 * idp + 1] += w * (v0_data[0] * dMRdP[idj].data()[(1 * 3 + 0) * ncol + 3 * idp + 1] + v0_data[1] * dMRdP[idj].data()[(1 * 3 + 1) * ncol + 3 * idp + 1] + v0_data[2] * dMRdP[idj].data()[(1 * 3 + 2) * ncol + 3 * idp + 1] + dMtdP[idj].data()[1 * ncol + 3 * idp + 1]);
                    dVdP_row2[3 * idp + 1] += w * (v0_data[0] * dMRdP[idj].data()[(2 * 3 + 0) * ncol + 3 * idp + 1] + v0_data[1] * dMRdP[idj].data()[(2 * 3 + 1) * ncol + 3 * idp + 1] + v0_data[2] * dMRdP[idj].data()[(2 * 3 + 2) * ncol + 3 * idp + 1] + dMtdP[idj].data()[2 * ncol + 3 * idp + 1]);
                    dVdP_row0[3 * idp + 2] += w * (v0_data[0] * dMRdP[idj].data()[(0 * 3 + 0) * ncol + 3 * idp + 2] + v0_data[1] * dMRdP[idj].data()[(0 * 3 + 1) * ncol + 3 * idp + 2] + v0_data[2] * dMRdP[idj].data()[(0 * 3 + 2) * ncol + 3 * idp + 2] + dMtdP[idj].data()[0 * ncol + 3 * idp + 2]);
                    dVdP_row1[3 * idp + 2] += w * (v0_data[0] * dMRdP[idj].data()[(1 * 3 + 0) * ncol + 3 * idp + 2] + v0_data[1] * dMRdP[idj].data()[(1 * 3 + 1) * ncol + 3 * idp + 2] + v0_data[2] * dMRdP[idj].data()[(1 * 3 + 2) * ncol + 3 * idp + 2] + dMtdP[idj].data()[1 * ncol + 3 * idp + 2]);
                    dVdP_row2[3 * idp + 2] += w * (v0_data[0] * dMRdP[idj].data()[(2 * 3 + 0) * ncol + 3 * idp + 2] + v0_data[1] * dMRdP[idj].data()[(2 * 3 + 1) * ncol + 3 * idp + 2] + v0_data[2] * dMRdP[idj].data()[(2 * 3 + 2) * ncol + 3 * idp + 2] + dMtdP[idj].data()[2 * ncol + 3 * idp + 2]);
                }

                ncol = smpl::HandModel::NUM_SHAPE_COEFFICIENTS;
                double* dVdc_row0 = dVdc_data + (i * 3 + 0) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS;
                double* dVdc_row1 = dVdc_data + (i * 3 + 1) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS;
                double* dVdc_row2 = dVdc_data + (i * 3 + 2) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS;
                for (int j = 0; j < parentIndexes[idj].size(); j++)
                {
                    const int idp = parentIndexes[idj][j];
                    dVdc_row0[3 * idp + 0] += w * (v0_data[0] * dMRdc[idj].data()[(0 * 3 + 0) * ncol + 3 * idp + 0] + v0_data[1] * dMRdc[idj].data()[(0 * 3 + 1) * ncol + 3 * idp + 0] + v0_data[2] * dMRdc[idj].data()[(0 * 3 + 2) * ncol + 3 * idp + 0] + dMtdc[idj].data()[0 * ncol + 3 * idp + 0]);
                    dVdc_row1[3 * idp + 0] += w * (v0_data[0] * dMRdc[idj].data()[(1 * 3 + 0) * ncol + 3 * idp + 0] + v0_data[1] * dMRdc[idj].data()[(1 * 3 + 1) * ncol + 3 * idp + 0] + v0_data[2] * dMRdc[idj].data()[(1 * 3 + 2) * ncol + 3 * idp + 0] + dMtdc[idj].data()[1 * ncol + 3 * idp + 0]);
                    dVdc_row2[3 * idp + 0] += w * (v0_data[0] * dMRdc[idj].data()[(2 * 3 + 0) * ncol + 3 * idp + 0] + v0_data[1] * dMRdc[idj].data()[(2 * 3 + 1) * ncol + 3 * idp + 0] + v0_data[2] * dMRdc[idj].data()[(2 * 3 + 2) * ncol + 3 * idp + 0] + dMtdc[idj].data()[2 * ncol + 3 * idp + 0]);
                    dVdc_row0[3 * idp + 1] += w * (v0_data[0] * dMRdc[idj].data()[(0 * 3 + 0) * ncol + 3 * idp + 1] + v0_data[1] * dMRdc[idj].data()[(0 * 3 + 1) * ncol + 3 * idp + 1] + v0_data[2] * dMRdc[idj].data()[(0 * 3 + 2) * ncol + 3 * idp + 1] + dMtdc[idj].data()[0 * ncol + 3 * idp + 1]);
                    dVdc_row1[3 * idp + 1] += w * (v0_data[0] * dMRdc[idj].data()[(1 * 3 + 0) * ncol + 3 * idp + 1] + v0_data[1] * dMRdc[idj].data()[(1 * 3 + 1) * ncol + 3 * idp + 1] + v0_data[2] * dMRdc[idj].data()[(1 * 3 + 2) * ncol + 3 * idp + 1] + dMtdc[idj].data()[1 * ncol + 3 * idp + 1]);
                    dVdc_row2[3 * idp + 1] += w * (v0_data[0] * dMRdc[idj].data()[(2 * 3 + 0) * ncol + 3 * idp + 1] + v0_data[1] * dMRdc[idj].data()[(2 * 3 + 1) * ncol + 3 * idp + 1] + v0_data[2] * dMRdc[idj].data()[(2 * 3 + 2) * ncol + 3 * idp + 1] + dMtdc[idj].data()[2 * ncol + 3 * idp + 1]);
                    dVdc_row0[3 * idp + 2] += w * (v0_data[0] * dMRdc[idj].data()[(0 * 3 + 0) * ncol + 3 * idp + 2] + v0_data[1] * dMRdc[idj].data()[(0 * 3 + 1) * ncol + 3 * idp + 2] + v0_data[2] * dMRdc[idj].data()[(0 * 3 + 2) * ncol + 3 * idp + 2] + dMtdc[idj].data()[0 * ncol + 3 * idp + 2]);
                    dVdc_row1[3 * idp + 2] += w * (v0_data[0] * dMRdc[idj].data()[(1 * 3 + 0) * ncol + 3 * idp + 2] + v0_data[1] * dMRdc[idj].data()[(1 * 3 + 1) * ncol + 3 * idp + 2] + v0_data[2] * dMRdc[idj].data()[(1 * 3 + 2) * ncol + 3 * idp + 2] + dMtdc[idj].data()[1 * ncol + 3 * idp + 2]);
                    dVdc_row2[3 * idp + 2] += w * (v0_data[0] * dMRdc[idj].data()[(2 * 3 + 0) * ncol + 3 * idp + 2] + v0_data[1] * dMRdc[idj].data()[(2 * 3 + 1) * ncol + 3 * idp + 2] + v0_data[2] * dMRdc[idj].data()[(2 * 3 + 2) * ncol + 3 * idp + 2] + dMtdc[idj].data()[2 * ncol + 3 * idp + 2]);
                }
        	}
        }
    }
}

void HandFastCost::SparseRegress(const Eigen::SparseMatrix<double>& reg, const double* V_data, const double* dVdP_data, const double* dVdc_data,
                   double* J_data, double* dJdP_data, double* dJdc_data) const
{
	const int num_J = reg.rows();
	std::fill(J_data, J_data + 3 * num_J, 0);

    for (int ic = 0; ic < total_vertex.size(); ic++)
    {
	    const int c = total_vertex[ic];
	    for (Eigen::SparseMatrix<double>::InnerIterator it(reg, c); it; ++it)
	    {
	        const int r = it.row();
	        const double value = it.value();
	        J_data[3 * r + 0] += value * V_data[3 * ic + 0];
	        J_data[3 * r + 1] += value * V_data[3 * ic + 1];
	        J_data[3 * r + 2] += value * V_data[3 * ic + 2];
	    }
	}

    if (dVdP_data != nullptr)  // need to pass back the correct Jacobian
    {
        assert(dVdc_data != nullptr && dJdP_data != nullptr && dJdc_data != nullptr);
        std::fill(dJdP_data, dJdP_data + 3 * num_J * smpl::HandModel::NUM_POSE_PARAMETERS, 0.0);
        std::fill(dJdc_data, dJdc_data + 3 * num_J * smpl::HandModel::NUM_SHAPE_COEFFICIENTS, 0.0);
        for (int ic = 0; ic < total_vertex.size(); ic++)
        {
            const int c = total_vertex[ic];
            for (Eigen::SparseMatrix<double>::InnerIterator it(reg, c); it; ++it)
            {
                const int r = it.row();
                const double value = it.value();
                for (int i = 0; i < smpl::HandModel::NUM_POSE_PARAMETERS; i++)
                {
                    dJdP_data[(3 * r + 0) * smpl::HandModel::NUM_POSE_PARAMETERS + i] += value * dVdP_data[(3 * ic + 0) * smpl::HandModel::NUM_POSE_PARAMETERS + i];
                    dJdP_data[(3 * r + 1) * smpl::HandModel::NUM_POSE_PARAMETERS + i] += value * dVdP_data[(3 * ic + 1) * smpl::HandModel::NUM_POSE_PARAMETERS + i];
                    dJdP_data[(3 * r + 2) * smpl::HandModel::NUM_POSE_PARAMETERS + i] += value * dVdP_data[(3 * ic + 2) * smpl::HandModel::NUM_POSE_PARAMETERS + i];
                }
                for (int i = 0; i < smpl::HandModel::NUM_SHAPE_COEFFICIENTS; i++)
                {
                    dJdc_data[(3 * r + 0) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS + i] += value * dVdc_data[(3 * ic + 0) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS + i];
                    dJdc_data[(3 * r + 1) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS + i] += value * dVdc_data[(3 * ic + 1) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS + i];
                    dJdc_data[(3 * r + 2) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS + i] += value * dVdc_data[(3 * ic + 2) * smpl::HandModel::NUM_SHAPE_COEFFICIENTS + i];
                }
            }
        }
    }
}