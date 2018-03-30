#include <simple.h>
#include "ceres/ceres.h"
#include "pose_to_transforms.h"

namespace smpl {
    const int SMPLModel::NUM_SHAPE_COEFFICIENTS;
    const int SMPLModel::NUM_VERTICES;
    const int SMPLModel::NUM_JOINTS;
    const int SMPLModel::NUM_POSE_PARAMETERS;

    void reconstruct_Eulers(const SMPLModel &smpl,
        const double *parm_coeffs,
        const double *parm_pose_eulers,
        double *outVerts,
        Eigen::VectorXd &transforms)
    {
        using namespace Eigen;
        Map< const Matrix<double, Dynamic, 1> > c(parm_coeffs, SMPLModel::NUM_SHAPE_COEFFICIENTS);

        Matrix<double, Dynamic, Dynamic, RowMajor> Vt(SMPLModel::NUM_VERTICES, 3);
        Map< Matrix<double, Dynamic, 1> > Vt_vec(Vt.data(), 3 * SMPLModel::NUM_VERTICES);

        Map< Matrix<double, Dynamic, Dynamic, RowMajor> >
            outV(outVerts, SMPLModel::NUM_VERTICES, 3);

        Vt_vec = smpl.mu_ + smpl.U_*c;

        Matrix<double, SMPLModel::NUM_JOINTS, 3, RowMajor> J;
        Map< Matrix<double, Dynamic, 1> > J_vec(J.data(), SMPLModel::NUM_JOINTS * 3);
        J_vec = smpl.J_mu_ + smpl.dJdc_*c;

        const int num_t = (SMPLModel::NUM_JOINTS) * 3 * 4;
        Matrix<double, Dynamic, 3 * SMPLModel::NUM_JOINTS, RowMajor> dTdP(num_t, 3 * SMPLModel::NUM_JOINTS);
        Matrix<double, Dynamic, 3 * SMPLModel::NUM_JOINTS, RowMajor> dTdJ(num_t, 3 * SMPLModel::NUM_JOINTS);
        //VectorXd transforms(3 * SMPLModel::NUM_JOINTS * 4);
        transforms.resize(3 * SMPLModel::NUM_JOINTS * 4);

        //Timer ts;
        ceres::AutoDiffCostFunction<PoseToTransformsNoLR_Eulers,
            (SMPLModel::NUM_JOINTS) * 3 * 4,
            (SMPLModel::NUM_JOINTS) * 3,
            (SMPLModel::NUM_JOINTS) * 3> p2t(new PoseToTransformsNoLR_Eulers(smpl));
        const double * parameters[2] = { parm_pose_eulers, J.data() };
        double * residuals = transforms.data();
        double * jacobians[2] = { dTdP.data(), dTdJ.data() };
        p2t.Evaluate(parameters, residuals, jacobians);     //automatically compute residuals and jacobians (dTdP and dTdJ)
                                                            //      std::cout << "P2T: " <<  ts.elapsed() << "\n";
                                                            //ts.reset();

        Matrix<double, Dynamic, SMPLModel::NUM_SHAPE_COEFFICIENTS, RowMajor> dTdc = dTdJ*smpl.dJdc_;
        lbs(smpl, Vt_vec.data(), transforms, outVerts);     //dVdc and dVdP are final output by using dTdP and dTdc
    }

    void lbs(const SMPLModel &smpl,
        const double *verts,
        const MatrixXdr& T,
        double *outVerts)       //output
    {
        using namespace Eigen;
        Map< const Matrix<double, Dynamic, Dynamic, RowMajor> >
            Vs(verts, SMPLModel::NUM_VERTICES, 3);
        Map< Matrix<double, Dynamic, Dynamic, RowMajor> >
            outV(outVerts, SMPLModel::NUM_VERTICES, 3);

        Map< const VectorXd > Tv(T.data(), T.rows()*T.cols());
        for (int idv = 0; idv<SMPLModel::NUM_VERTICES; idv++) {
            outV(idv, 0) = 0;
            outV(idv, 1) = 0;
            outV(idv, 2) = 0;
            for (int idj = 0; idj<SMPLModel::NUM_JOINTS; idj++) {
                if (smpl.W_(idv, idj)) {
                    double w = smpl.W_(idv, idj);
                    for (int idd = 0; idd<3; idd++) {
                        outV(idv, idd) += w*Vs(idv, 0)*Tv(idj * 3 * 4 + idd * 4 + 0);
                        outV(idv, idd) += w*Vs(idv, 1)*Tv(idj * 3 * 4 + idd * 4 + 1);
                        outV(idv, idd) += w*Vs(idv, 2)*Tv(idj * 3 * 4 + idd * 4 + 2);
                        outV(idv, idd) += w*Tv(idj * 3 * 4 + idd * 4 + 3);
                    }
                }
            }
        }
    }
}