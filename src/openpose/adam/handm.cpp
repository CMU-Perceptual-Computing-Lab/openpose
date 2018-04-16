#include <handm.h>
#include <stdio.h>
#include <json/json.h>
#include <iostream>
#include <fstream>
#include <typeinfo>
#include <string.h>
#include <Eigen/Dense>
#include "ceres/ceres.h"
#include "pose_to_transforms.h"
// #include<igl/cat.h>
// #include<igl/matlab/MatlabWorkspace.h>

namespace smpl{
    const int HandModel::NUM_SHAPE_COEFFICIENTS;
    const int HandModel::NUM_VERTICES;
    const int HandModel::NUM_JOINTS;
    const int HandModel::NUM_POSE_PARAMETERS;



template<typename Derived, int rows, int cols>
void initMatrix(Eigen::Matrix<Derived, rows, cols>& m, const Json::Value& value)
{
    if(m.cols() == 1) { // a vector
        m.resize(value.size(), 1);
        for (uint i = 0; i < value.size(); i++)
        {
            if (strcmp(typeid(Derived).name(), "i") == 0) // the passed in matrix is Int
                m(i, 0) = value[i].asInt();
            else // the passed in matrix is should be double
                m(i, 0) = value[i].asDouble();
        }
    }
    else  { // a matrix
        m.resize(value.size(), value[0u].size());
        for (uint i = 0; i < value.size(); i++)
            for (uint j = 0; j < value[i].size(); j++)
            {
                if (strcmp(typeid(Derived).name(), "i") == 0)
                    m(i, j) = value[i][j].asInt();
                else
                    m(i, j) = value[i][j].asDouble();
            }
    }
    // std::cout << m << std::endl;
}

template<typename Derived, int rows, int cols, int option>
void initMatrixRowMajor(Eigen::Matrix<Derived, rows, cols, option>& m, const Json::Value& value)
{
    if(m.cols() == 1) { // a vector
        m.resize(value.size(), 1);
        for (uint i = 0; i < value.size(); i++)
        {
            if (strcmp(typeid(Derived).name(), "i") == 0) // the passed in matrix is Int
                m(i, 0) = value[i].asInt();
            else // the passed in matrix is should be double
                m(i, 0) = value[i].asDouble();
        }
    }
    else  { // a matrix
        m.resize(value.size(), value[0u].size());
        for (uint i = 0; i < value.size(); i++)
            for (uint j = 0; j < value[i].size(); j++)
            {
                if (strcmp(typeid(Derived).name(), "i") == 0)
                    m(i, j) = value[i][j].asInt();
                else
                    m(i, j) = value[i][j].asDouble();
            }
    }
    // std::cout << m << std::endl;
}

void LoadHandModelFromJson( HandModel &handm, const std::string &path ) 
{
    printf("Loading from: %s\n", path.c_str());
    Json::Value root;
    std::ifstream file(path.c_str(), std::ifstream::in);
    file >> root;
    initMatrix(handm.update_inds_, root["update_inds"]);
    initMatrix(handm.parents_, root["parents"]);
    initMatrix(handm.m_jointmap_pm2model, root["joint_order"]);
    initMatrix(handm.pose_limits_, root["pose_limits"]);
    initMatrix(handm.coeffs_limits_, root["coeffs_limits"]);
    initMatrixRowMajor(handm.m_M_l2pl, root["MTa"]);
    initMatrixRowMajor(handm.m_M_w2l, root["MTi"]);
    initMatrix(handm.pose_prior_A_, root["pose_prior_A"]);
    initMatrix(handm.pose_prior_mu_, root["pose_prior_mu"]);
    initMatrix(handm.coeffs_prior_A_, root["coeffs_prior_A"]);
    initMatrix(handm.coeffs_prior_mu_, root["coeffs_prior_mu"]);
    initMatrixRowMajor(handm.V_, root["V"]);
    // initMatrixRowMajor(handm.m_T_hand2smpl, root["T_hand2smpl"]);
    int handRootJoint = handm.update_inds_(0);
    handm.m_M_rootlocal2smplW = handm.m_T_hand2smpl * handm.m_M_w2l.block(handRootJoint * 4, 0, 4, 4).inverse(); //Precomputing.
    initMatrix(handm.uv, root["texcoord"]);
    initMatrix(handm.W_, root["W"]);
    Eigen::Matrix<double, Eigen::Dynamic, 4> F_quad;
    initMatrix(F_quad, root["F"]);
    handm.F_ = Eigen::Matrix<double, Eigen::Dynamic, 3>(F_quad.rows()*2, 3);

    file.close();

    //convert quad mesh to triangles
    for (int r = 0; r < F_quad.rows(); ++r)
    {
        handm.F_(2 * r, 0) = F_quad(r, 0) -1; //Face index is saved by 1-based 
        handm.F_(2 * r, 1) = F_quad(r, 1) - 1;
        handm.F_(2 * r, 2) = F_quad(r, 2) - 1;

        handm.F_(2 * r +1, 0) = F_quad(r, 2) - 1;
        handm.F_(2 * r +1, 1) = F_quad(r, 3) - 1;
        handm.F_(2 * r +1, 2) = F_quad(r, 0) - 1;
    }

    handm.m_bInit = true;
}

// Reconstruct shape with pose & coefficients (no translation)
void reconstruct_joints_mesh(const HandModel &handm,
    const double *trans_,
    const double *coeffs,
    const double *pose,
    double *outJoints,
    double *out_v,
    MatrixXdr &dJdc,
    MatrixXdr &dJdP)
{
    using namespace Eigen;
    Map< const Matrix<double, 3, 1> > Trans(trans_);
    Map< const Matrix<double, Dynamic, 1> > c(coeffs, HandModel::NUM_SHAPE_COEFFICIENTS);
    Map< const Matrix<double, Dynamic, Dynamic> > p(pose, HandModel::NUM_JOINTS,3);


    Map< Matrix<double, Dynamic, Dynamic, RowMajor> >
        outJ(outJoints, HandModel::NUM_JOINTS, 3);
    dJdc.resize(HandModel::NUM_JOINTS * 3, 3 * HandModel::NUM_JOINTS);
    dJdP.resize(HandModel::NUM_JOINTS * 3, 3 * HandModel::NUM_JOINTS);
    const int num_t = (HandModel::NUM_JOINTS) * 3 * 4 * 2;      //note *2
    Matrix<double, Dynamic, 3 * HandModel::NUM_JOINTS, RowMajor> dTdc(num_t, 3 * HandModel::NUM_JOINTS);
    Matrix<double, Dynamic, 3 * HandModel::NUM_JOINTS, RowMajor> dTdP(num_t, 3 * HandModel::NUM_JOINTS);

    VectorXd transformsNJoints(3 * HandModel::NUM_JOINTS * 4 * 2);      //Transform + Joints
    Map< Matrix<double, 3 * HandModel::NUM_JOINTS, 4, RowMajor> > Tf(transformsNJoints.data());
    Map< Matrix<double, 3 * HandModel::NUM_JOINTS, 4, RowMajor> > Joints(transformsNJoints.data() + 3 * HandModel::NUM_JOINTS * 4);

    ceres::AutoDiffCostFunction<PoseToTransformsHand,
        (HandModel::NUM_JOINTS) * 3 * 4 * 2,        //Transform +  joints
        (HandModel::NUM_JOINTS) * 3,
        (HandModel::NUM_JOINTS) * 3> p2t(new PoseToTransformsHand(handm));
    // ForwardKinematics forward(handm);
    const double * parameters[2] = { coeffs, pose };
    double * residuals = transformsNJoints.data();
    double * jacobians[2] = { dTdc.data(), dTdP.data() };
    p2t.Evaluate(parameters, residuals, jacobians);
    // forward.forward(coeffs, pose, residuals);

    // std::cout << "Joints\n" << Joints << std::endl;

    for (int idji = 0; idji < HandModel::NUM_JOINTS; idji++)
    {
        int idj = handm.m_jointmap_pm2model(idji);      //mine -> matlab
        outJ(idji, 0) = Joints(idj * 3 + 0, 3);
        outJ(idji, 1) = Joints(idj * 3 + 1, 3);
        outJ(idji, 2) = Joints(idj * 3 + 2, 3);

        if (trans_ != NULL)
        {
            outJ(idji, 0) += Trans(0, 0);
            outJ(idji, 1) += Trans(1, 0);
            outJ(idji, 2) += Trans(2, 0);
        }

        // //Jacobian out
        // for (int idi = 0; idi < 3 * HandModel::NUM_JOINTS; idi++)
        // {
        //     dJdP(idji * 3 + 0, idi) = dTdP((idj * 3 + 0) * 4 + 3, idi);
        //     dJdP(idji * 3 + 1, idi) = dTdP((idj * 3 + 1) * 4 + 3, idi);
        //     dJdP(idji * 3 + 2, idi) = dTdP((idj * 3 + 2) * 4 + 3, idi);
        //     dJdc(idji * 3 + 0, idi) = dTdc((idj * 3 + 0) * 4 + 3, idi);
        //     dJdc(idji * 3 + 1, idi) = dTdc((idj * 3 + 1) * 4 + 3, idi);
        //     dJdc(idji * 3 + 2, idi) = dTdc((idj * 3 + 2) * 4 + 3, idi);
        // }
    }

    // std::cout << "Trans\n" << Trans << std::endl;
    // std::cout << "Coffs\n" << p << std::endl;
    // std::cout << "Pose\n" << c << std::endl;
    std::cout << "outJoints\n" << outJ << std::endl;
    lbs_hand(handm, transformsNJoints.data(), out_v);
    Map< Matrix<double, HandModel::NUM_VERTICES, 3, RowMajor> > outV(out_v);

    if (trans_ != NULL)
    {
        // Map< Matrix<double, HandModel::NUM_VERTICES, 3, RowMajor> > outV(out_v);
        for (int r = 0; r < outV.rows(); ++r)
        {
            outV(r, 0) += Trans(0, 0);
            outV(r, 1) += Trans(1, 0);
            outV(r, 2) += Trans(2, 0);
        }
    }
  //Transforming Vertices
}

void lbs_hand(const HandModel &handm, double* T, double* out_v_)
{
    using namespace Eigen;
    Map< Matrix<double, 3 * HandModel::NUM_JOINTS, 4, RowMajor> > Tv(T);
    Map< Matrix<double, HandModel::NUM_VERTICES, 3, RowMajor> > outV(out_v_);

    Map< const Matrix<double, Dynamic, Dynamic, RowMajor> >
        Vs(handm.V_.data(), HandModel::NUM_VERTICES, 3);

    for (int idv = 0; idv<HandModel::NUM_VERTICES; idv++)
    {
        outV(idv, 0) = 0;
        outV(idv, 1) = 0;
        outV(idv, 2) = 0;
        for (int idj = 0; idj<HandModel::NUM_JOINTS; idj++)
        {
            if (handm.W_(idv, idj))
            {
                double w = handm.W_(idv, idj);
                for (int idd = 0; idd<3; idd++)
                {
                    outV(idv, idd) += w*Vs(idv, 0)*Tv(idj * 3 * 4 + idd * 4 + 0);
                    outV(idv, idd) += w*Vs(idv, 1)*Tv(idj * 3 * 4 + idd * 4 + 1);
                    outV(idv, idd) += w*Vs(idv, 2)*Tv(idj * 3 * 4 + idd * 4 + 2);
                    outV(idv, idd) += w*Tv(idj * 3 * 4 + idd * 4 + 3);

                }
            }
        }
    }
}

void reconstruct_joints(const HandModel &handm,
    const double *trans_,
    const double *coeffs,
    const double *pose,
    double *outJoints)
{
    using namespace Eigen;
    Map< const Matrix<double, 3, 1> > Trans(trans_);
    Map< const Matrix<double, Dynamic, 1> > c(coeffs, HandModel::NUM_SHAPE_COEFFICIENTS);
    Map< const Matrix<double, Dynamic, Dynamic> > p(pose, HandModel::NUM_JOINTS,3);

    Map< Matrix<double, Dynamic, Dynamic, RowMajor> >
        outJ(outJoints, HandModel::NUM_JOINTS, 3);

    VectorXd transformsNJoints(3 * HandModel::NUM_JOINTS * 4 * 2);      //Transform + Joints
    Map< Matrix<double, 3 * HandModel::NUM_JOINTS, 4, RowMajor> > Tf(transformsNJoints.data());
    Map< Matrix<double, 3 * HandModel::NUM_JOINTS, 4, RowMajor> > Joints(transformsNJoints.data() + 3 * HandModel::NUM_JOINTS * 4);

    ceres::AutoDiffCostFunction<PoseToTransformsHand,
        (HandModel::NUM_JOINTS) * 3 * 4 * 2,        //Transform +  joints
        (HandModel::NUM_JOINTS) * 3,
        (HandModel::NUM_JOINTS) * 3> p2t(new PoseToTransformsHand(handm));
    // ForwardKinematics forward(handm);
    const double * parameters[2] = { coeffs, pose };
    double * residuals = transformsNJoints.data();
    p2t.Evaluate(parameters, residuals, NULL);

    for (int idji = 0; idji < HandModel::NUM_JOINTS; idji++)
    {
        int idj = handm.m_jointmap_pm2model(idji);      //mine -> matlab
        outJ(idji, 0) = Joints(idj * 3 + 0, 3);
        outJ(idji, 1) = Joints(idj * 3 + 1, 3);
        outJ(idji, 2) = Joints(idj * 3 + 2, 3);

        if (trans_ != NULL)
        {
            outJ(idji, 0) += Trans(0, 0);
            outJ(idji, 1) += Trans(1, 0);
            outJ(idji, 2) += Trans(2, 0);
        }

    }
}

}