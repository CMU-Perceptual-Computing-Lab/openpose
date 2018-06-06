#include <simple.h>
#include "ceres/ceres.h"
#include "pose_to_transforms.h"
#include <string>
#include <json/json.h>
#include <fstream>
#include <unsupported/Eigen/KroneckerProduct>

template<typename Derived, int rows, int cols>
void initMatrix(Eigen::Matrix<Derived, rows, cols>& m, const Json::Value& value)
{
    if(m.cols() == 1) { // a vector
        m.resize(value.size(), 1);
        auto* m_data = m.data();
        if (strcmp(typeid(Derived).name(), "i") == 0) // the passed in matrix is Int
            for (uint i = 0; i < value.size(); i++)
                m_data[i] = value[i].asInt();
        else // the passed in matrix should be double
            for (uint i = 0; i < value.size(); i++)
                m_data[i] = value[i].asDouble();
    }
    else  { // a matrix (column major)
        const int nrow = value.size(), ncol = value[0u].size();
        m.resize(nrow, ncol);
        auto* m_data = m.data();
        if (strcmp(typeid(Derived).name(), "i") == 0)
            for (uint i = 0; i < value.size(); i++)
                for (uint j = 0; j < value[i].size(); j++)
                    m_data[j * nrow + i] = value[i][j].asInt();
        else
            for (uint i = 0; i < value.size(); i++)
                for (uint j = 0; j < value[i].size(); j++)
                    m_data[j * nrow + i] = value[i][j].asDouble();
    }
    std::cout << "rows " << m.rows() << " cols " << m.cols() << std::endl;
}

template<typename Derived, int rows, int cols, int option>
void initRowMajorMatrix(Eigen::Matrix<Derived, rows, cols, option>& m, const Json::Value& value)
{
    if(m.cols() == 1) { // a vector
        m.resize(value.size(), 1);
        auto* m_data = m.data();
        if (strcmp(typeid(Derived).name(), "i") == 0) // the passed in matrix is Int
            for (uint i = 0; i < value.size(); i++)
                m_data[i] = value[i].asInt();
        else // the passed in matrix should be double
            for (uint i = 0; i < value.size(); i++)
                m_data[i] = value[i].asDouble();
    }
    else  { // a matrix (column major)
        const int nrow = value.size(), ncol = value[0u].size();
        m.resize(nrow, ncol);
        auto* m_data = m.data();
        if (strcmp(typeid(Derived).name(), "i") == 0)
            for (uint i = 0; i < value.size(); i++)
                for (uint j = 0; j < value[i].size(); j++)
                    m_data[i * ncol + j] = value[i][j].asInt();
        else
            for (uint i = 0; i < value.size(); i++)
                for (uint j = 0; j < value[i].size(); j++)
                    m_data[i * ncol + j] = value[i][j].asDouble();
    }
    std::cout << "rows " << m.rows() << " cols " << m.cols() << std::endl;
}

template<typename Derived, int option>
void initSparseMatrix(Eigen::SparseMatrix<Derived, option>& m, const Json::Value& value)
{
    if (strcmp(typeid(Derived).name(), "i") == 0)
    {
        // The first row specifies the size of the sparse matrix
        m.resize(value[0u][0u].asInt(), value[0u][1u].asInt());
        for (uint k = 1u; k < value.size(); k++)
        {
            assert(value[k].size() == 3);
            // From the second row on, triplet correspond to matrix entries
            const int i = value[k][0u].asInt();
            const int j = value[k][1u].asInt();
            m.insert(i, j) = value[k][2u].asInt();
        }
    }
    else
    {
        // The first row specifies the size of the sparse matrix
        m.resize(value[0u][0u].asInt(), value[0u][1u].asInt());
        for (uint k = 1u; k < value.size(); k++)
        {
            assert(value[k].size() == 3);
            // From the second row on, triplet correspond to matrix entries
            const int i = value[k][0u].asInt();
            const int j = value[k][1u].asInt();
            m.insert(i, j) = value[k][2u].asDouble();
        }
    }
    std::cout << "rows " << m.rows() << " cols " << m.cols() << std::endl;
}

namespace smpl {
    const int SMPLModel::NUM_SHAPE_COEFFICIENTS;
    const int SMPLModel::NUM_VERTICES;
    const int SMPLModel::NUM_JOINTS;
    const int SMPLModel::NUM_POSE_PARAMETERS;
    const int SMPLModel::NUM_LSP_JOINTS;
    const int SMPLModel::NUM_COCO_JOINTS;

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

    void init_smpl(SMPLModel& smplmodel)
    {
        std::string model_path("./model/smpl.json");
        printf("Loading from: %s\n", model_path.c_str());
        std::ifstream file(model_path.c_str(), std::ifstream::in);
        Json::Value root;
        file >> root;
        file.close();

        initMatrix(smplmodel.mu_, root["v_template"]);
        initRowMajorMatrix(smplmodel.U_, root["shapedirs"]);
        initSparseMatrix(smplmodel.J_reg_, root["J_regressor"]);
        initRowMajorMatrix(smplmodel.W_, root["weights"]);
        initRowMajorMatrix(smplmodel.pose_reg_, root["posedirs"]);
        initSparseMatrix(smplmodel.J_reg_coco_, root["cocoplus_regressor"]);
        smplmodel.J_reg_lsp_ = smplmodel.J_reg_coco_.block(0, 0, 14, SMPLModel::NUM_JOINTS);
        smplmodel.J_reg_big_ = Eigen::kroneckerProduct(smplmodel.J_reg_, Eigen::Matrix<double, 3, 3>::Identity());
        smplmodel.J_mu_ = smplmodel.J_reg_big_ * smplmodel.mu_;
        smplmodel.dJdc_ = smplmodel.J_reg_big_ * smplmodel.U_;
    }
}