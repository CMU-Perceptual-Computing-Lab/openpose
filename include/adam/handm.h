#ifndef HANDM_H
#define HANDM_H

#include <Eigen/Sparse>
#include "simple.h"
#include <json/json.h>
#include <Eigen/Dense>

namespace smpl {
      
    struct HandModel {
        static const int NUM_SHAPE_COEFFICIENTS = 21*3;
        static const int NUM_VERTICES = 2068;
        static const int NUM_JOINTS = 21;
        static const int NUM_POSE_PARAMETERS = NUM_JOINTS*3;

        // Template vertices (vector) <NUM_VERTICES*3, 1>
        Eigen::Matrix<double, Eigen::Dynamic, 1> mu_;
        
        // Triangle faces
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> faces_;
        
        // Kinematic tree
        Eigen::Matrix<int, Eigen::Dynamic, 1> update_inds_;   //Ordered to trace from parents to children
        Eigen::Matrix<int, Eigen::Dynamic, 1> parents_;
        //Eigen::Matrix<int, Eigen::Dynamic, 1> m_jointmap_pm2model;
        Eigen::Matrix<int, Eigen::Dynamic, 1> m_jointmap_pm2model; // m_jointmap_pm2model(pm_idx) = modelIdx
        // 20    0    1    2    3    4   12   13    5    6   14   15    7    8   16   17    9   10   18   19   11
        
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> pose_prior_A_;
        Eigen::Matrix<double, Eigen::Dynamic, 1> pose_prior_mu_;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> coeffs_prior_A_;
        Eigen::Matrix<double, Eigen::Dynamic, 1> coeffs_prior_mu_;

        //Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor> Mta_; // Bind pose transforms. Original naming of Tomas
        Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor> m_M_l2pl; // Local coordinate to parent's local coordinate. The parent joint is at the origin
                                                                            //m_M_l2pl(root): Transform from local coordinate of the root (root joint at origin) 
                                                                            //to world coordinate in binding pose 
        //Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor> MTi_; // Bind pose inverse transforms (with parent). Original Naming of Tomas
        Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor> m_M_w2l; // World to finger joint local coordinate. The reference Joint is at the origin. 
                                                                          //The world here means the binding location

        //Total Model Alignment
        Eigen::Matrix<double, Eigen::Dynamic, 4> m_T_hand2smpl;
        Eigen::Matrix<double, Eigen::Dynamic, 4> m_M_rootlocal2smplW;
        
        Eigen::Matrix<double, Eigen::Dynamic, 2> pose_limits_; // 3*NUM_JOINTS, 2
                                                                               // Lower and upper bounds
        Eigen::Matrix<double, Eigen::Dynamic, 2> coeffs_limits_; // 3*NUM_JOINTS, 2
                                                                                 // Lower and upper bounds

        //Mesh Model
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V_;
        Eigen::Matrix<double, Eigen::Dynamic, 3> F_;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W_;

        Eigen::Matrix<double, Eigen::Dynamic, 2> uv;      //uv map
        Eigen::SparseMatrix<double> STB_wrist_reg;  // root regressor for STB dataset

        // A model is fully specified by its coefficients, pose, and a translation
        Eigen::Matrix<double, NUM_SHAPE_COEFFICIENTS, 1> coeffs;
        Eigen::Matrix<double, NUM_JOINTS, 3, Eigen::RowMajor> pose;
        Eigen::Vector3d t;
        
        bool m_bInit;

        HandModel() {
            t.setZero();
            coeffs.setZero();
            pose.setZero();
            m_bInit = false;
        }
        
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
    
    void LoadHandModelFromJson( HandModel &handm, const std::string &path );

    void reconstruct_joints_mesh(const HandModel &handm,
        const double *trans_,
        const double *coeffs,
        const double *pose,
        double *outJoints,
        double *out_v,
        MatrixXdr &dJdc,
        MatrixXdr &dJdP,
        const int regressor_type=0);

    void lbs_hand(const HandModel &handm, double* V, double* out_v);

    void reconstruct_joints(const HandModel &handm,
        const double *trans_,
        const double *coeffs,
        const double *pose,
        double *outJoints);

    // template<typename Derived, int rows, int cols> void initMatrix(Eigen::Matrix<Derived, rows, cols>& m, const Json::Value& value);
    // template<typename Derived, int rows, int cols, int option> void initMatrixRowMajor(Eigen::Matrix<Derived, rows, cols, option>& m, const Json::Value& value);

}

#endif