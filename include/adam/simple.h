#ifndef SIMPLE_H
#define SIMPLE_H

#include <Eigen/Sparse>
#include "cv.h"
#include <vector>
#include <utility>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXdr;

namespace smpl {
    struct SMPLModel {
        static const int NUM_SHAPE_COEFFICIENTS = 10;
        static const int NUM_VERTICES = 6890;
        static const int NUM_JOINTS = 24;
        static const int NUM_POSE_PARAMETERS = NUM_JOINTS * 3;

        // Template vertices (vector) <NUM_VERTICES*3, 1>
        Eigen::Matrix<double, Eigen::Dynamic, 1> mu_;

        // Shape basis, <NUM_FACE_POINTS*3, NUM_COEFFICIENTS>
        Eigen::Matrix<double, Eigen::Dynamic, NUM_SHAPE_COEFFICIENTS, Eigen::RowMajor> U_;

        // LBS weights,
        Eigen::Matrix<double, Eigen::Dynamic, NUM_JOINTS, Eigen::RowMajor> W_;

        // J_mu_ = J_reg_big_ * mu_
        Eigen::Matrix<double, NUM_JOINTS * 3, 1> J_mu_;
        // dJdc = J_reg_big_ * U_
        Eigen::Matrix<double, Eigen::Dynamic, NUM_SHAPE_COEFFICIENTS, Eigen::RowMajor> dJdc_;

        // Joint regressor, <NUM_JOINTS, NUM_VERTICES>
        Eigen::SparseMatrix<double, Eigen::RowMajor> J_reg_;
        
        // Joint regressor, <NUM_JOINTS*, NUM_VERTICES*3>  kron(J_reg_, eye(3))
        Eigen::SparseMatrix<double, Eigen::RowMajor> J_reg_big_;
        Eigen::SparseMatrix<double, Eigen::ColMajor> J_reg_big_col_;

        // Pose regressor, <NUM_VERTICES*3, (NUM_JOINTS-1)*9>
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pose_reg_;

        // Shape coefficient weights
        Eigen::Matrix<double, Eigen::Dynamic, 1> d_;

        // Triangle faces
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> faces_;

        // Triangle UV map
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> uv;       //UV texture coordinate (same number as vertex)

        // Kinematic tree
        Eigen::Matrix<int, 2, Eigen::Dynamic> kintree_table_;
        int parent_[NUM_JOINTS];
        int id_to_col_[NUM_JOINTS];

        // Correspondences between smpl and kinect joints
        Eigen::Matrix<int, Eigen::Dynamic, 1> indices_correspond_smpl;
        Eigen::Matrix<int, Eigen::Dynamic, 1> indices_correspond_kinect;

        // For CMU mocap joints
        Eigen::Matrix<int, Eigen::Dynamic, 1> indices_correspond_cmu_smpl;
        Eigen::Matrix<int, Eigen::Dynamic, 1> indices_correspond_cmu_cmu;
        Eigen::Matrix<int, Eigen::Dynamic, 1> indices_correspond_rotations_cmu_smpl;
        Eigen::Matrix<int, Eigen::Dynamic, 1> indices_correspond_rotations_cmu_cmu;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            CmuMocapJointTransforms_;

        //Merging part (hand and face) together
        Eigen::Matrix<int, Eigen::Dynamic, 1> ver_idx_rhand_full;
        Eigen::Matrix<int, Eigen::Dynamic, 1> ver_idx_rhand_ignore;
        Eigen::Matrix<int, Eigen::Dynamic, 1> ver_idx_rhand_overlap;

        Eigen::Matrix<int, Eigen::Dynamic, 1> ver_idx_lhand_full;
        Eigen::Matrix<int, Eigen::Dynamic, 1> ver_idx_lhand_ignore;
        Eigen::Matrix<int, Eigen::Dynamic, 1> ver_idx_lhand_overlap;

        Eigen::Matrix<int, Eigen::Dynamic, 1> ver_idx_face_full;

        //(Precomputed) Flags for visualization. Size should be the same as NUM_VERTICES
        std::vector<bool> flag_vertex_rhand_full;
        std::vector<bool> flag_vertex_rhand_ignore;
        std::vector<bool> flag_vertex_rhand_overlap;

        std::vector<bool> flag_vertex_lhand_full;
        std::vector<bool> flag_vertex_lhand_ignore;
        std::vector<bool> flag_vertex_lhand_overlap;

        std::vector<bool> flag_vertex_face_full;

        //Some utilities about joints
        Eigen::Matrix<int, Eigen::Dynamic, 1> indices_map_smc2smpl; //map from smc2smple  indices_map_smc2smpl[smcIDx] == smplIdx. only 15 joints
        std::vector<float> smpl_default_boneLength_smcOrder;        //only 15 joints. in Meter unit

        //For SMC joints
        Eigen::Matrix<int, Eigen::Dynamic, 1> indices_jointConst_smpl;  //correspondence between smpl and smc (not all joints are included)
        Eigen::Matrix<int, Eigen::Dynamic, 1> indices_jointConst_smc;

        Eigen::Matrix<int, Eigen::Dynamic, 1> indices_jointConst_torsoOnly_smpl;    //correspondence between smpl and smc (not all joints are included)
        Eigen::Matrix<int, Eigen::Dynamic, 1> indices_jointConst_torsoOnly_smc;

        Eigen::Matrix<int, Eigen::Dynamic, 1> indices_jointConst_smpl_icp;  //correspondence between smpl and smc (not all joints are included)
        Eigen::Matrix<int, Eigen::Dynamic, 1> indices_jointConst_smc_icp;

        // Correspondence matrices
        Eigen::SparseMatrix<double, Eigen::RowMajor> J_reg_cmu_big_;
        Eigen::SparseMatrix<double, Eigen::ColMajor> J_reg_cmu_big_col_;
        Eigen::VectorXd J_reg_cmu_weights;


        // Pose machine correspondences
        Eigen::SparseMatrix<double, Eigen::RowMajor> J_reg_big_SMC_;
        
        // Pose machine correspondences
        Eigen::SparseMatrix<double, Eigen::RowMajor> J_reg_pm_big_;
        Eigen::SparseMatrix<double, Eigen::ColMajor> J_reg_pm_big_col_;
        Eigen::VectorXd J_reg_pm_weights;

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> pose_prior_A;
        Eigen::Matrix<double, Eigen::Dynamic, 1> pose_prior_mu;
        Eigen::Matrix<double, Eigen::Dynamic, 1> pose_prior_b;

        // A model is fully specified by its coefficients, pose, and a translation
        Eigen::Matrix<double, NUM_SHAPE_COEFFICIENTS, 1> coeffs;
        Eigen::Matrix<double, NUM_JOINTS, 3, Eigen::RowMajor> pose;
        Eigen::Vector3d t;

        //Glue smpl to other parts
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> glue_rhand_reg;
        Eigen::Matrix<int, Eigen::Dynamic, 1> glue_rhand_smpl_veridx;
        Eigen::Matrix<int, Eigen::Dynamic, 1> glue_rhand_target_veridx;
        Eigen::Matrix<int, Eigen::Dynamic, 2> glue_rhand_nnmap_smpl2rhand;
        Eigen::Matrix<int, Eigen::Dynamic, 2> glue_lhand_nnmap_smpl2lhand;
        Eigen::Matrix<int, Eigen::Dynamic, 2> glue_face_nnmap_smpl2face;
                
        //Masks
        std::vector<bool> m_mask_deformable_ids_byTotalBlend;   //vertices which affect total model

        bool bInit;
        SMPLModel() {
            t.setZero();
            coeffs.setZero();
            pose.setZero();
            bInit = false;
        }
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    struct SMPLParams {
        // A model is fully specified by its coefficients, pose, and a translation
        bool m_bIsTotalModel;
        int m_bVisualized;  //true if "visualize unit" is already generated
        int frame;
        int humanIdx;
        double kin_frame;

        int m_visType;      //type1, type2... just for visualization

        //Adam Model
        //Eigen::Matrix<double, TotalModel::NUM_JOINTS, 3, Eigen::RowMajor> m_adam_pose;
        //Eigen::Matrix<double, TotalModel::NUM_SHAPE_COEFFICIENTS, 1> m_adam_coeffs;
        Eigen::Matrix<double, 62, 3, Eigen::RowMajor> m_adam_pose;  //62 ==TotalModel::NUM_JOINTS
        Eigen::Matrix<double, 30, 1> m_adam_coeffs;         //30 ==TotalModel::NUM_SHAPE_COEFFICIENTS
        Eigen::Vector3d m_adam_t;
        Eigen::VectorXd m_adam_facecoeffs_exp;
        //Eigen::VectorXd m_adam_facecoeffs_id;


        Eigen::Vector3d t;
        Eigen::Matrix<double, SMPLModel::NUM_JOINTS, 3, Eigen::RowMajor> pose;
        Eigen::Matrix<double, SMPLModel::NUM_SHAPE_COEFFICIENTS, 1> coeffs;
        double m_bodyScale;     //to handle child's motion

        Eigen::Vector3d face_t;
        Eigen::Vector3d face_rot;
        Eigen::Vector3d face_rot_pivot;
        Eigen::VectorXd face_coeffs_id;
        Eigen::VectorXd face_coeffs_exp;

        std::vector<cv::Point3d> m_vertexOffset;
        std::vector<cv::Point3d> m_vertexOffset_canonical;      //offset in canonical space (for fusion)
        std::vector<cv::Point3d> m_vertexOffset_face;
        std::vector<cv::Point3d> m_vertexOffset_total;
        std::vector<double> m_vertexOffset_normal;
        
        std::vector<double> m_vertexOffset_face_normVal;

        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> hand_coeffs;

        Eigen::Vector3d handr_t;
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> handr_pose;

        Eigen::Vector3d handl_t;
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> handl_pose;

        bool m_bValidBody;
        bool m_bValidFace;
        bool m_bValidRHand;
        bool m_bValidLHand;
        bool m_bValidAdam;

        // TODO refactor to remove hard coded sizes
        SMPLParams() {
            kin_frame = 0;
            humanIdx = -1;

            m_visType = 0;

            t.setZero();
            pose.setZero();
            coeffs.setZero();
            m_bodyScale = 1.0f;


            face_t.setZero();
            face_rot.setZero();
            face_rot_pivot.setZero();
            face_coeffs_id.resize(150, 1);
            face_coeffs_id.setZero();
            face_coeffs_exp.resize(200, 1);
            face_coeffs_exp.setZero();

            handr_t.setZero();
            handl_t.setZero();
            handr_pose.resize(21, 3);
            handr_pose.setZero();
            handl_pose.resize(21, 3);
            handl_pose.setZero();
            hand_coeffs.resize(21, 3);
            hand_coeffs.setConstant(1.0);

            m_adam_t.resize(3, 1);
            m_adam_t.setZero();
            m_adam_pose.resize(62, 3);
            m_adam_pose.setZero();
            m_adam_coeffs.resize(30, 1);  //30 ==TotalModel::NUM_SHAPE_COEFFICIENTS
            m_adam_coeffs.setZero();
            m_adam_facecoeffs_exp.resize(200, 1);
            m_adam_facecoeffs_exp.setZero();

            m_bValidAdam = m_bValidBody = m_bValidFace = m_bValidLHand = m_bValidRHand = false;

            m_bVisualized = false;

            m_bIsTotalModel = false;        //total model needs different way of visualization 
        }
    };

    void reconstruct_Eulers(const SMPLModel &mod,
        const double *coeffs,
        const double *pose_eulers,
        double *outVerts, //output
        Eigen::VectorXd &transforms);  //output

    void lbs(const SMPLModel &mod,
        const double *verts,
        const MatrixXdr& T,
        double *outVerts);

}

#endif