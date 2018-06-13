#ifdef WITH_3D_ADAM_MODEL
    #include <adam/FitToBody.h>
#endif
#include <openpose/3d/jointAngleEstimation.hpp>

namespace op
{
    #ifdef WITH_3D_ADAM_MODEL
        std::shared_ptr<TotalModel> sTotalModel;
        const int NUMBER_BODY_KEYPOINTS = 20;
        const int NUMBER_HAND_KEYPOINTS = 21;
        const int NUMBER_FACE_KEYPOINTS = 70;
        const int NUMBER_FOOT_KEYPOINTS = 3;
        const int NUMBER_KEYPOINTS = 3*(NUMBER_BODY_KEYPOINTS + 2*NUMBER_HAND_KEYPOINTS); // targetJoints: Only for Body, LHand, RHand. No Face, no Foot

        const std::shared_ptr<const TotalModel> loadTotalModel(const std::string& mObjectPath, const std::string& mGTotalModelPath,
                                                               const std::string& mPcaPath, const std::string& mCorrespondencePath)
        {
            try
            {
                if (sTotalModel == nullptr)
                {
                    // Initialize model
                    sTotalModel = std::make_shared<TotalModel>();
                    // Load spTotalModel (model + data)
                    // ~100 milliseconds
                    LoadTotalModelFromObj(*sTotalModel, mObjectPath);
                    // ~25 seconds
                    LoadTotalDataFromJson(*sTotalModel, mGTotalModelPath, mPcaPath, mCorrespondencePath);
                }
                // Return result
                return sTotalModel;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return nullptr;
            }
        }

        void updateKeypoint(Eigen::MatrixXd& targetJoint, const float* const poseKeypoint3D, const int part)
        {
            try
            {
                // Keypoint found
                if (poseKeypoint3D[2] > 0.5)
                {
                    targetJoint(0, part) = poseKeypoint3D[0];
                    targetJoint(1, part) = poseKeypoint3D[1];
                    targetJoint(2, part) = poseKeypoint3D[2];
                }
                // Keypoint not found - Keep last known keypoint or 0
                // Implicitly done with the initial setZero()
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }
    #endif

    struct JointAngleEstimation::ImplJointAngleEstimation
    {
        #ifdef WITH_3D_ADAM_MODEL
            // Adam model files
            const std::string mGTotalModelPath;
            const std::string mPcaPath;
            const std::string mObjectPath;
            const std::string mCorrespondencePath;

            // Processing
            const bool mFillVtAndJ0Vecs;
            const bool mCeresDisplayReport;
            bool mInitialized;

            Eigen::MatrixXd mBodyJoints;
            Eigen::MatrixXd mFaceJoints;
            Eigen::MatrixXd mLHandJoints;
            Eigen::MatrixXd mRHandJoints;
            Eigen::MatrixXd mLFootJoints;
            Eigen::MatrixXd mRFootJoints;

            // Other parameters
            smpl::SMPLParams mFrameParams;

            // Shared parameters
            Eigen::Matrix<double, Eigen::Dynamic, 1> mVtVec;
            Eigen::Matrix<double, Eigen::Dynamic, 1> mJ0Vec;
            const std::shared_ptr<const TotalModel> spTotalModel;

            ImplJointAngleEstimation(const bool fillVtAndJ0Vecs,
                                     const bool ceresDisplayReport) :
                mGTotalModelPath{"./model/adam_v1_plus2.json"},
                mPcaPath{"./model/adam_blendshapes_348_delta_norm.json"},
                mObjectPath{"./model/mesh_nofeet.obj"},
                mCorrespondencePath{"./model/correspondences_nofeet.txt"},
                mFillVtAndJ0Vecs{fillVtAndJ0Vecs},
                mCeresDisplayReport{ceresDisplayReport},
                mInitialized{false},
                mBodyJoints(5, NUMBER_BODY_KEYPOINTS),
                mFaceJoints(5, NUMBER_FACE_KEYPOINTS),// (3, landmarks_face.size());
                mLHandJoints(5, NUMBER_HAND_KEYPOINTS),// (3, HandModel::NUM_JOINTS);
                mRHandJoints(5, NUMBER_HAND_KEYPOINTS),// (3, HandModel::NUM_JOINTS);
                mLFootJoints(5, 3),// (3, 3);        // Heel, Toe
                mRFootJoints(5, 3),// (3, 3);        // Heel, Toe
                spTotalModel{loadTotalModel(mObjectPath, mGTotalModelPath, mPcaPath, mCorrespondencePath)}
            {
            }
        #endif
    };

    int mapOPToAdam(const int oPPart)
    {
        if (oPPart >= 0 && oPPart < 19)
        {
            // Nose
            if (oPPart == 0)
                return 1;
            // Neck
            else if (oPPart == 1)
                return 0;
            // Right arm
            else if (oPPart == 2)
                return 9;
            else if (oPPart == 3)
                return 10;
            else if (oPPart == 4)
                return 11;
            // Left arm
            else if (oPPart == 5)
                return 3;
            else if (oPPart == 6)
                return 4;
            else if (oPPart == 7)
                return 5;
            // Mid-hip
            else if (oPPart == 8)
                return 2;
            // Right leg
            else if (oPPart == 9)
                return 12;
            else if (oPPart == 10)
                return 13;
            else if (oPPart == 11)
                return 14;
            // Left leg
            else if (oPPart == 12)
                return 6;
            else if (oPPart == 13)
                return 7;
            else if (oPPart == 14)
                return 8;
            // Face
            else if (oPPart == 15)
                return 17;
            else if (oPPart == 16)
                return 15;
            else if (oPPart == 17)
                return 18;
            else if (oPPart == 18)
                return 16;
            else
                error("Wrong body part (" + std::to_string(oPPart) + ").",
                          __LINE__, __FUNCTION__, __FILE__);
        }
        error("Wrong body part (" + std::to_string(oPPart) + ").",
                  __LINE__, __FUNCTION__, __FILE__);
        return -1;
    }

    const std::shared_ptr<const TotalModel> JointAngleEstimation::getTotalModel()
    {
        try
        {
            return sTotalModel;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    JointAngleEstimation::JointAngleEstimation(const bool fillVtAndJ0Vecs,
                                               const bool ceresDisplayReport)
        #ifdef WITH_3D_ADAM_MODEL
            : spImpl{std::make_shared<ImplJointAngleEstimation>(fillVtAndJ0Vecs, ceresDisplayReport)}
        #endif
    {
        try
        {
            // error("JointAngleEstimation (`ik_threads` flag) buggy and not working yet, but we are working on it!"
            //       " Coming soon!", __LINE__, __FUNCTION__, __FILE__);
            #ifdef WITH_3D_ADAM_MODEL
            #else
                UNUSED(fillVtAndJ0Vecs);
                UNUSED(ceresDisplayReport);
                error("OpenPose must be compiled with the `WITH_3D_ADAM_MODEL` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::tuple<Eigen::MatrixXd, Eigen::Vector3d, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd,
        float, float, float, float> JointAngleEstimation::runAdam(
            const Array<float>& poseKeypoints3D,
            const Array<float>& faceKeypoints3D,
            const std::array<Array<float>, 2>& handKeypoints3D)
    {
        try
        {
            // Security checks
            if (!poseKeypoints3D.empty() && poseKeypoints3D.getSize(1) != 19 && poseKeypoints3D.getSize(1) != 25)
                error("Only working for BODY_19 or BODY_25 (#parts = "
                      + std::to_string(poseKeypoints3D.getSize(2)) + ").",
                      __LINE__, __FUNCTION__, __FILE__);
            // Shorter naming
            auto& frameParams = spImpl->mFrameParams;
            // Reset to 0 all keypoints - Otherwise undefined behavior when fitting
            // It must be done on every iteration, otherwise errors, e.g., if face
            // was detected in frame i-1 but not in i
            spImpl->mBodyJoints.setZero();
            spImpl->mFaceJoints.setZero();
            spImpl->mLHandJoints.setZero();
            spImpl->mRHandJoints.setZero();
            spImpl->mLFootJoints.setZero();
            spImpl->mRFootJoints.setZero();
            // If keypoints detected
            if (!poseKeypoints3D.empty())
            {
                // Update body
                for (auto part = 0 ; part < 19; part++)
                    updateKeypoint(spImpl->mBodyJoints,
                                   &poseKeypoints3D[{0, part, 0}],
                                   mapOPToAdam(part));
                // Update left/right hand
                for (auto hand = 0u ; hand < handKeypoints3D.size(); hand++)
                    if (!handKeypoints3D.at(hand).empty())
                        for (auto part = 0 ; part < handKeypoints3D[hand].getSize(1); part++)
                            updateKeypoint((hand == 0 ? spImpl->mLHandJoints : spImpl->mRHandJoints),
                                           &handKeypoints3D[hand][{0, part, 0}],
                                           part);
                // Update Foot data
                if (poseKeypoints3D.getSize(1) == 25)
                {
                    // Update LFoot
                    for (auto adamPart = 0 ; adamPart < NUMBER_FOOT_KEYPOINTS; adamPart++)
                        updateKeypoint(spImpl->mLFootJoints,
                                       &poseKeypoints3D[{0, adamPart + 19, 0}],
                                       adamPart);
                    // Update RFoot
                    for (auto adamPart = 0 ; adamPart < NUMBER_FOOT_KEYPOINTS; adamPart++)
                        updateKeypoint(spImpl->mRFootJoints,
                                       &poseKeypoints3D[{0, adamPart + 19 + NUMBER_FOOT_KEYPOINTS, 0}],
                                       adamPart);
                }
                // Update Face data
                if (!faceKeypoints3D.empty())
                    for (auto part = 0 ; part < NUMBER_FACE_KEYPOINTS; part++)
                        updateKeypoint(spImpl->mFaceJoints,
                                       &faceKeypoints3D[{0, part, 0}],
                                       part);
                // Meters --> cm
                spImpl->mBodyJoints *= 1e2;
                if (!handKeypoints3D.at(0).empty())
                    spImpl->mLHandJoints *= 1e2;
                if (!handKeypoints3D.at(1).empty())
                    spImpl->mRHandJoints *= 1e2;
                if (!faceKeypoints3D.empty())
                    spImpl->mFaceJoints *= 1e2;
                spImpl->mLFootJoints *= 1e2;
                spImpl->mRFootJoints *= 1e2;
            }

            // Initialization (e.g., first frame)
            if (!spImpl->mInitialized)
            {
                spImpl->mInitialized = true;
                // We make T-pose start with:
                // 1. Root translation similar to current 3-d location of the mid-hip
                // 2. x-orientation = 180, i.e., person standing up & looking to the camera
                // 3. Because otherwise, if we call Adam_FastFit_Initialize twice (e.g., if a new person appears),
                // it would use the latest ones from the last Adam_FastFit
                frameParams.m_adam_t(0) = spImpl->mBodyJoints(0, 2);
                frameParams.m_adam_t(1) = spImpl->mBodyJoints(1, 2);
                frameParams.m_adam_t(2) = spImpl->mBodyJoints(2, 2);
                frameParams.m_adam_pose(0, 0) = 3.14159265358979323846264338327950288419716939937510582097494459;
                // Fit initialization
                // Adam_FastFit_Initialize only changes frameParams
                Adam_FastFit_Initialize(*spImpl->spTotalModel, frameParams, spImpl->mBodyJoints, spImpl->mRFootJoints, spImpl->mLFootJoints,
                                        spImpl->mRHandJoints, spImpl->mLHandJoints, spImpl->mFaceJoints, spImpl->mCeresDisplayReport);
                spImpl->mVtVec = spImpl->spTotalModel->m_meanshape + spImpl->spTotalModel->m_shapespace_u * frameParams.m_adam_coeffs;
                spImpl->mJ0Vec = spImpl->spTotalModel->J_mu_ + spImpl->spTotalModel->dJdc_ * frameParams.m_adam_coeffs;
            }
            // Other frames
            else
            {
                // // Fast way (it doesn't look right)
                // // Adam_FastFit only changes frameParams
                // Adam_FastFit(*spImpl->spTotalModel, frameParams, spImpl->mBodyJoints, spImpl->mRFootJoints, spImpl->mLFootJoints, spImpl->mRHandJoints,
                //              spImpl->mLHandJoints, spImpl->mFaceJoints, spImpl->mCeresDisplayReport);
                // Slow way
                // Adam_FastFit_Initialize only changes frameParams
                Adam_FastFit_Initialize(*spImpl->spTotalModel, frameParams, spImpl->mBodyJoints, spImpl->mRFootJoints, spImpl->mLFootJoints,
                                        spImpl->mRHandJoints, spImpl->mLHandJoints, spImpl->mFaceJoints, spImpl->mCeresDisplayReport);
                spImpl->mVtVec = spImpl->spTotalModel->m_meanshape + spImpl->spTotalModel->m_shapespace_u * frameParams.m_adam_coeffs;
                spImpl->mJ0Vec = spImpl->spTotalModel->J_mu_ + spImpl->spTotalModel->dJdc_ * frameParams.m_adam_coeffs;
            }
            // Fill Datum
            Eigen::MatrixXd vtVec;
            Eigen::MatrixXd j0Vec;
            // ~0.5 ms
            if (spImpl->mFillVtAndJ0Vecs)
            {
                vtVec = spImpl->mVtVec;
                j0Vec = spImpl->mJ0Vec;
            }
            return std::make_tuple(
                frameParams.m_adam_pose, frameParams.m_adam_t, vtVec, j0Vec, frameParams.m_adam_facecoeffs_exp,
                frameParams.mouth_open, frameParams.reye_open, frameParams.leye_open, frameParams.dist_root_foot
            );
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_tuple(Eigen::MatrixXd{}, Eigen::Vector3d{}, Eigen::MatrixXd{}, Eigen::MatrixXd{},
                                   Eigen::VectorXd{}, -1.f, -1.f, -1.f, -1.f);
        }
    }
}
