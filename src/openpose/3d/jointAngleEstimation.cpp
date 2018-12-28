#ifdef USE_3D_ADAM_MODEL
#ifdef USE_3D_ADAM_MODEL
    #include <adam/FitToBody.h>
    #include <adam/totalmodel.h>
#endif
#include <openpose/3d/jointAngleEstimation.hpp>

namespace op
{
    #ifdef USE_3D_ADAM_MODEL
        std::shared_ptr<TotalModel> sTotalModel;
        const int NUMBER_BODY_KEYPOINTS = 20;
        const int NUMBER_HAND_KEYPOINTS = 21;
        const int NUMBER_FACE_KEYPOINTS = 70;
        const int NUMBER_FOOT_KEYPOINTS = 3;
        // targetJoints: Only for Body, LHand, RHand. No Face, no Foot
        const int NUMBER_KEYPOINTS = 3*(NUMBER_BODY_KEYPOINTS + 2*NUMBER_HAND_KEYPOINTS);

        const std::shared_ptr<const TotalModel> loadTotalModel(const std::string& mObjectPath,
                                                               const std::string& mGTotalModelPath,
                                                               const std::string& mPcaPath,
                                                               const std::string& mCorrespondencePath)
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
        #ifdef USE_3D_ADAM_MODEL
            // Adam model files
            const std::string mGTotalModelPath;
            const std::string mPcaPath;
            const std::string mObjectPath;
            const std::string mCorrespondencePath;

            // Processing
            const bool mReturnJacobian;
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

            ImplJointAngleEstimation(const bool returnJacobian) :
                mGTotalModelPath{"./model/adam_v1_plus2.json"},
                mPcaPath{"./model/adam_blendshapes_348_delta_norm.json"},
                mObjectPath{"./model/mesh_nofeet.obj"},
                mCorrespondencePath{"./model/correspondences_nofeet.txt"},
                mReturnJacobian{returnJacobian},
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

    JointAngleEstimation::JointAngleEstimation(const bool returnJacobian)
        #ifdef USE_3D_ADAM_MODEL
            : spImpl{std::make_shared<ImplJointAngleEstimation>(returnJacobian)}
        #endif
    {
        try
        {
            // error("JointAngleEstimation (`ik_threads` flag) buggy and not working yet, but we are working on it!"
            //       " No coming soon...", __LINE__, __FUNCTION__, __FILE__);
            #ifndef USE_3D_ADAM_MODEL
                error("OpenPose must be compiled with the `USE_3D_ADAM_MODEL` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void JointAngleEstimation::~JointAngleEstimation()
    {
    }

    void JointAngleEstimation::initializationOnThread()
    {
    }

    void JointAngleEstimation::adamFastFit(Eigen::Matrix<double, 62, 3, Eigen::RowMajor>& adamPose,
                                           Eigen::Vector3d& adamTranslation,
                                           Eigen::Matrix<double, Eigen::Dynamic, 1>& vtVec,
                                           Eigen::Matrix<double, Eigen::Dynamic, 1>& j0Vec,
                                           Eigen::VectorXd& adamFacecoeffsExp,
                                           const Array<float>& poseKeypoints3D,
                                           const Array<float>& faceKeypoints3D,
                                           const std::array<Array<float>, 2>& handKeypoints3D)
    {
        try
        {
            // Sanity check
            if (!poseKeypoints3D.empty() && poseKeypoints3D.getSize(1) != 19 && poseKeypoints3D.getSize(1) != 25
                 && poseKeypoints3D.getSize(1) != 65)
                error("Only working for BODY_19 or BODY_25 or BODY_65 (#parts = "
                      + std::to_string(poseKeypoints3D.getSize(2)) + ").",
                      __LINE__, __FUNCTION__, __FILE__);
            // Shorter naming
            auto& frameParams = spImpl->mFrameParams;
            // If keypoints detected
            if (!poseKeypoints3D.empty())
            {
                // Reset to 0 all keypoints - Otherwise undefined behavior when fitting
                // It must be done on every iteration, otherwise errors, e.g., if face
                // was detected in frame i-1 but not in i
                spImpl->mBodyJoints.setZero();
                spImpl->mFaceJoints.setZero();
                spImpl->mLHandJoints.setZero();
                spImpl->mRHandJoints.setZero();
                spImpl->mLFootJoints.setZero();
                spImpl->mRFootJoints.setZero();
                // Update body
                for (auto part = 0 ; part < 19; part++)
                    updateKeypoint(spImpl->mBodyJoints,
                                   &poseKeypoints3D[{0, part, 0}],
                                   mapOPToAdam(part));
                // Update left/right hand
                if (poseKeypoints3D.getSize(1) == 65)
                {
                    // Wrists
                    updateKeypoint(spImpl->mLHandJoints,
                                   &poseKeypoints3D[{0, 7, 0}],
                                   0);
                    updateKeypoint(spImpl->mRHandJoints,
                                   &poseKeypoints3D[{0, 4, 0}],
                                   0);
                    // Left
                    for (auto part = 0 ; part < 20; part++)
                        updateKeypoint(spImpl->mLHandJoints,
                                       &poseKeypoints3D[{0, part+25, 0}],
                                       part+1);
                    // Right
                    for (auto part = 0 ; part < 20; part++)
                        updateKeypoint(spImpl->mRHandJoints,
                                       &poseKeypoints3D[{0, part+25+20, 0}],
                                       part+1);
                }
                else
                {
                    for (auto hand = 0u ; hand < handKeypoints3D.size(); hand++)
                        if (!handKeypoints3D.at(hand).empty())
                            for (auto part = 0 ; part < handKeypoints3D[hand].getSize(1); part++)
                                updateKeypoint((hand == 0 ? spImpl->mLHandJoints : spImpl->mRHandJoints),
                                               &handKeypoints3D[hand][{0, part, 0}],
                                               part);
                }
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
                if (!handKeypoints3D.at(0).empty() || poseKeypoints3D.getSize(1) == 65)
                    spImpl->mLHandJoints *= 1e2;
                if (!handKeypoints3D.at(1).empty() || poseKeypoints3D.getSize(1) == 65)
                    spImpl->mRHandJoints *= 1e2;
                if (!faceKeypoints3D.empty())
                    spImpl->mFaceJoints *= 1e2;
                spImpl->mLFootJoints *= 1e2;
                spImpl->mRFootJoints *= 1e2;

                // Initialization (e.g., first frame)
                const bool fastVersion = false;
                const bool freezeMissing = true;
                const bool ceresDisplayReport = false;
                // Fill Datum
                if (!spImpl->mInitialized || !fastVersion)
                {
                    if (!spImpl->mInitialized)
                    {
                        frameParams.m_adam_t(0) = spImpl->mBodyJoints(0, 2);
                        frameParams.m_adam_t(1) = spImpl->mBodyJoints(1, 2);
                        frameParams.m_adam_t(2) = spImpl->mBodyJoints(2, 2);
                        frameParams.m_adam_pose(0, 0) = 3.14159265358979323846264338327950288419716939937510582097494459;
                        spImpl->mInitialized = true;
                    }
                    // We make T-pose start with:
                    // 1. Root translation similar to current 3-d location of the mid-hip
                    // 2. x-orientation = 180, i.e., person standing up & looking to the camera
                    // 3. Because otherwise, if we call Adam_FastFit_Initialize twice (e.g., if a new person appears),
                    // it would use the latest ones from the last Adam_FastFit
                    // Fit initialization
                    // Adam_FastFit_Initialize only changes frameParams
                    const auto multistageFitting = true;
                    const auto handEnabled = !handKeypoints3D[0].empty() || !handKeypoints3D[1].empty()
                        || poseKeypoints3D.getSize(1) == 65;
                    const auto fitFaceExponents = !faceKeypoints3D.empty();
                    const auto fastSolver = true;
                    Adam_FastFit_Initialize(*spImpl->spTotalModel, frameParams, spImpl->mBodyJoints, spImpl->mRFootJoints,
                                            spImpl->mLFootJoints, spImpl->mRHandJoints, spImpl->mLHandJoints,
                                            spImpl->mFaceJoints, freezeMissing, ceresDisplayReport,
                                            multistageFitting, handEnabled, fitFaceExponents, fastSolver);
                    // The following 2 operations takes ~12 msec
                    if (spImpl->mReturnJacobian)
                    {
                        vtVec = spImpl->spTotalModel->m_meanshape
                              + spImpl->spTotalModel->m_shapespace_u * frameParams.m_adam_coeffs;
                        j0Vec = spImpl->spTotalModel->J_mu_ + spImpl->spTotalModel->dJdc_ * frameParams.m_adam_coeffs;
                        if (fastVersion)
                        {
                            spImpl->mVtVec = vtVec;
                            spImpl->mJ0Vec = j0Vec;
                        }
                    }
                }
                // Other frames if fastVersion
                else // if (spImpl->mInitialized && fastVersion)
                {
                    // Adam_FastFit only changes frameParams
                    Adam_FastFit(*spImpl->spTotalModel, frameParams, spImpl->mBodyJoints, spImpl->mRFootJoints,
                                 spImpl->mLFootJoints, spImpl->mRHandJoints, spImpl->mLHandJoints,
                                 spImpl->mFaceJoints, ceresDisplayReport);
                    if (spImpl->mReturnJacobian)
                    {
                        vtVec = spImpl->mVtVec;
                        j0Vec = spImpl->mJ0Vec;
                    }
                }
                adamPose = frameParams.m_adam_pose;
                adamTranslation = frameParams.m_adam_t;
                adamFacecoeffsExp = frameParams.m_adam_facecoeffs_exp;
                // // Not used anymore
                // frameParams.mouth_open, frameParams.reye_open, frameParams.leye_open, frameParams.dist_root_foot
            }
            else
                spImpl->mInitialized = false;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
#endif
