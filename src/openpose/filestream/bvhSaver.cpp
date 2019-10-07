#ifdef USE_3D_ADAM_MODEL
#include <openpose/filestream/bvhSaver.hpp>
#ifdef USE_3D_ADAM_MODEL
    #include <adam/BVHWriter.h>
#endif

namespace op
{
    struct BvhSaver::ImplBvhSaver
    {
        #ifdef USE_3D_ADAM_MODEL
            // Write BVH file
            const std::string mBvhFilePath;
            const double mFps;
            std::unique_ptr<BVHWriter> spBvhWriter;
            // Record the translation across frames
            std::vector<Eigen::Matrix<double, 3, 1>> mTranslations;
            // Record the pose change
            std::vector<Eigen::Matrix<double, TotalModel::NUM_JOINTS, 3, Eigen::RowMajor>> mPoses;
            Eigen::Matrix<double, Eigen::Dynamic, 1> mJ0VecFrame0;
            bool mInitialized;

            // Shared parameters
            const std::shared_ptr<const TotalModel> spTotalModel;

            ImplBvhSaver(const std::string bvhFilePath, const std::shared_ptr<const TotalModel>& totalModel,
                         const double fps) :
                mBvhFilePath{bvhFilePath},
                mFps{fps},
                mInitialized{false},
                spTotalModel{totalModel}
            {
                try
                {
                    // Sanity check
                    if (!mBvhFilePath.empty() && spTotalModel == nullptr)
                        error("Given totalModel is a nullptr.", __LINE__, __FUNCTION__, __FILE__);
                }
                catch (const std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                }
            }

            void writeBvhOnDisk()
            {
                try
                {
                    if (!mBvhFilePath.empty())
                    {
                        const auto secondsPerFrame = 1./mFps;
                        const bool unityCompatible = true;
                        spBvhWriter.reset(new BVHWriter{spTotalModel->m_parent, unityCompatible});
                        spBvhWriter->parseInput(mJ0VecFrame0, mTranslations, mPoses);
                        spBvhWriter->writeBVH(mBvhFilePath, secondsPerFrame);
                    }
                }
                catch (const std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                }
            }
        #endif
    };

    BvhSaver::BvhSaver(const std::string bvhFilePath, const std::shared_ptr<const TotalModel>& totalModel,
                       const double fps) :
        spImpl{std::make_shared<ImplBvhSaver>(bvhFilePath, totalModel, fps)}
    {
        try
        {
            #ifndef USE_3D_ADAM_MODEL
                UNUSED(bvhFilePath);
                UNUSED(totalModel);
                UNUSED(fps);
                error("OpenPose CMake must be compiled with the `USE_3D_ADAM_MODEL` flag in order to use the"
                      " Adam visualization renderer. Alternatively, set 2-D/3-D rendering with `--display 2`"
                      " or `--display 3`.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    BvhSaver::~BvhSaver()
    {
        try
        {
            #ifdef USE_3D_ADAM_MODEL
                spImpl->writeBvhOnDisk();
            #endif
        }
        catch (const std::exception& e)
        {
            errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void BvhSaver::initializationOnThread()
    {
    }

    void BvhSaver::updateBvh(const Eigen::Matrix<double, 62, 3, Eigen::RowMajor>& adamPose,
                             const Eigen::Vector3d& adamTranslation,
                             const Eigen::Matrix<double, Eigen::Dynamic, 1>& j0Vec)
    {
        try
        {
            #ifdef USE_3D_ADAM_MODEL
                // BVH-Unity generation
                spImpl->mPoses.push_back(adamPose);
                spImpl->mTranslations.push_back(adamTranslation);
                if (!spImpl->mInitialized)
                {
                    spImpl->mJ0VecFrame0 = j0Vec;
                    spImpl->mInitialized = true;
                }
            #else
                UNUSED(adamPose);
                UNUSED(adamTranslation);
                UNUSED(j0Vec);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}

#endif
