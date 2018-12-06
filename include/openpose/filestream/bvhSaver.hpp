#ifdef USE_3D_ADAM_MODEL
#ifndef OPENPOSE_FILESTREAM_BVH_SAVER_HPP
#define OPENPOSE_FILESTREAM_BVH_SAVER_HPP

#ifdef USE_3D_ADAM_MODEL
    #include <adam/totalmodel.h>
#endif
#include <openpose/core/common.hpp>

namespace op
{
    class OP_API BvhSaver
    {
    public:
        BvhSaver(const std::string bvhFilePath,
                 const std::shared_ptr<const TotalModel>& totalModel = nullptr,
                 const double fps = 30.);

        virtual ~BvhSaver();

        void initializationOnThread();

        void updateBvh(const Eigen::Matrix<double, 62, 3, Eigen::RowMajor>& adamPose,
                       const Eigen::Vector3d& adamTranslation,
                       const Eigen::Matrix<double, Eigen::Dynamic, 1>& j0Vec);


    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplBvhSaver;
        std::shared_ptr<ImplBvhSaver> spImpl;

        // PIMP requires DELETE_COPY & destructor, or extra code
        // http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html
        DELETE_COPY(BvhSaver);
    };
}

#endif // OPENPOSE_FILESTREAM_BVH_SAVER_HPP
#endif
