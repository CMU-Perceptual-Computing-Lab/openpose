#ifndef OPENPOSE_EXPERIMENTAL_3D_RENDERER_HPP
#define OPENPOSE_EXPERIMENTAL_3D_RENDERER_HPP

#include <openpose/core/common.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/thread/workerConsumer.hpp>

namespace op
{
    // This worker will do 3-D rendering
    class OP_API WRender3D : public WorkerConsumer<std::shared_ptr<std::vector<Datum>>>
    {
    public:
        WRender3D(const PoseModel poseModel = PoseModel::COCO_18);

        ~WRender3D();

        void initializationOnThread();

        void workConsumer(const std::shared_ptr<std::vector<Datum>>& datumsPtr);
    };
}

#endif // OPENPOSE_EXPERIMENTAL_3D_RENDERER_HPP
