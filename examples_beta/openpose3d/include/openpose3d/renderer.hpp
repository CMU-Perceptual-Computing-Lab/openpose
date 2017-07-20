#ifndef OPENPOSE3D_RENDERER_HPP
#define OPENPOSE3D_RENDERER_HPP

#include <thread>
#include <vector>
#include <openpose/headers.hpp>
#include <openpose3d/datum3D.hpp>

// This worker will do 3-D rendering
class WRender3D : public op::WorkerConsumer<std::shared_ptr<std::vector<Datum3D>>>
{
public:
    WRender3D(const op::PoseModel poseModel = op::PoseModel::COCO_18);

    void initializationOnThread() {}

    void workConsumer(const std::shared_ptr<std::vector<Datum3D>>& datumsPtr);

private:
    std::thread mRenderThread;

    void visualizationThread();
};

#endif // OPENPOSE3D_RENDERER_HPP
