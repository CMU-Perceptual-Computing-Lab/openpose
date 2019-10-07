#ifdef USE_3D_ADAM_MODEL
#ifndef OPENPOSE_GUI_W_GUI_ADAM_HPP
#define OPENPOSE_GUI_W_GUI_ADAM_HPP

#include <openpose/core/common.hpp>
#include <openpose/gui/guiAdam.hpp>
#include <openpose/thread/workerConsumer.hpp>

namespace op
{
    template<typename TDatums>
    class WGuiAdam : public WorkerConsumer<TDatums>
    {
    public:
        explicit WGuiAdam(const std::shared_ptr<GuiAdam>& guiAdam);

        virtual ~WGuiAdam();

        void initializationOnThread();

        void workConsumer(const TDatums& tDatums);

    private:
        std::shared_ptr<GuiAdam> spGuiAdam;

        DELETE_COPY(WGuiAdam);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WGuiAdam<TDatums>::WGuiAdam(const std::shared_ptr<GuiAdam>& guiAdam) :
        spGuiAdam{guiAdam}
    {
    }

    template<typename TDatums>
    WGuiAdam<TDatums>::~WGuiAdam()
    {
    }

    template<typename TDatums>
    void WGuiAdam<TDatums>::initializationOnThread()
    {
        try
        {
            spGuiAdam->initializationOnThread();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums>
    void WGuiAdam<TDatums>::workConsumer(const TDatums& tDatums)
    {
        try
        {
            // tDatums might be empty but we still wanna update the GUI
            if (tDatums != nullptr)
            {
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Update cvMat & keypoints
                if (!tDatums->empty())
                {
                    // Update cvMat
                    std::vector<Mat> cvOutputDatas;
                    for (auto& tDatum : *tDatums)
                        cvOutputDatas.emplace_back(tDatumPtr->cvOutputData);
                    spGuiAdam->setImage(cvOutputDatas);
                    // Update keypoints
                    const auto& tDatumPtr = (*tDatums)[0];
                    if (!tDatumPtr->poseKeypoints3D.empty())
                        spGuiAdam->generateMesh(
                            tDatumPtr->poseKeypoints3D, tDatumPtr->faceKeypoints3D, tDatumPtr->handKeypoints3D,
                            tDatumPtr->adamPose.data(), tDatumPtr->adamTranslation.data(), tDatumPtr->vtVec.data(),
                            tDatumPtr->vtVec.rows(), tDatumPtr->j0Vec.data(), tDatumPtr->j0Vec.rows(),
                            tDatumPtr->adamFaceCoeffsExp.data());
                }
                // Refresh/update GUI
                spGuiAdam->update();
                // Profiling speed
                if (!tDatums->empty())
                {
                    Profiler::timerEnd(profilerKey);
                    Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                }
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WGuiAdam);
}

#endif // OPENPOSE_GUI_W_GUI_ADAM_HPP
#endif
