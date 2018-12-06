#ifndef OPENPOSE_GUI_W_GUI_3D_HPP
#define OPENPOSE_GUI_W_GUI_3D_HPP

#include <openpose/core/common.hpp>
#include <openpose/gui/gui3D.hpp>
#include <openpose/thread/workerConsumer.hpp>

namespace op
{
    // This worker will do 3-D rendering
    template<typename TDatums>
    class WGui3D : public WorkerConsumer<TDatums>
    {
    public:
        explicit WGui3D(const std::shared_ptr<Gui3D>& gui3D);

        virtual ~WGui3D();

        void initializationOnThread();

        void workConsumer(const TDatums& tDatums);

    private:
        std::shared_ptr<Gui3D> spGui3D;

        DELETE_COPY(WGui3D);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WGui3D<TDatums>::WGui3D(const std::shared_ptr<Gui3D>& gui3D) :
        spGui3D{gui3D}
    {
    }

    template<typename TDatums>
    WGui3D<TDatums>::~WGui3D()
    {
    }

    template<typename TDatums>
    void WGui3D<TDatums>::initializationOnThread()
    {
        try
        {
            spGui3D->initializationOnThread();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums>
    void WGui3D<TDatums>::workConsumer(const TDatums& tDatums)
    {
        try
        {
            // tDatums might be empty but we still wanna update the GUI
            if (tDatums != nullptr)
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Update cvMat & keypoints
                if (!tDatums->empty())
                {
                    // Update cvMat
                    std::vector<cv::Mat> cvOutputDatas;
                    for (auto& tDatum : *tDatums)
                        cvOutputDatas.emplace_back(tDatum.cvOutputData);
                    spGui3D->setImage(cvOutputDatas);
                    // Update keypoints
                    auto& tDatum = (*tDatums)[0];
                    spGui3D->setKeypoints(tDatum.poseKeypoints3D, tDatum.faceKeypoints3D, tDatum.handKeypoints3D[0],
                                          tDatum.handKeypoints3D[1]);
                }
                // Refresh/update GUI
                spGui3D->update();
                // Profiling speed
                if (!tDatums->empty())
                {
                    Profiler::timerEnd(profilerKey);
                    Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                }
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WGui3D);
}

#endif // OPENPOSE_GUI_W_GUI_3D_HPP
