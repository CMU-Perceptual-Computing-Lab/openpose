#ifndef OPENPOSE_CORE_W_KEYPOINT_SCALER_HPP
#define OPENPOSE_CORE_W_KEYPOINT_SCALER_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/keypointScaler.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WKeypointScaler : public Worker<TDatums>
    {
    public:
        explicit WKeypointScaler(const std::shared_ptr<KeypointScaler>& keypointScaler);

        virtual ~WKeypointScaler();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<KeypointScaler> spKeypointScaler;
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WKeypointScaler<TDatums>::WKeypointScaler(const std::shared_ptr<KeypointScaler>& keypointScaler) :
        spKeypointScaler{keypointScaler}
    {
    }

    template<typename TDatums>
    WKeypointScaler<TDatums>::~WKeypointScaler()
    {
    }

    template<typename TDatums>
    void WKeypointScaler<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WKeypointScaler<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Rescale pose data
                for (auto& tDatum : *tDatums)
                {
                    std::vector<Array<float>> arraysToScale{tDatum.poseKeypoints, tDatum.handKeypoints[0],
                                                            tDatum.handKeypoints[1], tDatum.faceKeypoints};
                    spKeypointScaler->scale(arraysToScale, tDatum.scaleInputToOutput, tDatum.scaleNetToOutput,
                                            Point<int>{tDatum.cvInputData.cols, tDatum.cvInputData.rows});
                    // Rescale part candidates
                    spKeypointScaler->scale(tDatum.poseCandidates, tDatum.scaleInputToOutput, tDatum.scaleNetToOutput,
                                            Point<int>{tDatum.cvInputData.cols, tDatum.cvInputData.rows});
                }
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            tDatums = nullptr;
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WKeypointScaler);
}

#endif // OPENPOSE_CORE_W_KEYPOINT_SCALER_HPP
