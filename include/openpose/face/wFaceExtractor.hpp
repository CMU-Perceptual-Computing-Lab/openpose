#ifndef OPENPOSE_FACE_W_FACE_DETECTOR_HPP
#define OPENPOSE_FACE_W_FACE_DETECTOR_HPP

#include <memory> // std::shared_ptr
#include <openpose/thread/worker.hpp>
#include "faceRenderer.hpp"

namespace op
{
    template<typename TDatums>
    class WFaceExtractor : public Worker<TDatums>
    {
    public:
        explicit WFaceExtractor(const std::shared_ptr<FaceExtractor>& faceExtractor);

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<FaceExtractor> spFaceExtractor;

        DELETE_COPY(WFaceExtractor);
    };
}





// Implementation
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/core/macros.hpp>
#include <openpose/utilities/pointerContainer.hpp>
#include <openpose/utilities/profiler.hpp>
namespace op
{
    template<typename TDatums>
    WFaceExtractor<TDatums>::WFaceExtractor(const std::shared_ptr<FaceExtractor>& faceExtractor) :
        spFaceExtractor{faceExtractor}
    {
    }

    template<typename TDatums>
    void WFaceExtractor<TDatums>::initializationOnThread()
    {
        spFaceExtractor->initializationOnThread();
    }

    template<typename TDatums>
    void WFaceExtractor<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Extract people face
                for (auto& tDatum : *tDatums)
                {
                    spFaceExtractor->forwardPass(tDatum.faceRectangles, tDatum.cvInputData, tDatum.scaleInputToOutput);
                    tDatum.faceKeypoints = spFaceExtractor->getFaceKeypoints();
                }
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__, Profiler::DEFAULT_X);
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

    COMPILE_TEMPLATE_DATUM(WFaceExtractor);
}

#endif // OPENPOSE_FACE_W_FACE_DETECTOR_HPP
