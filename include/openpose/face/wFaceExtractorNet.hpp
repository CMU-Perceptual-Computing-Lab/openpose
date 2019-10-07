#ifndef OPENPOSE_FACE_W_FACE_DETECTOR_NET_HPP
#define OPENPOSE_FACE_W_FACE_DETECTOR_NET_HPP

#include <openpose/core/common.hpp>
#include <openpose/face/faceRenderer.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WFaceExtractorNet : public Worker<TDatums>
    {
    public:
        explicit WFaceExtractorNet(const std::shared_ptr<FaceExtractorNet>& faceExtractorNet);

        virtual ~WFaceExtractorNet();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<FaceExtractorNet> spFaceExtractorNet;

        DELETE_COPY(WFaceExtractorNet);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WFaceExtractorNet<TDatums>::WFaceExtractorNet(const std::shared_ptr<FaceExtractorNet>& faceExtractorNet) :
        spFaceExtractorNet{faceExtractorNet}
    {
    }

    template<typename TDatums>
    WFaceExtractorNet<TDatums>::~WFaceExtractorNet()
    {
    }

    template<typename TDatums>
    void WFaceExtractorNet<TDatums>::initializationOnThread()
    {
        spFaceExtractorNet->initializationOnThread();
    }

    template<typename TDatums>
    void WFaceExtractorNet<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Extract people face
                for (auto& tDatumPtr : *tDatums)
                {
                    spFaceExtractorNet->forwardPass(tDatumPtr->faceRectangles, tDatumPtr->cvInputData);
                    tDatumPtr->faceHeatMaps = spFaceExtractorNet->getHeatMaps().clone();
                    tDatumPtr->faceKeypoints = spFaceExtractorNet->getFaceKeypoints().clone();
                }
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            tDatums = nullptr;
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WFaceExtractorNet);
}

#endif // OPENPOSE_FACE_W_FACE_DETECTOR_NET_HPP
