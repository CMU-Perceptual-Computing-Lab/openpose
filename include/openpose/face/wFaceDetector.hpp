#ifndef OPENPOSE_FACE_W_FACE_EXTRACTOR_HPP
#define OPENPOSE_FACE_W_FACE_EXTRACTOR_HPP

#include <openpose/core/common.hpp>
#include <openpose/face/faceRenderer.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WFaceDetector : public Worker<TDatums>
    {
    public:
        explicit WFaceDetector(const std::shared_ptr<FaceDetector>& faceDetector);

        virtual ~WFaceDetector();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<FaceDetector> spFaceDetector;

        DELETE_COPY(WFaceDetector);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WFaceDetector<TDatums>::WFaceDetector(const std::shared_ptr<FaceDetector>& faceDetector) :
        spFaceDetector{faceDetector}
    {
    }

    template<typename TDatums>
    WFaceDetector<TDatums>::~WFaceDetector()
    {
    }

    template<typename TDatums>
    void WFaceDetector<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WFaceDetector<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Detect people face
                for (auto& tDatum : *tDatums)
                    tDatum.faceRectangles = spFaceDetector->detectFaces(tDatum.poseKeypoints);
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

    COMPILE_TEMPLATE_DATUM(WFaceDetector);
}

#endif // OPENPOSE_FACE_W_FACE_EXTRACTOR_HPP
