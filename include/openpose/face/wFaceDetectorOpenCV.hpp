#ifndef OPENPOSE_FACE_W_FACE_EXTRACTOR_OPENCV_HPP
#define OPENPOSE_FACE_W_FACE_EXTRACTOR_OPENCV_HPP

#include <openpose/core/common.hpp>
#include <openpose/face/faceRenderer.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WFaceDetectorOpenCV : public Worker<TDatums>
    {
    public:
        explicit WFaceDetectorOpenCV(const std::shared_ptr<FaceDetectorOpenCV>& faceDetectorOpenCV);

        virtual ~WFaceDetectorOpenCV();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<FaceDetectorOpenCV> spFaceDetectorOpenCV;

        DELETE_COPY(WFaceDetectorOpenCV);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WFaceDetectorOpenCV<TDatums>::WFaceDetectorOpenCV(const std::shared_ptr<FaceDetectorOpenCV>& faceDetectorOpenCV) :
        spFaceDetectorOpenCV{faceDetectorOpenCV}
    {
    }

    template<typename TDatums>
    WFaceDetectorOpenCV<TDatums>::~WFaceDetectorOpenCV()
    {
    }

    template<typename TDatums>
    void WFaceDetectorOpenCV<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WFaceDetectorOpenCV<TDatums>::work(TDatums& tDatums)
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
                    tDatum.faceRectangles = spFaceDetectorOpenCV->detectFaces(tDatum.cvInputData);
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

    COMPILE_TEMPLATE_DATUM(WFaceDetectorOpenCV);
}

#endif // OPENPOSE_FACE_W_FACE_EXTRACTOR_OPENCV_HPP
